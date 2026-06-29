# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import itertools  # noqa: I001
import json
import random
from functools import cache

import datasets
import requests
from datasets import load_dataset
from sklearn.metrics import d2_log_loss_score
from tqdm import tqdm

from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS, get_tokenizer

CONFIG = {
    "tokens_to_generate": 32,
    # "template": """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
    # pt-pt version
    "template": """Responde à pergunta com base nos documentos fornecidos. Dá-me apenas a resposta e não escrevas mais nenhuma palavra.\n\nOs seguintes documentos são fornecidos.\n\n{context}\n\nResponde à pergunta com base nos documentos fornecidos. Dá-me apenas a resposta e não escrevas mais nenhuma palavra.\n\nPergunta: {query}""",
    # "answer_prefix": """Answer:""",
    "answer_prefix": """Resposta:""",
}
SEED = 42
TEMPLATE = CONFIG["template"]
# DOCUMENT_PROMPT = "Document {i}:\n{document}"
DOCUMENT_PROMPT = "Documento {i}:\n{document}"


@cache
def download_json(url) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data


"""
@cache
def read_squad(
    url="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
) -> tuple[list[dict], list[str]]:
    try:
        data = download_json(url)
    except Exception as e:
        print(f"Error downloading SQuAD dataset from {url}: {e}. Training local version.")
        # TODO just to get it to work when there is no internet connection
        with open("tasks/amalia-bench/ruler_en/dataset/dev-v2.0.json", "r") as f:
            data = json.load(f)

    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data["data"]:
        more_docs = [total_docs_dict[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for qas in p["qas"]:
                if not qas["is_impossible"]:
                    total_qas.append(
                        {
                            "query": qas["question"],
                            "outputs": [a["text"] for a in qas["answers"]],
                            "context": [total_docs_dict[p["context"]]],
                            "more_context": [
                                idx
                                for idx in more_docs
                                if idx != total_docs_dict[p["context"]]
                            ],
                        }
                    )

    return total_qas, total_docs
"""


# pt pt version of squad
# TODO translation of squad is pretty old so take these results with a grain of salt
def read_squad() -> tuple[list[dict], list[str]]:

    dataset = load_dataset("amalia-llm/squadv2.0-mt-pt", split="validation")

    # 1. Extract unique contexts to maintain your total_docs indexing
    # We use a dict to keep insertion order (Python 3.7+)
    total_docs = list({row["context"]: None for row in dataset}.keys())
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    # 2. Group document indices by title to recreate 'more_context'
    # In HF format, many rows share the same title and context
    title_to_doc_indices = {}
    for row in dataset:
        t = row["title"]
        idx = total_docs_dict[row["context"]]
        if t not in title_to_doc_indices:
            title_to_doc_indices[t] = set()
        title_to_doc_indices[t].add(idx)

    total_qas = []

    # 3. Process the flat rows
    for row in dataset:
        # SQuAD v2 uses 'answers' as a dict of lists in HF
        # Check for empty answers (equivalent to is_impossible in v2)
        if not row["answers"]["text"]:
            continue

        current_ctx_idx = total_docs_dict[row["context"]]
        all_docs_for_this_title = title_to_doc_indices[row["title"]]

        total_qas.append({
            "query": row["question"],
            "outputs": row["answers"]["text"],
            "context": [current_ctx_idx],
            "more_context": [
                idx for idx in all_docs_for_this_title
                if idx != current_ctx_idx
            ],
        })

    return total_qas, total_docs

"""
@cache
def read_hotpotqa(
    url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
) -> tuple[list[dict], list[str]]:

    try:
        data = download_json(url)
    except Exception as e:
        print(f"Error downloading HotpotQA dataset from {url}: {e}. Training local version.")
        # TODO just to get it to work when there is no internet connection
        with open("tasks/amalia-bench/ruler_en/dataset/hotpot_dev_distractor_v1.json", "r") as f:
            data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
            }
        )

    return total_qas, total_docs
"""


# pt-br version of hotpotqa taken from here https://github.com/i2ca/pt-hotpotqa and adapted to the original format
# TODO again take these results with a grain of salt since it is pt-br and not pt-pt, but it is the best we have for now

@cache
def read_hotpotqa() -> tuple[list[dict], list[str]]:

    with open("tasks/amalia-bench/ruler_pt/dataset/pt_hotpotqa_distractor_validation_transformed.json", "r") as f:
        data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
            }
        )

    return total_qas, total_docs


def generate_input_output(
    index: int, num_docs: int, qas: list[dict], docs: list[str]
) -> tuple[str, list[str]]:
    curr_q: str = qas[index]["query"]
    curr_a: list[str] = qas[index]["outputs"]
    curr_docs: list[int] = qas[index]["context"]
    curr_more: list[int] = qas[index].get("more_context", [])
    if num_docs < len(docs):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [
                i for i, d in enumerate(docs) if i not in curr_docs + curr_more
            ]
            all_docs = (
                curr_docs
                + curr_more
                + random.sample(
                    addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more))
                )
            )
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))

        all_docs = [docs[idx] for idx in all_docs]
    else:
        all_docs = docs

    random.Random(SEED).shuffle(all_docs)

    context = "\n\n".join(
        [DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)]
    )
    input_text = TEMPLATE.format(context=context, query=curr_q)
    return input_text, curr_a


def generate_samples(
    tokenizer,
    docs: list[str],
    qas: list[dict],
    max_seq_length: int,
    num_samples: int = 500,
    tokens_to_generate: int = 32,
    pre_samples: int = 0,
    incremental: int = 10,
    remove_newline_tab=False,
) -> list[dict]:
    write_jsons = []

    # Find the perfect num_docs
    num_docs = incremental
    total_tokens = 0

    while total_tokens + tokens_to_generate < max_seq_length:
        input_text, answer = generate_input_output(0, num_docs, qas=qas, docs=docs)
        total_tokens = len(tokenizer(input_text + f" {answer}").input_ids)

        if total_tokens + tokens_to_generate > max_seq_length:
            num_docs -= incremental
            break

        num_docs += incremental
        if num_docs > len(docs):
            num_docs = len(docs)
            break

    # print("Number of documents:", num_docs)

    skipped = 0

    for index in tqdm(
        range(num_samples), desc=f"Generating QA Samples | {max_seq_length}"
    ):
        used_docs = num_docs
        input_text = None

        while True:
            try:
                input_text, answer = generate_input_output(
                    index + pre_samples, used_docs, qas=qas, docs=docs
                )
                length = len(tokenizer(input_text).input_ids) + tokens_to_generate
                assert length <= max_seq_length
                break

            except AssertionError:
                if used_docs > incremental:
                    used_docs -= incremental
                else:
                    print(
                        f"[SKIP] Sample {index} cannot fit into max_seq_length="
                        f"{max_seq_length} even with used_docs={used_docs}"
                    )
                    skipped += 1
                    break

        if input_text is None:
            continue

        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )

        write_jsons.append(
            {
                "index": index,
                "input": input_text,
                "outputs": answer,
                "length": length,
                "max_length": max_seq_length,
                "gen_prefix": "Answer:",
            }
        )

    print(f"Skipped {skipped} samples for max_seq_length={max_seq_length}")
    return write_jsons


def get_dataset(pretrained, docs, qas, max_seq_length=None, **kwargs) -> list[dict]:

    tokenizer = get_tokenizer(pretrained)
    write_jsons = generate_samples(
        tokenizer=tokenizer,
        docs=docs,
        qas=qas,
        num_samples=500,
        tokens_to_generate=32,
        max_seq_length=max_seq_length,
    )
    return write_jsons


def get_qa_dataset(ds, **kwargs) -> dict[str, datasets.Dataset]:

    pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", None))

    print("Using tokenizer:", pretrained)
    print("Using kwargs", kwargs)

    if ds == "squad":
        qas, docs = read_squad()
    else:
        qas, docs = read_hotpotqa()
    df = (
        get_dataset(pretrained=pretrained, docs=docs, qas=qas, max_seq_length=seq)
        for seq in kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    )

    return {
        "test": datasets.Dataset.from_list(
            list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
        )
    }


def get_squad(**kwargs):
    return get_qa_dataset("squad", **kwargs)


def get_hotpotqa(**kwargs):
    return get_qa_dataset("hotpotqa", **kwargs)


"""
if __name__ == "__main__":

    model_location = 'Qwen/Qwen3-8B'

    c_args = {'version': 1.0, 'pretrained': model_location, 'max_model_len': 32768,
              'gpu_memory_utilization': 0.7, 'dtype': 'bfloat16', 'think_end_token': '</think>',
              'enable_thinking': False, 'trust_remote_code': True, 'max_seq_lengths': [4096, 8192, 16384, 32768]}

    get_hotpotqa(**c_args)
"""
