import random
import datasets


def process_dsl_tl_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Shuffles the answer choices for each document in the DSL-TL dataset.
    """

    def _process_doc(doc):
        choices = ['PT-PT', 'PT-BR', 'N/A']
        random.shuffle(choices)  # Seed the random generator

        choices_str = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}"
        # choices_arr = [choices[0], choices[1], choices[2]]
        choices_arr = ['A', 'B', 'C']
        if doc['label'] == 0:
            target_letter = 'ABC'[choices.index('PT-PT')]
        elif doc['label'] == 1:
            target_letter = 'ABC'[choices.index('PT-BR')]
        else:
            target_letter = 'ABC'[choices.index('N/A')]

        out_doc = {
            "text": doc["text"],
            "label": target_letter,
            "choices": choices_arr,
            "choices_str": choices_str,
            # "target_letter": target_letter
        }

        # print(out_doc)

        return out_doc

    # set the seed
    random.seed(42)

    return dataset.map(_process_doc)


def process_dsl_tl_docs_no_na(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Shuffles the answer choices for each document in the DSL-TL dataset.
    """

    def _process_doc(doc):
        choices = ['PT-PT', 'PT-BR']
        random.shuffle(choices)  # Seed the random generator

        choices_str = f"A. {choices[0]}\nB. {choices[1]}"
        # choices_arr = [choices[0], choices[1], choices[2]]
        choices_arr = ['A', 'B']
        if doc['label'] == 0:
            target_letter = 'AB'[choices.index('PT-PT')]
        elif doc['label'] == 1:
            target_letter = 'AB'[choices.index('PT-BR')]
        else:
            raise ValueError(f"Invalid label: {doc['label']}")

        out_doc = {
            "text": doc["text"],
            "label": target_letter,
            "choices": choices_arr,
            "choices_str": choices_str,
            # "target_letter": target_letter
        }

        # print(out_doc)

        return out_doc

    # set the seed
    random.seed(42)

    dataset = dataset.filter(lambda doc: doc['label'] != 2)
    return dataset.map(_process_doc)




#########################################




from functools import partial
from datasets import load_dataset
import os, random


# Label 0 - PT-PT
# Label 1 - PT-BR
# Label 2 - PT

labels_dict = {
    0: "PT-PT",
    1: "PT-BR",
    2: "PT"
}


def format_nshot_example(example, including_answer=True):
    if including_answer:
        prompt = example["question"]
    else:
        prompt = "\nTexto: " + example["text"] + "\nResposta: "
    return prompt


doc_to_text = partial(format_nshot_example, including_answer=False)
fewshot_to_text = partial(format_nshot_example, including_answer=True)


def doc_to_target(example):
    return labels_dict[example["label"]]


def process_docs(dataset):
    return dataset.filter(lambda x: x["label"] != 2)



def process_for_fewshot(num_labels):
    examples = []
    dataset = load_dataset("amalia-llm/dsl_tl_fewshot", split=f"{num_labels}shot")
    data = dataset["examples"]
    for row in data:
        examples.append({"question": row, "target": ""})
    return examples


process_for_fewshot_2 = partial(process_for_fewshot, num_labels=2)
process_for_fewshot_3 = partial(process_for_fewshot, num_labels=3)