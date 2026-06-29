from datasets import load_dataset
import json


def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]


# cleans poorly translated examples from the Winogrande dataset that are missing the need underscore
def clean_winogrande_translatation():

    x = load_dataset("amalia-llm/winogrande-mt-pt")

    for split in x:
        print(f"Processing split: {split}")
        examples = []
        counts = 0
        for i, doc in enumerate(x[split]):
            if "_" not in doc["sentence"]:
                print(doc)
                counts += 1
            else:
                examples.append(doc)

        # write to json file
        with open(f"winogrande_{split}_pt.jsonl", "w") as f:
            for doc in examples:
                f.write(json.dumps(doc) + "\n")

        print(f"Number of documents without underscore in {split}: {counts}")


if __name__ == "__main__":
    clean_winogrande_translatation()
