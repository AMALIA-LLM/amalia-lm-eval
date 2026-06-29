import random
import datasets
import re


def process_pt_text_completion_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Shuffles the answer choices for each document in the PT completion dataset.
    """

    def _process_doc(doc):

        text = doc["text"]
        # replace ___* with a single _
        text = re.sub(r"_+", "_", text)  # simplify underscores

        pt_pt = doc["pt-pt"]
        pt_br = doc["pt-br"]

        choices = [pt_pt, pt_br]
        random.shuffle(choices)  # Seed the random generator

        choices_str = f"\nA. {choices[0]}\nB. {choices[1]}\n"
        choices_arr = ['A', 'B']

        # target letter is the one with the pt-pt version
        if choices[0] == pt_pt:
            target_letter = 'A'
        elif choices[0] == pt_br:
            target_letter = 'B'
        else:
            raise ValueError(f"No correspondence found")

        out_doc = {
            "text": text,
            "label": target_letter,
            "choices": choices_arr,
            "choices_str": choices_str,
        }

        # print(out_doc)

        return out_doc

    # set the seed
    random.seed(42)

    return dataset.map(_process_doc)
