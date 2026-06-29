import random
import datasets


def process_brvid_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Shuffles the answer choices for each document in the DSL-TL dataset.
    """

    def _process_doc(doc):
        choices = ['PT-PT', 'PT-BR']
        random.shuffle(choices)  # Seed the random generator

        choices_str = f"A. {choices[0]}\nB. {choices[1]}"
        # choices_arr = [choices[0], choices[1]]
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

    return dataset.map(_process_doc)
