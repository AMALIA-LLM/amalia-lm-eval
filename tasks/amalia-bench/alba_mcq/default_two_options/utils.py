import random
import datasets


def process_alba_two_options(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Keeps only two options (the highest and lowest scored ones) from the ALBA dataset,
    and shuffles them randomly.
    """

    def _process_doc(doc):

        question = doc["question"]
        choices = doc["choices"]

        scores = doc["scores"]

        # get the index of the lowest and highest option in scores
        lower_index = scores.index(min(scores))
        higher_index = scores.index(max(scores))

        # randomly assign the lower score to one of the choices
        if random.random() < 0.5:
            shuffled_choices = [choices[lower_index], choices[higher_index]]
            final_scores = [scores[lower_index], scores[higher_index]]
            target_letter = 'B'
        else:
            shuffled_choices = [choices[higher_index], choices[lower_index]]
            final_scores = [scores[higher_index], scores[lower_index]]
            target_letter = 'A'


        text = f"{question.strip()}\nA. {shuffled_choices[0]}\nB. {shuffled_choices[1]}"

        out_doc = {
            "text": text,
            "label": target_letter,
            "choices": ['A', 'B'],
            "scores": final_scores,
            "correct_choice": "AB".index(target_letter),
        }

        return out_doc

    # set the seed
    random.seed(42)

    return dataset.map(_process_doc)
