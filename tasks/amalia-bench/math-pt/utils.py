import re
import datasets


def process_docs_no_figures(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Filter out examples that have figures in the question or solution.
    Note: Only works for ptpt_multiple_choice which has these fields.
    ptbr_multiple_choice doesn't have figure metadata, so this filter has no effect.
    """
    def _filter_doc(doc: dict) -> bool:
        # If fields don't exist, keep the document (ptbr case)
        has_question_fig = doc.get("question_has_figure", False)
        has_solution_fig = doc.get("solution_has_figure", False)
        return not has_question_fig and not has_solution_fig

    return dataset.filter(_filter_doc)


class ExtractAnswerMathPT:
    """
    Extracts the answer from the response and returns it for MATH-PT MCQ tasks.
    """

    def __init__(self):
        pass

    def apply(self, resps: list[list[str]], docs: list[dict]):

        def extract_answer(responses):

            filtered_responses = []
            for r in responses:
                """Extract the answer letter from the model's response."""
                # Look for \boxed{X} pattern
                boxed_match = re.search(r"\\boxed{([A-E])}", r, re.IGNORECASE)
                if boxed_match:
                    filtered_responses.append(boxed_match.group(1).upper())
                    break

                # Try to find patterns like "A resposta é A" or "Opção A é correta"
                answer_pattern = re.search(
                    r"(?:resposta correta|resposta certa|resposta verdadeira|resposta|answer|opção correta|opção certa|opção verdadeira|correct option).{0,10}([A-E])\b",
                    r,
                    re.IGNORECASE,
                )
                if answer_pattern:
                    filtered_responses.append(answer_pattern.group(1).upper())
                    break

                # Look for a letter at the end of the response
                final_letter = re.search(r"([A-E])\.?\s*$", r.strip())
                if final_letter:
                    filtered_responses.append(final_letter.group(1).upper())
                    break

                # If all else fails, try to find any letter A-E in the response
                any_letter = re.search(r"\b([A-E])\b", r)
                if any_letter:
                    filtered_responses.append(any_letter.group(1).upper())
                    break

                filtered_responses.append("[not extracted]")

            return filtered_responses

        filtered_resps = [extract_answer(resp) for resp in resps]

        return filtered_resps