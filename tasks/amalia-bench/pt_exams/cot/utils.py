import re


class ExtractAnswerR:
    """
    Extracts the answer from the response and returns it.
    """

    def __init__(self):
        pass

    def apply(self, resps: list[list[str]], docs: list[dict]):

        # print(resps)
        # print(docs)

        def extract_answer(responses):

            filtered_responses = []
            for r in responses:
                """Extract the answer letter from the model's response."""
                # Look for \boxed{X} pattern
                boxed_match = re.search(r"\\boxed{([A-D])}", r, re.IGNORECASE)
                if boxed_match:
                    filtered_responses.append(boxed_match.group(1).upper())
                    break

                # Try to find patterns like "The answer is A" or "Option A is correct"
                # answer_pattern = re.search(r"(?:resposta|answer|opção correta|correct option).*?([A-D])\b", r, re.IGNORECASE)
                answer_pattern = re.search(r"(?:resposta correta|resposta certa|resposta verdadeira|answer|opção correta|opção certa|opção verdadeira|correct option).{0,10}([A-D])\b", r, re.IGNORECASE)
                if answer_pattern:
                    filtered_responses.append(answer_pattern.group(1).upper())
                    break

                # Look for a letter at the end of the response
                # final_letter = re.search(r"([A-D])\.?\s*$", r.strip(), re.IGNORECASE)
                final_letter = re.search(r"([A-D])\.?\s*$", r.strip())
                if final_letter:
                    filtered_responses.append(final_letter.group(1).upper())
                    break

                # If all else fails, try to find any letter A-D in the response - problem if sentence starts with A it will not work correctly
                # any_letter = re.search(r"\b([A-D])\b", r, re.IGNORECASE)
                any_letter = re.search(r"\b([A-D])\b", r)
                if any_letter:
                    filtered_responses.append(any_letter.group(1).upper())
                    break

                filtered_responses.append("[not extracted]")

            return filtered_responses

        filtered_resps = [extract_answer(resp) for resp in resps]

        return filtered_resps
