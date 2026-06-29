import re


def extract_first_word(input_string):
    # Define a regular expression pattern to match the first word
    pattern = r'\b\w+\b'

    # Use the findall function from the re module to find all matches
    matches = re.findall(pattern, input_string)

    if matches:
        return matches[0]
    else:
        return ""


class ExtractAnswerR:
    """
    Extracts the answer from the response and returns it.
    """

    def __init__(self):
        pass

    def apply(self, resps: list[list[str]], docs: list[dict]):

        def extract_answer(responses):

            filtered_responses = []
            for r in responses:

                response = r.strip()
                if not response:
                    filtered_responses.append("")
                    break

                response = extract_first_word(response)
                filtered_responses.append(response)

            return filtered_responses

        filtered_resps = [extract_answer(resp) for resp in resps]

        return filtered_resps


# test
if __name__ == "__main__":
    extractor = ExtractAnswerR()
    responses = [
        [" reserva.\nA série"],
    ]
    for resp in extractor.apply(responses, []):
        print(resp)  # Expected output: ['reserva']