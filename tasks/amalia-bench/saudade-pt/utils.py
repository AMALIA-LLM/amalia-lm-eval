import json
import logging
import re
from pathlib import Path

from datasets import Dataset


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


def _record_key(doc):
    return doc["question"], tuple(doc["expected_answers"]), tuple(doc["facts_pair"])


_BINARY_POLICY = {
    _record_key(entry["record"]): entry
    for entry in json.loads(
        Path(__file__).with_name("binary_track_manifest.json").read_text(encoding="utf-8")
    )
}
_LOGGER = logging.getLogger(__name__)


def _prepare_doc(doc):
    question = doc.get("question")
    facts = doc.get("facts_pair")
    answers = doc.get("expected_answers")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if not isinstance(facts, list) or len(facts) != 2 or any(
        not isinstance(fact, str) or not fact.strip() for fact in facts
    ):
        raise ValueError("facts_pair must contain two nonempty strings")
    if not isinstance(answers, list) or len(answers) != 1 or not isinstance(
        answers[0], str
    ) or not answers[0].strip():
        raise ValueError("expected_answers must contain one nonempty string")

    policy = _BINARY_POLICY.get(_record_key(doc))
    if policy is not None and policy["action"] == "exclude":
        return None, policy

    if 'primeiro: ' in question:
        prompt = question.split('primeiro:')[0].strip() + ' primeiro:'
    elif ": " in question:
        prompt = question.split(':')[0].strip() + ':'
    elif "ocorreu primeiro " in question:
        prompt = question.split('ocorreu primeiro')[0].strip() + ' ocorreu primeiro'
    else:
        raise ValueError("no prompt prefix; record is not a canonical exclusion")

    expected = answers[0].strip().lower()
    matches = [i for i, fact in enumerate(facts) if fact.strip().lower() == expected]
    if len(matches) == 1:
        answer_index = matches[0]
    elif not matches and policy is not None and policy["action"] == "answer":
        answer_index = policy["answer_index"]
    else:
        raise ValueError("reference answer must match exactly one option")

    if "prompt" in doc and doc["prompt"] != prompt:
        raise ValueError("prepared prompt disagrees with the source question")
    if "answer_index" in doc and (
        type(doc["answer_index"]) is not int or doc["answer_index"] != answer_index
    ):
        raise ValueError("prepared answer_index disagrees with the reference answer")
    return {"prompt": prompt, "answer_index": answer_index}, policy


def process_docs(dataset: Dataset) -> Dataset:
    retained, prompts, answers = [], [], []
    seen_policy = set()
    for row, doc in enumerate(dataset):
        try:
            prepared, policy = _prepare_doc(doc)
            if policy is not None:
                key = _record_key(doc)
                if key in seen_policy:
                    raise ValueError("canonical policy record occurs more than once")
                seen_policy.add(key)
                _LOGGER.info(
                    "SAUDADE source row %d (%s): %s (manifest line %d)",
                    row + 1, doc.get("entity", "unknown entity"),
                    policy["reason"], policy["source_line"],
                )
        except ValueError as error:
            raise ValueError(
                f"SAUDADE source row {row + 1} ({doc.get('entity', 'unknown entity')}): "
                f"{error}"
            ) from error
        if prepared is not None:
            retained.append(row)
            prompts.append(prepared["prompt"])
            answers.append(prepared["answer_index"])
    if not retained:
        raise ValueError("SAUDADE preparation produced no evaluation records")
    result = dataset.select(retained)
    for field, values in [("prompt", prompts), ("answer_index", answers)]:
        if field in result.column_names:
            result = result.remove_columns(field)
        result = result.add_column(field, values)
    _LOGGER.info(
        "SAUDADE prepared %d of %d records; excluded %d canonical prefix failures",
        len(result), len(dataset), len(dataset) - len(result),
    )
    return result
