import json
from pathlib import Path
from difflib import SequenceMatcher


def normalize_text(text):
    """Normalize text for comparison."""
    return text.strip().lower()


def find_answer_index(expected_answer, facts_pair):
    """
    Find the index of the expected answer in facts_pair.
    Returns (index, success) tuple where index is 0 or 1, or (None, False) if not found.
    """
    expected_norm = normalize_text(expected_answer)

    # Try exact match first
    for i, fact in enumerate(facts_pair):
        if normalize_text(fact) == expected_norm:
            return i, True

    # Try substring match
    for i, fact in enumerate(facts_pair):
        fact_norm = normalize_text(fact)
        if expected_norm in fact_norm or fact_norm in expected_norm:
            return i, True

    # Try fuzzy match as last resort
    best_match_idx = None
    best_ratio = 0.0
    for i, fact in enumerate(facts_pair):
        ratio = SequenceMatcher(None, expected_norm, normalize_text(fact)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match_idx = i

    if best_ratio > 0.8:  # Threshold for fuzzy match
        return best_match_idx, True

    return None, False


def extract_prompt_prefix(question):
    """
    Extract the part before 'primeiro:' from the question.
    If 'primeiro:' is not present, extract the part before the first ': '.
    Returns (prompt, success) tuple.
    """
    if 'primeiro: ' in question:
        return question.split('primeiro:')[0].strip() + ' primeiro:', True
    elif ": " in question:
        return question.split(':')[0].strip() + ':', True
    elif "ocorreu primeiro " in question:
        return question.split('ocorreu primeiro')[0].strip() + ' ocorreu primeiro', True
    else:
        return None, False


def process_dataset(input_path, output_path):
    """
    Process the temporal reasoning dataset and add:
    - prompt: the question text before 'primeiro:'
    - answer_index: the index (0 or 1) of the expected answer in facts_pair
    """
    processed_data = []
    answer_errors = []
    prompt_extraction_failures = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            data = json.loads(line.strip())

            # Extract prompt prefix
            prompt, success = extract_prompt_prefix(data['question'])
            if not success:
                prompt_extraction_failures.append({
                    'line': line_num,
                    'question': data['question'],
                    'entity': data['entity']
                })
                # Skip this item
                continue

            # Find answer index
            expected_answer = data['expected_answers'][0]
            answer_index, success = find_answer_index(expected_answer, data['facts_pair'])
            if not success:
                answer_errors.append({
                    'line': line_num,
                    'entity': data['entity'],
                    'expected': expected_answer,
                    'facts_pair': data['facts_pair']
                })
                # Skip this item
                continue

            # Add new fields
            processed_item = {
                **data,
                'prompt': prompt,
                'answer_index': answer_index
            }

            processed_data.append(processed_item)

    # Write processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✓ Processed {len(processed_data)} samples")
    print(f"✓ Output written to: {output_path}")

    if prompt_extraction_failures:
        print(f"\n⚠ {len(prompt_extraction_failures)} items failed prompt extraction and were skipped:")
        for failure in prompt_extraction_failures:
            print(f"\n  Line {failure['line']}: {failure['entity']}")
            print(f"    Question: {failure['question']}")

    if answer_errors:
        print(f"\n⚠ {len(answer_errors)} items had answer matching issues and were skipped:")
        for err in answer_errors[:5]:
            print(f"\n  Line {err['line']}: {err['entity']}")
            print(f"    Expected: {err['expected']}")
            print(f"    Facts: {err['facts_pair']}")
        if len(answer_errors) > 5:
            print(f"\n  ... and {len(answer_errors) - 5} more")

    # Print statistics
    answer_distribution = {}
    for item in processed_data:
        idx = item['answer_index']
        answer_distribution[idx] = answer_distribution.get(idx, 0) + 1

    print(f"\nAnswer index distribution:")
    for idx in sorted(answer_distribution.keys()):
        count = answer_distribution[idx]
        pct = (count / len(processed_data)) * 100
        print(f"  Index {idx}: {count} ({pct:.1f}%)")

    total_skipped = len(prompt_extraction_failures) + len(answer_errors)
    if total_skipped > 0:
        print(f"\n⚠ Total skipped: {total_skipped} items")


# this script extracts from the raw question the options for easier creation of MCQ benchmark
if __name__ == '__main__':
    input_file = Path(__file__).parent / 'TemporalPT-EU_reasoning_v4_test_flat.jsonl'
    output_file = Path(__file__).parent / 'TemporalPT-EU_reasoning_v4_test_processed.jsonl'

    print(f"Processing {input_file.name}...")
    process_dataset(input_file, output_file)
