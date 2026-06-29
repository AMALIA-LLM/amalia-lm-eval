import json


def transform_json(input_data):
    output = []

    for item in input_data:
        new_item = {
            "_id": item["id"],
            "answer": item["answer"],  # Note: You'll need to translate 'não' to 'no'
            "question": item["question"],  # Note: You'll need to translate this
            "type": item["type"],
            "level": item["level"]
        }

        # 1. Transform supporting_facts: Zip title and sent_id together
        titles = item["supporting_facts"]["title"]
        sent_ids = item["supporting_facts"]["sent_id"]
        new_item["supporting_facts"] = [list(a) for a in zip(titles, sent_ids)]

        # 2. Transform context: Zip title and sentences together
        ctx_titles = item["context"]["title"]
        ctx_sentences = item["context"]["sentences"]
        new_item["context"] = [list(a) for a in zip(ctx_titles, ctx_sentences)]

        output.append(new_item)

    return output


if __name__ == "__main__":
    # from here https://raw.githubusercontent.com/i2ca/pt-hotpotqa/refs/heads/main/datasets/pt_hotpotqa_distractor_validation.json
    with open("pt_hotpotqa_distractor_validation.json", "r") as f:
        input_data = json.load(f)

    transformed_data = transform_json(input_data)

    with open("pt_hotpotqa_distractor_validation_transformed.json", "w") as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)
