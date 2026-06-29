import datasets

def process_xtest_samples(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda sample: sample.get("label") == "unsafe")

def process_wild_guard_test_samples(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda sample: sample.get("label") == "harmful" or sample.get("prompt_harm_label") == "harmful")

def process_harmbench_contextual_samples(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(lambda sample: {
        'prompt': sample['prompt'] + '\n' + sample['context'],
        'category': sample['category'],
        'original_prompt': sample['prompt'],
        'original_context': sample['context']
    })

def process_toxic_chat_samples(dataset: datasets.Dataset) -> datasets.Dataset:
    temp_dataset = dataset.filter(lambda sample: sample.get("toxicity") == 1 or sample.get("jailbreaking") == 1)
    # rename user_input to prompt
    return temp_dataset.map(lambda sample: {
        'prompt': sample['user_input'],
        'model_output': sample['model_output'],
        'toxicity': sample['toxicity'],
        'jailbreaking': sample['jailbreaking'],
        'human_annotation': sample['human_annotation'],
        'openai_moderation': sample['openai_moderation']
    })
