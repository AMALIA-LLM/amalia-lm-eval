import copy
import os
import requests
import time
from ruamel.yaml import YAML
import json
import traceback


DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")  # TODO add your env variable here
SOURCE_LANG = "EN"
TARGET_LANG = "PT-PT"

translation_cache = {}  # Dictionary to store translations
yaml = YAML()
yaml.preserve_quotes = True  # Preserve quotes in YAML
n_requests = 0  # Number of requests made to DeepL API


def translate_text(text, n_retries=3):
    """Translates text using DeepL API, using cache."""
    if text in translation_cache:
        print("Found in cache")
        return translation_cache[text]

    url = "https://api-free.deepl.com/v2/translate"
    headers = {
        "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "text": [text],
        "source_lang": SOURCE_LANG,
        "target_lang": TARGET_LANG
    }
    error_timeouts = [0.5, 1.0, 2.0, 4.0]
    for i in range(n_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            translated_text = response.json()["translations"][0]["text"]
            translation_cache[text] = translated_text  # Cache the translation
            translation_cache[translated_text] = translated_text  # Cache in case same language to avoid going through the API again
            global n_requests
            n_requests += 1
            print("Original text:", text)
            print("Translated text:", translated_text)
            print("Number of requests made:", n_requests)
            time.sleep(0.2)  # Wait before making another request
            return translated_text
        except requests.exceptions.RequestException as e:
            print(f"Translation error: {e}")
            time.sleep(error_timeouts[i])  # Wait before retrying

    print(f"Failed to translate text!!!")
    return None


def process_yaml_file(filepath, process_description, process_samples):
    """Reads a YAML file, translates the 'description' attribute, and writes the changes."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = yaml.load(file)

        # translate description
        if isinstance(data, dict) and "description" in data and process_description:
            original_description = data["description"]
            translated_description = translate_text(original_description)

            if translated_description:
                data["description"] = translated_description
                with open(filepath, "w", encoding="utf-8") as file:
                    yaml.dump(data, file)
                print(f"Translated description in {filepath}")
            else:
                print(f"Failed to translate description in {filepath}")

        # translate samples
        if isinstance(data, dict) and "fewshot_config" in data and process_samples:
            for sample in data["fewshot_config"]["samples"]:
                original_question = sample["question"]
                translated_question = translate_text(original_question)

                if translated_question:
                    sample["question"] = translated_question
                    with open(filepath, "w", encoding="utf-8") as file:
                        yaml.dump(data, file)
                    print(f"Translated question in {filepath}")
                else:
                    print(f"Failed to translate question in {filepath}")

                original_target = sample["target"]
                translated_target = translate_text(original_target)

                if translated_target:
                    sample["target"] = translated_target
                    with open(filepath, "w", encoding="utf-8") as file:
                        yaml.dump(data, file)
                    print(f"Translated target in {filepath}")
                else:
                    print(f"Failed to translate target in {filepath}")

    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"An unexpected error occurred processing {filepath}: {e}")
        traceback.print_exc()
        pass


def process_directory(directory, load_translation_cache: bool, process_description: bool, process_samples: bool):
    global translation_cache
    global n_requests

    if load_translation_cache and os.path.exists("translation_cache.json"):
        with open("translation_cache.json", "r") as f:
            translation_cache = json.load(f)

    """Recursively processes all YAML files in a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                filepath = os.path.join(root, file)
                process_yaml_file(filepath, process_description, process_samples)

    print(f"Requests made: {n_requests}")
    print(f"Translations cached: {len(translation_cache)}")

    # write translation cache to json file to avoid recompute
    with open("translation_cache.json", "w") as f:
        json.dump(translation_cache, f, indent=4)


def get_all_unique_descriptions_in_yamls(directory):
    unique_descriptions = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.load(f)
                    if isinstance(data, dict) and "description" in data:
                        unique_descriptions.add(data["description"])

    print(f"Number of unique descriptions: {len(unique_descriptions)}")
    print("Total number of tokens in unique descriptions:", sum(len(desc.split()) for desc in unique_descriptions))
    print("Total number of characters in unique descriptions:", sum(len(desc) for desc in unique_descriptions))
    return unique_descriptions


def get_unique_samples_in_yamls(directory):
    unique_samples = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.load(f)
                    if isinstance(data, dict) and "fewshot_config" in data:
                        for sample in data["fewshot_config"]["samples"]:
                            unique_samples.add(sample["question"])
                            unique_samples.add(sample["target"])

    print(f"Number of unique samples: {len(unique_samples)}")
    print("Total number of tokens in unique samples:", sum(len(sample.split()) for sample in unique_samples))
    print("Total number of characters in unique samples:", sum(len(sample) for sample in unique_samples))
    return unique_samples


def translate_descriptions():
    directory_to_process = "."
    get_all_unique_descriptions_in_yamls(directory_to_process)
    process_directory(directory_to_process, load_translation_cache=True, process_description=True, process_samples=False)


def translate_samples():
    directory_to_process = "flan_cot_fewshot"
    get_unique_samples_in_yamls(directory_to_process)
    process_directory(directory_to_process, load_translation_cache=True, process_description=False, process_samples=True)


def translate_cot_prompts(load_translation_cache: bool = True):

    file_to_process = "flan_cot_fewshot/_cot_prompts.json"

    global translation_cache
    global n_requests

    # Load translation cache
    if load_translation_cache and os.path.exists("translation_cache.json"):
        with open("translation_cache.json", "r") as f:
            translation_cache = json.load(f)

    with open(file_to_process, "r") as f:
        cot_file = json.load(f)

        # number of unique descriptions
        print(f"Number of unique descriptions: {len(cot_file)}")
        print("Total number of tokens in unique descriptions:", sum(len(desc.split()) for desc in cot_file.values()))

        for subject, description in copy.deepcopy(cot_file).items():
            translated_description = translate_text(description)
            if translated_description:
                # fix common \nQ: and \nA: patterns to \nP: and \nR: for pt-pt
                translated_description = translated_description.replace("\nQ:", "\nP:").replace("\nA:", "\nR:")
                cot_file[subject] = translated_description

                # add to cache
                translation_cache[description] = translated_description
                translation_cache[translated_description] = translated_description

            else:
                print(f"Failed to translate description for {subject}")

    print(f"Requests made: {n_requests}")
    print(f"Translations cached: {len(translation_cache)}")

    # write the file cot_prompts.json with the translated descriptions
    with open(file_to_process, "w") as f:
        json.dump(cot_file, f, indent=4)

    # write translation cache to json file to avoid recompute
    with open("translation_cache.json", "w") as f:
        json.dump(translation_cache, f, indent=4)


if __name__ == "__main__":
    # translate_descriptions()

    # translate_samples()

    # translate_cot_prompts(load_translation_cache=True)

    pass


# example usage
# 1. Set/Provide the DeepL API key
# 1. Change the main function to run what you want
# 2. Run the script - using the this level as the working dir
# python3 _translate_descriptions.py
