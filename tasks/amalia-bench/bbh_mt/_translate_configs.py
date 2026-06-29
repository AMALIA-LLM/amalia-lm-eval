import json
import os
import time
from typing import Optional
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True  # Keep original quoting

import requests


class DeepLTranslator:

    def __init__(self, url: str = "https://api-free.deepl.com/v2/translate", api_key: str = None):
        super().__init__()
        self.url = url
        self.api_key = api_key if api_key else os.getenv("DEEPL_API_KEY")

    def run_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> Optional[str]:

        url = "https://api-free.deepl.com/v2/translate"
        headers = {
            "Authorization": f"DeepL-Auth-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "text": [text],
            "source_lang": source_lang,
            "target_lang": target_lang
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        translated_text = response.json()["translations"][0]["text"]

        return translated_text


if __name__ == "__main__":
    translator = DeepLTranslator()

    translation_cache = {}
    translation_cache_path = "translation_cache.json"
    if os.path.exists(translation_cache_path):
        with open(translation_cache_path, "r", encoding="utf-8") as f:
            translation_cache = json.load(f)
    else:
        with open(translation_cache_path, "w", encoding="utf-8") as f:
            json.dump(translation_cache, f, indent=4)

    try:
        start_dir = "."
        # recursively search for all yaml files
        for root, _, files in os.walk(start_dir):
            for file in files:
                if file.endswith(".yaml"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        yaml_content = yaml.load(f)

                    if "task" in yaml_content:
                        if isinstance(yaml_content["task"], list):
                            new_task = []
                            for task in yaml_content["task"]:
                                if "_mt_pt" not in task:
                                    new_task.append(task + "_mt_pt")
                                else:
                                    new_task.append(task)
                            yaml_content["task"] = new_task
                        else:
                            if "_mt_pt" not in yaml_content["task"]:
                                # append _mt_pt to the task name
                                yaml_content["task"] = yaml_content["task"] + "_mt_pt"

                    if "group" in yaml_content and "_mt_pt" not in yaml_content["group"]:
                        yaml_content["group"] = yaml_content["group"] + "_mt_pt"

                    if "tasks" in yaml_content:
                        # now for each  task add mt_pt
                        for task in yaml_content["tasks"]:
                            if "_mt_pt" not in task["name"]:
                                task["name"] = task["name"] + "_mt_pt"

                    # change dataset_path to amalia-llm/bigbenchhard-mt-pt
                    if "dataset_path" in yaml_content:
                        yaml_content["dataset_path"] = "amalia-llm/bigbenchhard-mt-pt"

                    # translate the description
                    if "description" in yaml_content:

                        description = yaml_content["description"]

                        if description in translation_cache:
                            translated_description = translation_cache[description]
                        else:
                            translated_description = translator.run_translation(
                                description,
                                source_lang="EN",
                                target_lang="PT-PT"
                            )
                        yaml_content["description"] = translated_description
                        # sleep to avoid hitting the rate limit
                        time.sleep(0.1)
                        # add to cache
                        translation_cache[yaml_content["description"]] = translated_description
                        translation_cache[translated_description] = translated_description

                    if "doc_to_text" in yaml_content:
                        yaml_content["doc_to_text"] = yaml_content["doc_to_text"].replace("Q:", "P:").replace("A:", "R:").replace("Let's think step by step.", "Vamos pensar passo a passo.")

                    # translate the few shot configs
                    if "fewshot_config" in yaml_content:
                        few_shot_translated = []
                        for shot in yaml_content["fewshot_config"]["samples"]:

                            shot_input = shot.get("input", "")
                            if shot_input in translation_cache:
                                translated_input = translation_cache[shot_input]
                            else:
                                translated_input = translator.run_translation(
                                    shot_input,
                                    source_lang="EN",
                                    target_lang="PT-PT"
                                )
                                # sleep to avoid hitting the rate limit
                                time.sleep(0.1)
                                # add to cache
                                translation_cache[shot_input] = translated_input
                                translation_cache[translated_input] = translated_input

                            shot_target = shot.get("target", "")
                            if shot_target in translation_cache:
                                translated_target = translation_cache[shot_target]
                            else:
                                translated_target = translator.run_translation(
                                    shot_target,
                                    source_lang="EN",
                                    target_lang="PT-PT"
                                )
                                # sleep to avoid hitting the rate limit
                                time.sleep(0.1)
                                # add to cache
                                translation_cache[shot_target] = translated_target
                                translation_cache[translated_target] = translated_target

                            few_shot_translated.append({
                                "input": translated_input,
                                "target": translated_target
                            })

                        yaml_content["fewshot_config"]["samples"] = few_shot_translated

                    if "filter_list" in yaml_content:
                        # translate the filter list
                        pass  # TODO this will be manually translated

                    # save the modified yaml file
                    with open(file_path, "w", encoding="utf-8") as f:
                        yaml.dump(yaml_content, f)

                elif file.endswith("_yaml"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        yaml_content = yaml.load(f)

                        yaml_content["dataset_path"] = "amalia-llm/bigbenchhard-mt-pt"

                        if yaml_content["generation_kwargs"]["until"]:
                            if "<|endoftext|>" not in yaml_content["generation_kwargs"]["until"]:
                                yaml_content["generation_kwargs"]["until"].append("<|endoftext|>")
                            if "<|im_end|>" not in yaml_content["generation_kwargs"]["until"]:
                                yaml_content["generation_kwargs"]["until"].append("<|im_end|>")
                            if "P:" not in yaml_content["generation_kwargs"]["until"]:
                                yaml_content["generation_kwargs"]["until"].append("P:")

                    # save the modified yaml file
                    with open(file_path, "w", encoding="utf-8") as f:
                        yaml.dump(yaml_content, f)


    except Exception as e:
        print(f"An error occurred: {e}")
        # print stack trace
        import traceback
        traceback.print_exc()
    finally:
        if translation_cache:
            with open(translation_cache_path, "w", encoding="utf-8") as f:
                json.dump(translation_cache, f, indent=4)
