"""
Take in a YAML, and output all other splits with this YAML
"""

import argparse
import os
import re
import traceback
import json

import datasets
import requests
import yaml
from tqdm import tqdm

from lm_eval import utils
from _translate_configs import DeepLTranslator
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="zeroshot")
    parser.add_argument("--cot", default=False)
    parser.add_argument("--fewshot", default=False)
    parser.add_argument("--task_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    translation_cache_path = "translation_cache.json"
    # create empty translation cache if it does not exist
    translation_cache = {}
    if os.path.exists(translation_cache_path):
        with open(translation_cache_path, "r", encoding="utf-8") as f:
            translation_cache = json.load(f)
    else:
        with open(translation_cache_path, "w", encoding="utf-8") as f:
            json.dump(translation_cache, f, indent=4)

    translator = DeepLTranslator()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    # base_doc_to_text = "Q: {{input}}\nA:"
    base_doc_to_text = "P: {{input}}\nR:"
    # answer_regex = re.compile("(?<=answer is )(.*)(?=.)")
    answer_regex = re.compile("(?<=resposta é )(.*)(?=\\n)")

    dataset_path = "lukaemon/bbh"
    try:
        for task in tqdm(datasets.get_dataset_infos(dataset_path).keys()):
            resp = requests.get(
                f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/{task}.txt"
            ).content.decode("utf-8")
            prompt = resp.split("\n-----\n")[-1]
            description_orig, *few_shot_orig = prompt.split("\n\n")

            if description_orig in translation_cache:
                description = translation_cache[description_orig]
            else:

                # translate the promt and the description
                description = translator.run_translation(
                    description_orig,
                    source_lang="EN",
                    target_lang="PT-PT"
                )
                # sleep to avoid hitting the rate limit
                time.sleep(0.1)

                # add to cache
                translation_cache[description_orig] = description
                translation_cache[description] = description

            few_shot = []
            for shot in few_shot_orig:
                if shot in translation_cache:
                    few_shot.append(translation_cache[shot])
                else:
                    # translate the few shot
                    translated_shot = translator.run_translation(
                        shot,
                        source_lang="EN",
                        target_lang="PT-PT"
                    )
                    # sleep to avoid hitting the rate limit
                    time.sleep(0.1)

                    few_shot.append(translated_shot)
                    # add to cache
                    translation_cache[shot] = translated_shot
                    translation_cache[translated_shot] = translated_shot

            prefix_doc_to_text = ""
            if args.fewshot:
                if args.cot:
                    prefix_doc_to_text = "\n\n".join(few_shot) + "\n\n"
                else:
                    for shot in few_shot:
                        try:
                            answer = answer_regex.search(shot)[0]
                        except Exception:
                            print("task", task)
                            print(shot)
                        example = shot.split("Vamos pensar passo a passo.")[0]
                        prefix_doc_to_text += f"{example}{answer}\n\n"

            doc_to_text = prefix_doc_to_text + base_doc_to_text
            if args.cot:
                doc_to_text = doc_to_text + " Vamos pensar passo a passo.\n"

            yaml_dict = {
                "include": base_yaml_name,
                "task": f"bbh{args.task_prefix}_{task}_mt_pt",
                "dataset_name": task,
                "description": description + "\n\n",
                "doc_to_text": doc_to_text,
            }

            file_save_path = args.save_prefix_path + f"/{task}.yaml"
            utils.eval_logger.info(f"Saving yaml for subset {task} to {file_save_path}")
            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(
                    yaml_dict,
                    yaml_file,
                    width=float("inf"),
                    allow_unicode=True,
                    default_style='"',
                )

    except Exception as e:
        print(f"An error occurred: {e}")
        # print stack trace
        traceback.print_exc()
    finally:
        if translation_cache:
            with open(translation_cache_path, "w", encoding="utf-8") as f:
                json.dump(translation_cache, f, indent=4)
