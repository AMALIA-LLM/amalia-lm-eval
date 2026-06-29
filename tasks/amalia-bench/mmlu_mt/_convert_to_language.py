import argparse
import os
import traceback
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True  # Keep original quoting


def process_yaml_files(folder_path: str, dataset_path: str = "LumiOpen/opengpt-x_mmlux", language: str = "_PT-PT",
                       task_prefix: str = "amalia_", task_suffix: str ="_mt_pt"):

    """
    Recursively finds YAML files in a folder, modifies specific attributes, and saves them.

    Args:
        folder_path (str): The path to the folder to process.
        dataset_path (str): The dataset path to set in the YAML files.
        language (str): The language to append to the dataset name.
        task_prefix (str): The prefix to add to the task name.
        task_suffix (str): The suffix to add to the task name
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml") or file.endswith("_yaml"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        # data = yaml.safe_load(f)
                        data = yaml.load(f)

                    do_some_processing = False
                    if data and isinstance(data, dict):
                        if "dataset_path" in data:
                            data["dataset_path"] = dataset_path
                            do_some_processing = True
                        if "dataset_name" in data and language not in data["dataset_name"]:
                            data["dataset_name"] = data["dataset_name"] + language
                            do_some_processing = True
                        if "tag" in data and not task_prefix in data["tag"] and not task_suffix in data["tag"]:
                            data["tag"] = task_prefix + data["tag"] + task_suffix
                            do_some_processing = True
                        if "task" in data:
                            if isinstance(data["task"], str) and not task_prefix in data["task"] and not task_suffix in data["task"]:
                                data["task"] = task_prefix + data["task"] + task_suffix
                                do_some_processing = True
                            # in case it is a list replace the task in all tasks
                            elif isinstance(data["task"], list):
                                for i, task in enumerate(data["task"]):
                                    if isinstance(task, str) and not task_prefix in task and not task_suffix in task:
                                        data["task"][i] = task_prefix + task + task_suffix
                                        do_some_processing = True
                                    elif isinstance(task, dict) and "task" in task:
                                        if isinstance(task["task"], str) and not task_prefix in task["task"] and not task_suffix in task["task"]:
                                            data["task"][i]["task"] = task_prefix + task["task"] + task_suffix
                                            do_some_processing = True
                                        elif isinstance(task["task"], list):
                                            for j, sub_task in enumerate(task["task"]):
                                                if isinstance(sub_task, str) and not task_prefix in sub_task and not task_suffix in sub_task:
                                                    data["task"][i]["task"][j] = task_prefix + sub_task + task_suffix
                                                    do_some_processing = True

                    if do_some_processing:
                        with open(file_path, "w") as f:
                            yaml.dump(data, f)

                        print(f"Processed: {file_path}")

                except (FileNotFoundError, AttributeError) as e:
                    print(f"Error processing {file_path}: {e}")
                except Exception as e:
                    print(f"Unexpected error processing {file_path}: {e}")
                     # print traceback
                    traceback.print_exc()
                    raise e


def main():
    parser = argparse.ArgumentParser(description="Process YAML files in a folder.")
    parser.add_argument("folder_path", type=str, help="The path to the folder to process.")
    parser.add_argument("--dataset_path", type=str, default="LumiOpen/opengpt-x_mmlux", help="The dataset path to set in the YAML files.")
    parser.add_argument("--language", type=str, default="_PT-PT", help="The language to append to the dataset name.")
    parser.add_argument("--task_prefix", type=str, default="amalia_", help="The prefix to add to the task name.")
    parser.add_argument("--task_suffix", type=str, default="_mt_pt", help="The suffix to add to the task name.")

    args = parser.parse_args()

    if os.path.exists(args.folder_path) and os.path.isdir(args.folder_path):
        process_yaml_files(args.folder_path, args.dataset_path, args.language, args.task_prefix, args.task_suffix)
    else:
        print("Invalid folder path.")


if __name__ == "__main__":
    main()
