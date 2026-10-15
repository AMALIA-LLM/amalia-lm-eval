import argparse
import json
import os
import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
from create_table_utils import create_tables
from model_task_pretty_names import turn_task_to_pretty_name, model_to_pretty_name_dict
from create_graphs import create_graph_per_model_task


DEV_TASKS = ['ARC-C', 'MMLU', 'G-PiQA-c', 'G-PiQA-p', 'GSM8K', 'PT-E', 'PT-E (CoT)', 'FRMT pt-pt', 'HelSw',
             'TQA', 'FRMT pt-pt (0-shot)', 'IFEval', 'ARC-C (en)', 'HelSw (en)', 'PT-C',
             'GSM8K (en)', 'TQA (en)', 'IFEval (en)', 'MMLU (en)']


def handle_non_aggregating(data):
    """
    Some tasks are not aggregating the results properly so this function calculates the aggregate metrics for a task
    """

    # new lm-eval-harness release first check if groups exist and has the value
    if data.get("groups"):
        group_key = list(data.get("groups").keys())[0]
        data_for_group = data.get("groups")[group_key]  # dict[str, any]
        metric_values = defaultdict(dict)
        for k, v in data_for_group.items():
            # skip metadata fields
            if k in ["alias", "name", "sample_len", "sample_count"]:
                continue
            else:
                metric_values[k] = v

        if metric_values:
            return metric_values

    # old method
    n_samples = data.get("n-samples", {})
    samples_dict = {}  # saves the number of samples for each subtask
    for subtask, n_sample_data in n_samples.items():
        if "effective" in n_sample_data:
            samples_dict[subtask] = n_sample_data.get("effective", 0)
        else:
            samples_dict[subtask] = n_sample_data.get("original", 0)

    metrics_dict = defaultdict(dict)  # saves the metrics for each subtask
    # get the metrics for each subtask
    for subtask, metrics_data in data.get("results", {}).items():
        for metric_name, value in metrics_data.items():
            if metric_name in ["alias", "name", "sample_len", "sample_count"]:
                continue
            else:
                metrics_dict[subtask][metric_name] = value

    metrics_values = {}  # key is metric name, value is the average value weighted by the number of samples
    for subtask, metrics_data in metrics_dict.items():
        for metric_name, value in metrics_data.items():
            if metric_name not in metrics_values:
                metrics_values[metric_name] = 0
            # if value is numeric round it to 2 decimal places
            if isinstance(value, (int, float)):
                value = round(value, 3)
            metrics_values[metric_name] += value * samples_dict[subtask]
    # normalize the metrics by the number of samples
    total_samples = sum(samples_dict.values())
    for metric_name, value in metrics_values.items():
        if total_samples > 0:
            metrics_values[metric_name] = round(value / total_samples, 3)
        else:
            metrics_values[metric_name] = 0.0
    # return the metrics values
    return metrics_values


def is_amalia_model(model_name: str) -> bool:
    """
    Check if the model name corresponds to an AMALIA model.
    """
    amalia_indicators = ["amalia", "checkpoint-"]
    return any(indicator in model_name.lower() for indicator in amalia_indicators)


def is_open_source(model_name: str) -> bool:
    """
    Check if the model name corresponds to an open-source model.
    """
    # TODO update as needed
    open_source_indicators = ["olmo", "salamandra", "eurollm", "apertus", "amalia", "checkpoint"]
    return any(indicator in model_name.lower() for indicator in open_source_indicators)


# it is pt if it mentiones the word amalia in task or has mt-pt or mt_pt
def is_pt_dataset(task_name: str) -> bool:
    task_name = task_name.lower()
    # TODO add other indicators if needed
    if "amalia" in task_name or "mt-pt" in task_name or "mt_pt" in task_name or "pt_" in task_name or "_port" in task_name or "_pt" in task_name or "alba" in task_name:
        return True
    return False


def is_dev_task(task_name: str) -> bool:
    task_name = task_name.lower()
    dev_indicators = [
        "commonsense_qa",
        "hellaswag",
        "arc",
        "truthfulqa",
        "gsm8k",
        # "mmlu",
        "mmlu_mt_pt",
        "pt_exams",
        "mt_bench",
        "simple safety tests",
        "ifeval",
        "frmt",
        "pt_completion",
    ]
    # contains relationship
    for indicator in dev_indicators:
        if indicator in task_name:
            return True
        # special case where mmlu must match exactly since there are other with mmlu in the name
        if task_name == "mmlu":
            return True

    return False


class Tags:

    MATH = "math"
    COMMONSENSE = "commonsense reasoning"
    REASONING = "reasoning"
    LEGAL = "legal"
    PT_SPECIFIC = "pt specific"
    GENERAL_KNOWLEDGE = "general_knowledge"
    INSTRUCTION_FOLLOWING = "instruction following"
    SAFETY = "safety"
    TRANSLATION = "translation"
    PT_VARIANT = "pt variant id"
    NLU = "nlu"
    CODE = "code"
    LONG_CONTEXT = "long context"

    # this the same name might have both
    GENERATION = "generation"
    MCQ = "multiple choice"

    OTHER = "other"



def add_tags_to_datasets(task_name: str) -> List[str]:

    task_sub_names = {
        "piqa": [Tags.COMMONSENSE, Tags.MCQ],
        "arc_challenge": [Tags.COMMONSENSE, Tags.MCQ],
        "gsm8k": [Tags.MATH, Tags.GENERATION],
        "ifeval": [Tags.INSTRUCTION_FOLLOWING, Tags.GENERATION],
        "pt_exams": [Tags.PT_SPECIFIC, Tags.MCQ],
        "truthfulqa": [Tags.NLU, Tags.MCQ],
        "dsl_tl": [Tags.PT_VARIANT, Tags.MCQ],
        "global_piqa": [Tags.PT_SPECIFIC, Tags.MCQ, Tags.COMMONSENSE],
        "social_iqa": [Tags.COMMONSENSE, Tags.MCQ],
        "siqa": [Tags.COMMONSENSE, Tags.MCQ],
        "frmt": [Tags.TRANSLATION, Tags.MCQ],
        "legalbench": [Tags.LEGAL, Tags.MCQ],
        "alba": [Tags.PT_SPECIFIC, Tags.MCQ],
        "ptbrvid": [Tags.PT_VARIANT, Tags.MCQ],
        "bbh": [Tags.REASONING, Tags.GENERATION],
        "pt_text_completion": [Tags.PT_SPECIFIC, Tags.MCQ],
        "mmlu": [Tags.GENERAL_KNOWLEDGE, Tags.MCQ],
        "hellaswag": [Tags.COMMONSENSE, Tags.MCQ],
        "openbookqa": [Tags.GENERAL_KNOWLEDGE, Tags.MCQ],
        "winogrande": [Tags.COMMONSENSE, Tags.MCQ],
        "humaneval": [Tags.CODE, Tags.GENERATION],
        "calame": [Tags.GENERATION],
        "babilong": [Tags.LONG_CONTEXT],
        "ruler": [Tags.LONG_CONTEXT],
        "scrolls": [Tags.LONG_CONTEXT],
        # safety benchmarks"
        "simple_safety_tests": [Tags.SAFETY, Tags.GENERATION],
        "multilingual_advbench": [Tags.SAFETY, Tags.GENERATION],
        "xstest": [Tags.SAFETY, Tags.GENERATION],
        "wild_guard_test": [Tags.SAFETY, Tags.GENERATION],
        "harmbench_standard": [Tags.SAFETY, Tags.GENERATION],
        "harmbench_contextual": [Tags.SAFETY, Tags.GENERATION],
        "toxic_chat": [Tags.SAFETY, Tags.GENERATION],
        # other math datasets
        "aime": [Tags.MATH, Tags.GENERATION],
        "minerva_math": [Tags.MATH, Tags.GENERATION],
        "math_pt": [Tags.MATH, Tags.GENERATION],
        # new pt datasets
        "cultura_viva": [Tags.PT_SPECIFIC, Tags.MCQ],
        "saudade": [Tags.PT_SPECIFIC, Tags.MCQ],
    }

    tags = []
    for sub_name in task_sub_names:
        if sub_name in task_name.lower():
            tags.extend(task_sub_names[sub_name])

    tags = list(set(tags))  # remove duplicates

    if not tags:
        tags.append(Tags.OTHER)

    return tags


def extract_results(folder_paths: List[str], only_most_recent: bool, ignore_subcategories: bool,
                    include_models: List[str] = None, exclude_models: List[str] = None,
                    min_checkpoint: int = None, max_checkpoint: int = None,
                    checkpoints: str = None, include_datasets: List[str] = None, exclude_datasets: List[str] = None
                    ) -> pd.DataFrame:
    """
    Recursively finds JSON files, extracts results, task, metrics, model name, and date,
    and returns a pandas DataFrame.

    If only_most_recent is True, only the most recent result for each task and model is kept.
    """
    results_list: List[Dict[str, Any]] = []
    results_folder = set()
    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.startswith("results_") and file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    try:
                        # Extract date from filename by capturing it in single
                        # e.g. date 2025-03-18T19-55-29.060947
                        date_str_match = re.search(r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6})", file)
                        if date_str_match:
                            date_str = date_str_match.group(1)
                            date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H-%M-%S.%f")
                            date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            date = "Unknown"

                        with open(file_path, "r") as f:
                            data: Dict[str, Any] = json.load(f)
                            model_name: str = data["model_name"]
                            fewshot_as_multiturn = data.get("fewshot_as_multiturn", False)
                            chat_template = bool(data.get("chat_template", None))
                            config_keys = list(data.get("configs", {}).keys())
                            enable_thinking = False
                            if config_keys:
                                num_fewshot = data.get("configs")[config_keys[0]].get("num_fewshot", 0)

                                # special case for babilong_longctx where we grab metadata from the config
                                babi_max_seq_lengths = data.get("configs")[config_keys[0]].get("metadata", {}).get("max_seq_lengths", None)

                                # thinking models
                                enable_thinking = data.get("configs")[config_keys[0]].get("metadata", {}).get("enable_thinking", False)

                            # Determine main task and subtasks from groups
                            groups = data.get("groups", {})
                            main_task = None
                            subtasks_set = set()

                            # If there are groups, the first group is the main task
                            if groups:
                                main_task = list(groups.keys())[0]
                                group_subtasks = list(data.get("group_subtasks", {}).keys())
                                subtasks_set = set(group_subtasks)

                                # also add each larger group and the subtasks in them
                                for subtask_id, other_subtasks_list in data.get("group_subtasks", {}).items():
                                    if subtask_id != main_task:
                                        subtasks_set.add(subtask_id)
                                    subtasks_set.update(other_subtasks_list)

                                # remove main task in case it is stuck there
                                subtasks_set.discard(main_task)

                            for i, (task, task_data) in enumerate(data["results"].items()):
                                # If main_task is still None, use the first task as main task
                                if main_task is None and i == 0:
                                    main_task = task

                                task_complete_name = task + (" | ct" if chat_template else "") + " | " + str(num_fewshot)
                                if enable_thinking:
                                    task_complete_name = task_complete_name + " | thinking"

                                if task_complete_name.startswith("amalia_"):
                                    task_complete_name = task_complete_name.replace("amalia_", "")

                                # include datasets if needed
                                if include_datasets is not None:
                                    if task_complete_name not in include_datasets:
                                        # also check pretty name
                                        if turn_task_to_pretty_name(task_complete_name) not in include_datasets:
                                            continue

                                # exclude datasets if needed
                                if exclude_datasets is not None:
                                    if task_complete_name in exclude_datasets:
                                        continue
                                    # also check pretty name
                                    if turn_task_to_pretty_name(task_complete_name) in exclude_datasets:
                                        continue

                                if ignore_subcategories and i > 0:
                                    # if ignore_subcategories is True, only keep the first task
                                    break

                                # TODO special case for pt_exams_cot which is not aggregating the results despite by best efforts
                                if task in ["pt_exams_cot", "bbh_cot_fewshot_mt_pt", "calame_pt", "cultura_viva_pt_chat", "cultura_viva_pt_chat_cot", "pt_exams_chat", "pt_exams_chat_chat_gen", "amalia_mmlu_mt_pt_chat", "amalia_mmlu_mt_pt_chat_gen"]: #and len(task_data) == 1:
                                    values_dict = handle_non_aggregating(data)
                                    j = 0
                                    for metric, value in values_dict.items():

                                        if metric not in ["alias", "name", "sample_len", "sample_count"]:
                                            j += 1

                                            results_list.append({
                                                "task": task,
                                                "main_task": main_task,
                                                "is_subcategory": task in subtasks_set and task != main_task,
                                                "is_main_metric": j == 1,
                                                "chat_template": chat_template,
                                                "fewshot_as_multiturn": fewshot_as_multiturn,
                                                "num_fewshot": num_fewshot,
                                                "metric": metric,
                                                "value": value,
                                                "model_name": model_name,
                                                "pretty_model_name": model_to_pretty_name_dict.get(model_name, model_name),
                                                "is_amalia_model": is_amalia_model(model_name),
                                                "is_open_source_model": is_open_source(model_name),
                                                "task_complete": task_complete_name,
                                                "pretty_task_name": turn_task_to_pretty_name(task_complete_name),
                                                "is_pt_dataset": is_pt_dataset(task),
                                                "is_dev_task": is_dev_task(task),
                                                "is_thinking": enable_thinking,
                                                "tags": add_tags_to_datasets(task),
                                                "date": date,
                                                "file_path": file,
                                            })

                                j = 0
                                is_ruler = main_task in ["ruler", "ruler_en", "ruler_pt"]

                                # Special case for minerva_math: check if math_verify metric exists (any variant)
                                is_minerva_math = "minerva_math" in main_task
                                has_math_verify = is_minerva_math and any("math_verify" in m for m in task_data.keys())

                                for metric, value in task_data.items():

                                    # ignore metadata fields
                                    if metric in ["alias", "name", "sample_len", "sample_count"]:
                                        continue
                                    # special case for ruler to ignore tasks that are not run an N/A
                                    if is_ruler and (value == -1 or isinstance(value, str)):
                                        continue

                                    # For ruler: Every valid metric is a 'main' metric
                                    # For minerva_math: math_verify is main if it exists, otherwise first metric
                                    # For others: Only the first metric in the loop is 'main'
                                    if not is_ruler:
                                        j += 1
                                        if is_minerva_math and has_math_verify:
                                            is_main = "math_verify" in metric and "stderr" not in metric
                                        else:
                                            is_main = (j == 1)
                                    else:
                                        is_main = True

                                    # if value is numeric round it to 2 decimal places
                                    if isinstance(value, (int, float)):
                                        if "chrf" in metric:
                                            value = round(value / 100, 3)
                                        else:
                                            value = round(value, 3)

                                    # special case for ruler and babilong where we want to have different entries for each length
                                    suffix = ""
                                    if is_ruler:
                                        suffix = " | " + metric.split(",")[0]
                                    elif main_task == "babilong_longctx" and babi_max_seq_lengths is not None:
                                        suffix = " | " + str(babi_max_seq_lengths)

                                    results_list.append({
                                        "task": task,
                                        "main_task": main_task,
                                        "is_subcategory": task in subtasks_set and task != main_task,
                                        "is_main_metric": is_main,
                                        "chat_template": chat_template,
                                        "fewshot_as_multiturn": fewshot_as_multiturn,
                                        "num_fewshot": num_fewshot,
                                        "metric": metric,
                                        "value": value,
                                        "model_name": model_name,
                                        "pretty_model_name": model_to_pretty_name_dict.get(model_name, model_name),
                                        "is_amalia_model": is_amalia_model(model_name),
                                        "is_open_source_model": is_open_source(model_name),
                                        "task_complete": task_complete_name + suffix,
                                        "pretty_task_name": turn_task_to_pretty_name(task_complete_name) + suffix,
                                        "is_pt_dataset": is_pt_dataset(task),
                                        "is_dev_task": is_dev_task(task),
                                        "is_thinking": enable_thinking,
                                        "tags": add_tags_to_datasets(task),
                                        "date": date,
                                        "file_path": file,
                                    })

                                    results_folder.add(os.path.dirname(file_path))

                    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
                        print(f"Error processing {file_path}: {e}")

    results_df = pd.DataFrame(results_list)
    if only_most_recent:
        # when task and model_name is the same keep the most recent result
        results_df = results_df.sort_values("date", ascending=False).drop_duplicates(["task_complete", "model_name", "metric"])
        # put in original order
        results_df = results_df.sort_index()

    # filter by include_models
    if include_models is not None:
        results_df = results_df[results_df["model_name"].isin(include_models)]
        # also filter by pretty_model_name if it exists in the df
        if "pretty_model_name" in results_df.columns:
            results_df = results_df[results_df["pretty_model_name"].isin(include_models)]

    # Sort columns to ensure the task column is first and checkpoints are sorted
    checkpoint_rows = results_df["model_name"].unique()

    def extract_ckpt_num(model):
        match = re.search(r'checkpoint-(\d+)', model)
        return int(match.group(1)) if match else float('inf')

    sorted_checkpoint_rows = sorted(checkpoint_rows, key=extract_ckpt_num)

    # Reorder the DataFrame based on sorted model names
    results_df['model_name'] = pd.Categorical(results_df['model_name'], categories=sorted_checkpoint_rows, ordered=True)
    results_df = results_df.sort_values('model_name')

    # filter by exclude_models
    if exclude_models is not None:
        results_df = results_df[~results_df["model_name"].isin(exclude_models)]
        # also filter by pretty_model_name if it exists in the df
        if "pretty_model_name" in results_df.columns:
            results_df = results_df[~results_df["pretty_model_name"].isin(exclude_models)]

    # Filter checkpoints based on parameters
    if checkpoints is not None or min_checkpoint is not None or max_checkpoint is not None:
        filtered_rows = []

        # Parse checkpoints parameter if provided
        selected_checkpoints = None
        if checkpoints is not None:
            selected_checkpoints = set(int(x.strip()) for x in checkpoints.split(','))

        for row in sorted_checkpoint_rows:
            match = re.search(r'checkpoint-(\d+)', row)
            if match:
                ckpt_num = int(match.group(1))
                # Include checkpoint if it matches the criteria
                if selected_checkpoints is not None:
                    # Use specific checkpoints list
                    if ckpt_num in selected_checkpoints:
                        filtered_rows.append(row)
                else:
                    # Use minimum and/or maximum checkpoint threshold
                    include = True
                    if min_checkpoint is not None and ckpt_num < min_checkpoint:
                        include = False
                    if max_checkpoint is not None and ckpt_num > max_checkpoint:
                        include = False
                    if include:
                        filtered_rows.append(row)
            else:
                # Keep columns that don't have checkpoint numbers
                filtered_rows.append(row)

        results_df = results_df[results_df["model_name"].isin(filtered_rows)]

    return results_df


def simplify_results_df(df: pd.DataFrame, avoid_subcategories: bool) -> pd.DataFrame:
    # Initialize a list to store selected rows
    selected_rows = []

    for _, row in df.iterrows():
        # Skip rows where the metric contains 'stderr'
        if "stderr" in row["metric"]:
            continue

        # avoid sublasses
        if avoid_subcategories and "is_subcategory" in row and row["is_subcategory"]:
            continue

        # avoid non main metrics
        if "is_main_metric" in row and not row["is_main_metric"]:
            continue

        selected_rows.append(row)


    # Convert list of rows to DataFrame
    simple_df = pd.DataFrame(selected_rows)

    # if task contains bbh_cot_fewshot and _mt_pt and metric is exact_match,bbh-pt-answer-extractor remove
    simple_df = simple_df[~((simple_df["task"].str.contains("bbh_cot_fewshot")) & (simple_df["task"].str.contains("_mt_pt")) & (simple_df["metric"] == "exact_match,bbh-pt-answer-extractor"))]

    # if task contains pt_exams_cot and metric is exact_match,get-answer remove
    simple_df = simple_df[~((simple_df["task"].str.contains("pt_exams_cot")) & (simple_df["metric"] == "exact_match,get-answer"))]

    # check if all task_complete have the same main metric
    task_metric_check = simple_df.groupby("task_complete")["metric"].nunique()
    tasks_with_multiple_metrics = task_metric_check[task_metric_check > 1]
    if not tasks_with_multiple_metrics.empty:
        print("Warning: The following tasks have multiple main metrics:")
        print(tasks_with_multiple_metrics)

        # print indices of the rows that have multiple metrics for the same task_complete
        for task in tasks_with_multiple_metrics.index:
            indices = simple_df[simple_df["task_complete"] == task].index.tolist()
            print(f"Task: {task}, Indices: {indices}")

    # Pivot to get tasks as index, models as columns, and values as cell values
    simple_df = simple_df.pivot(index="task_complete", columns="model_name", values="value")

    # Sort columns to ensure the task column is first and checkpoints are sorted
    checkpoint_cols = simple_df.columns
    def extract_ckpt_num(col):
        match = re.search(r'checkpoint-(\d+)', col)
        return int(match.group(1)) if match else float('inf')
    sorted_checkpoint_cols = sorted(checkpoint_cols, key=extract_ckpt_num)
    simple_df = simple_df[sorted_checkpoint_cols]

    return simple_df


def main():
    """
    Main function to execute the script.
    """
    parser = argparse.ArgumentParser(description="Process JSON results and create a LaTeX table.")
    parser.add_argument("--folder_path", type=str, nargs="+", required=True, help="List of folder paths containing JSON result files.")
    parser.add_argument("--output_path", type=str, default="", help="Path to save the CSV file (optional).")
    parser.add_argument("--only_most_recent", action="store_true", help="Keep only the most recent result for each task and model.")
    parser.add_argument("--ignore_subcategories", action="store_true", help="Ignore subcategories in task names.")
    parser.add_argument("--plot_graphs", action="store_true", help="Whether to plot graphs for each model and task.")

    # filtering options for the graphs
    # include and exclude models (use the name in model_name column or pretty name)
    parser.add_argument("--include_models", type=str, nargs="+", default=None, help="If passed only include these models in the final results.")
    parser.add_argument("--exclude_models", type=str, nargs="*", default=None, help="If passed exclude these models from the final results.")
    parser.add_argument("--min_checkpoint", type=int, default=None,
                        help="Minimum checkpoint number to include (e.g., 500 to include only checkpoint-500 and above)")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated list of checkpoint numbers to include (e.g., '100,300,500')")
    parser.add_argument("--max_checkpoint", type=int, default=None,
                        help="Maximum checkpoint number to include (e.g., 500 to include only up to checkpoint-500)")

    # include and exclude datasets (use the name in task_complete column or pretty name)
    parser.add_argument("--include_datasets", type=str, nargs="+", default=None, help="If passed only include these datasets in the final results. ")
    parser.add_argument("--exclude_datasets", type=str, nargs="*", default=None, help="If passed exclude these datasets from the final results. ")
    parser.add_argument("--dev_only", action="store_true", help="If passed include only dev tasks in the final results. ")

    args = parser.parse_args()

    # Extract results from JSON files
    df = extract_results(
        folder_paths=args.folder_path,
        only_most_recent=args.only_most_recent,
        ignore_subcategories=args.ignore_subcategories,
        include_models=args.include_models,
        exclude_models=args.exclude_models,
        min_checkpoint=args.min_checkpoint,
        max_checkpoint=args.max_checkpoint,
        checkpoints=args.checkpoints,
        include_datasets=args.include_datasets if not args.dev_only else DEV_TASKS ,
        exclude_datasets=args.exclude_datasets
    )

    # if df is empty raise an error
    if df.empty:
        raise ValueError(f"No results found in the provided folder {args.folder_path}")

    # simplify the results df
    simple_df = simplify_results_df(df, avoid_subcategories=True)
    print("\nSimplified Results DataFrame:\n")
    print(simple_df)
    print()

    # simple df with subcategories
    simple_df_with_subcategories = simplify_results_df(df, avoid_subcategories=False)

    # Save results to CSV file
    if args.output_path:

        # create dir if it does not exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        df.to_csv(args.output_path, index=False)
        print(f"\nResults saved to {args.output_path}")

        # do the same for the simple_df
        simple_df.to_csv(args.output_path.replace(".csv", "_simple.csv"))
        print(f"\nSimplified Results saved to {args.output_path.replace('.csv', '_simple.csv')}")

        # do the same for the simple_df_with_subcategories
        simple_df_with_subcategories.to_csv(args.output_path.replace(".csv", "_simple_with_subcategories.csv"))
        print(f"\nSimplified Results (with subcategories) saved to {args.output_path.replace('.csv', '_simple_with_subcategories.csv')}")

        # create tables
        create_tables(args.output_path.replace(".csv", "_simple.csv"), outfile="results_by_model_ordered.csv", extra_tables=False)
        print("\n Tables created.\n")

        # create tables for the simple_df_with_subcategories
        create_tables(args.output_path.replace(".csv", "_simple_with_subcategories.csv"), outfile="results_by_model_ordered_subcategories.csv")
        print("\n Tables (with subcategories) created.\n")

        # plot the graphs
        if args.plot_graphs:
            print("Plotting graphs per model and task...")
            create_graph_per_model_task(csv_results_path=args.output_path, output_folder=os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
    pass


# Example usage:
# python3 scripts/create_result_tables.py --folder_path "./output" --only_most_recent --output_path "./output/results.csv"
