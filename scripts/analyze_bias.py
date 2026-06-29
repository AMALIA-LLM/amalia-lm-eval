import json
import os
from collections import defaultdict, Counter
from typing import List


def analyze_bias_in_folder(folder_path: str, prefix_tasks: List[str] = None):
    """
    Analyze bias for certain options in MCQ benchmarks
    """

    if prefix_tasks is None:
        prefix_tasks = [
            "mmlu",
            "amalia_dsl_tl_pt_mcq_shuffled_no_na",
            "amalia_pt_text_completion",
            "amalia_ptbrvid_mcq_shuffle",
            "amalia_truthfulqa_mc2_mt_pt",
            "piqa-mt-pt",
            "pt_exams",
        ]

    task_counters = defaultdict(Counter)

    for file in os.listdir(folder_path):
        if file.endswith(".jsonl") and any(prefix in file for prefix in prefix_tasks):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    resps = sample["resps"]  # list like [[['score', 'flag']], ...]

                    # Flatten properly: extract (float(score), flag)
                    parsed_resps = []
                    for r in resps:
                        if isinstance(r, list) and len(r) > 0 and isinstance(r[0], list) and len(r[0]) == 2:
                            score_str, flag = r[0]
                            parsed_resps.append((float(score_str), flag))
                        else:
                            # unexpected format
                            continue

                    if not parsed_resps:
                        continue

                    # Priority 1: any True flags
                    true_indices = [i for i, (_, flag) in enumerate(parsed_resps) if flag == "True"]
                    if len(true_indices) == 1:
                        task_counters[file][f"resp_{true_indices[0]}"] += 1
                        continue
                    elif len(true_indices) > 1:
                        for idx in true_indices:
                            task_counters[file][f"resp_{idx}"] += 1
                            break  # if multiple True, count first one
                        continue

                    # Priority 2: highest score
                    scores = [s for s, _ in parsed_resps]
                    max_score = max(scores)
                    winners = [i for i, (s, _) in enumerate(parsed_resps) if s == max_score]

                    if len(winners) == 1:
                        task_counters[file][f"resp_{winners[0]}"] += 1
                    else:
                        # tie — optional logging
                        print(f"Warning: tie in {file}")
                        continue

    # Reporting
    for task, counter in task_counters.items():
        total = sum(counter.values())
        print(f"Task: {task}")
        for resp, count in counter.items():
            ratio = count / total if total else 0
            print(f"  {resp}: {count} ({ratio:.2%})")

        if total > 0:
            top_resp, top_count = counter.most_common(1)[0]
            if top_count / total > 0.8:
                print(f"  ALERT: {top_resp} is dominant with {top_count / total:.2%}")
        print()

    # Reporting results
    alerts = []
    for task, counter in task_counters.items():
        total = sum(counter.values())
        print(f"Task: {task}")
        # Sort keys alphabetically before reporting
        for resp in sorted(counter.keys()):
            count = counter[resp]
            ratio = count / total if total else 0
            print(f"  {resp}: {count} ({ratio:.2%})")

        # Dominance check
        if total > 0:
            most_common_resp, most_common_count = counter.most_common(1)[0]
            if most_common_count / total > 0.8:
                print(f"  ALERT: {most_common_resp} is dominant with {most_common_count / total:.2%}")
                alerts.append((task, most_common_resp, most_common_count / total))
        print()

    print(f"Total Bias Alerts: {len(alerts)}")
    for task, resp, ratio in alerts:
        print(f"  Task: {task}, Dominant Response: {resp}, Ratio: {ratio:.2%}")


def main():

    # this path should point to a folder with jsonl results from LM eval harness
    # change this
    folder_path = "/user/home/username/eval_outputs/model_results"
    analyze_bias_in_folder(folder_path)


if __name__ == "__main__":
    main()
