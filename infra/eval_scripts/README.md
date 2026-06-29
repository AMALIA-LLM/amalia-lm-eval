# Run Helpers for evaluations

This folder contains helper scripts used to schedule and run model evaluations on using the `AMALIA` evaluation CLI.

**Files:**
- [run_base_dev.sh](run_base_dev.sh) — base-model dev evaluation tasks
- [run_base_heldout.sh](run_base_heldout.sh) — base-model held-out evaluation tasks
- [run_chat_dev.sh](run_chat_dev.sh) — instruction-tuned models dev evaluation tasks
- [run_chat_dev_think_mode.sh](run_chat_dev_think_mode.sh) — instruction-tuned models dev evaluation tasks with thinking mode
- [run_chat_heldout.sh](run_chat_heldout.sh) — instruction-tuned models held-out evaluation tasks
- [run_chat_coding.sh](run_chat_coding.sh) — instruction-tuned models coding evaluation tasks
- [run_chat_long_context.sh](run_chat_long_context.sh) — instruction-tuned models long context evaluation tasks
- [run_chat_safety.sh](run_chat_safety.sh) — instruction-tuned models safety evaluation tasks
- [util_env.sh](util_env.sh) — shared environment setup (loads .env, activates Python environment)
- [util_functions.sh](util_functions.sh) — shared utility functions used across all scripts


### Setting up `.env` (required)
The run scripts in this folder expect a project-level `.env` file to be present at the repository root. To help, a sample file is included named `.env_copy`.

Follow these steps to create and configure your `.env` file:

1. Copy the sample file and rename it to `.env` at the project root:

```bash
cp .env_copy .env
```

2. Open the newly created `.env` and update the variables for your environment. The key variables are:

- `ACTIVATE_ENV` — full path to the virtualenv/conda `activate` script (e.g. `/home/you/envs/py312_amalia/bin/activate`). This is required; the scripts will stop with an error if it's missing or the path does not exist.
- `DEFAULT_EVAL_OUTPUT_DIR` — default evaluation outputs directory (used when you don't pass `-o/--output`). This is required unless you always pass `-o` on the command line.
- `HF_CACHE_REMOTE` — base path to your HuggingFace cache directory; the scripts use this to access HF information and clean up dataset lock files.

Example `.env` snippet (already provided in `.env_copy`):


## Suggested quick checklist before running
- [ ] Update the `.env` file
- [ ] Ensure `amalia` CLI is installed and on PATH (check top level [README.md](../../README.md) for install instructions if needed)

**Examples:**
- Evaluate all checkpoints in a model directory (default output dir):
```
bash infra/eval_scripts/run_chat_dev.sh "/path/to/model/root"`
```

- Evaluate a single model (pass the model_path_directly):
```
bash infra/eval_scripts/run_chat_dev.sh "/path/to/model/root/checkpoint-1092"
```


**High-level behavior:**
- Each script discovers checkpoint directories under a given model root (directories named `checkpoint-<num>`), or accepts a direct checkpoint path.
- By default, scripts will skip any checkpoint whose output directory already contains `.json` / `.jsonl` results (to avoid redundant reruns). Use a custom `-o` output directory to override this behavior.
- Scripts schedule `amalia` jobs via Slurm (they call the `amalia` CLI which itself enqueues Slurm jobs).
- Scripts activate a local Python environment at the top of the script — **UPDATE** that path (`source /home/unl/.../py312_amalia/bin/activate`) to match your environment if needed.

**Common options:**
- <model_path(s)> — one or more model root paths or a direct checkpoint path. If a checkpoint path (contains `checkpoint-` or a folder containing `.safetensors` files) is passed, the script will evaluate that single checkpoint.
- `-n <NUM_RECENT>` — only evaluate the N most recent checkpoints.
- `-c id1 id2 ...` — evaluate specific checkpoint ids (concatenated as `checkpoint-<id>` under the provided model path).
- `-o` / `--output <OUTPUT_DIR>` — set custom output directory. When omitted, scripts use the default configured OUTPUT_DIR and build subfolders using the model and checkpoint names.


**Practical tips and troubleshooting:**
- Update the virtualenv activation path: the top of each script activates a local environment using `source /home/unl/.../py312_amalia/bin/activate`. **UPDATE** it to your conda/venv location.
- `amalia` missing: if `amalia` is not on PATH, install the package (in dev mode). Check the top-level [README.md](../../README.md) for instructions.
- Locks in HF datasets cache: scripts remove `*.lock` files in a hard-coded datasets cache path — change or remove the `find ... -delete` line if it doesn't match your environment.
- Queue size heuristics: tune `MAX_SLURM_JOBS`, `JOBS_PER_CHECKPOINT`, and `BUFFER_JOBS` to match your cluster's fair use policies.


## Quick Visualization Script
This project includes [create_result_tables.py](../../scripts/create_result_tables.py), a small utility to aggregate and simplify JSON evaluation outputs produced by `amalia` cli and to  create csv tables and graphs.

**Example usage:**
```bash
# Aggregate results in ./output/instruct_pt and write a CSV + graphs
python3 scripts/create_result_tables.py --folder_path </path/to/output/instruct_pt> \
    --only_most_recent --output_path </path/to/output/instruct_pt/results.csv> --plot_graphs
```

**What it does:**
- Recursively scans one or more folders for files named `results_*.json` and loads them.
- Extracts metadata (model name, date parsed from filename when available, whether a chat template or few-shot was used) and per-task metrics.
- Produces a consolidated CSV with one row per (task, metric, model) and also creates simplified pivoted CSVs for easier inspection.

**Important script flags:**
- `--folder_path` (required): one or more folders to scan recursively for `results_*.json` files.
- `--output_path` (optional): path where the combined CSV is written. It needs to end with **.csv**. If provided, the script also writes two derived CSVs with suffixes `_simple.csv` in the same location.
- `--only_most_recent`: keep only the most recent result per (task, model, metric) when multiple files exist (this should always be used to avoid duplicates).
- `--ignore_subcategories`: when set, only the top-level task entry found in a results JSON is kept (useful when you only want the main task aggregated row).
- `--plot_graphs`: if set, the script calls the graphing helper to generate per-model/per-task plots into the directory containing `--output_path`.

**Filtering options:**
- `--include_models` — list of exact model names to keep (the `model_name` or `pretty_model_name` field).
- `--exclude_models` — list of model names to exclude (the `model_name` or `pretty_model_name` field).
- `--min_checkpoint` / `--max_checkpoint` — numeric checkpoint thresholds; keeps only models whose name contains `checkpoint-<num>` within the range.
- `--checkpoints` — comma-separated list of checkpoint numbers to include (e.g. `--checkpoints 100,300,500`).
- `--include_datasets` — list of dataset names (the `task_complete` or `pretty_task_name` field) to include in the aggregation.
- `--exclude_datasets` — list of dataset names (the `task_complete` or `pretty_task_name` field) to exclude from the aggregation.

**Outputs produced:**
- `<output_path>` — full aggregated CSV (one row per task/metric/model) saved at the path you set.
- `<output_path.replace('.csv', '_simple.csv')>` — simplified pivot table with tasks as rows and models as columns (main metric per task).
- `<output_path.replace('.csv', '_simple_with_subcategories.csv')>` — same as above but preserving subcategories.
- ProPor tables:  It will produce the repository's ProPor table artifacts
- Graphs (optional): if `--plot_graphs` is set, plots are generated by `create_graph_per_model_task` into the folder that contains your `--output_path`.

To set **pretty names** for tasks and models check [model_task_pretty_names.py](../../scripts/model_task_pretty_names.py) and add the task to `dataset_to_prety_name_dict` and model to `model_to_pretty_name_dict` dictionary.