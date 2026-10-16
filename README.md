<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/AMALIA-LLM/amalia-llm.github.io/blob/main/source/_static/logo/logo-color-white.png?raw=true" width="300">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/AMALIA-LLM/amalia-llm.github.io/blob/main/source/_static/logo/logo-color-black.png?raw=true" width="300">
  <img src="https://github.com/AMALIA-LLM/amalia-llm.github.io/blob/main/source/_static/logo/logo-color-black.png?raw=true" alt="AMALIA" width="300"/>
</picture>
<br/>

**A Fully Open Large Language Model for European Portuguese**

[![Website](https://img.shields.io/badge/Website-amaliallm.pt-blue)](https://amaliallm.pt/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-amalia--llm-yellow)](https://huggingface.co/amalia-llm)
[![Paper](https://img.shields.io/badge/Paper-Technical%20Report-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.26511)
 
</div> 


---

# AMALIA LM Eval

This repository contains the code for evaluating the AMALIA language model, using as basis the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## AMALIA-Bench

This repository includes **AMALIA-Bench**, a comprehensive benchmark suite for evaluating Large Language Models with a focus on European Portuguese (PT-PT). 

The benchmark spans multiple evaluation categories including general knowledge, commonsense reasoning, mathematics, natural language understanding, reading comprehension, code generation, instruction following, Portuguese variant identification, translation, safety, and long-context tasks.

AMALIA-Bench includes both machine-translated datasets from English and original Portuguese datasets. 
For detailed information about the benchmark, see [tasks/amalia-bench/README.md](tasks/amalia-bench/README.md).

This evaluation framework also supports all tasks available in the original lm-eval-harness.

## Setup

Run this bash script to create a functioning environment (we use conda and CUDA 12.4):

```bash
CONDA_HOME=<path_to_conda> ENV_NAME=py312_amalia ./infra/create_conda-env.sh
```

Optionally, the script will ask you to pack your environment so that it can be transfered across machines. The package will contain:
- python 3.12
- cudnn, cuda-nvcc, and cuda-toolkit for CUDA 12.4

### Setting up `.env` (required for run scripts)
The helper run scripts (for example in [eval_scripts](infra/eval_scripts)) expect a project-level `.env` file at the repository root. A sample file named `.env_copy` is included to make setup easy.

1. Copy the sample file and rename it to `.env` at the project root:

```bash
cp .env_copy .env
```

2. Edit `.env` and update the key variables:

- `ACTIVATE_ENV` — full path to your Python virtualenv or conda `activate` script (required). Example: `/home/you/envs/py312_amalia/bin/activate`.
- `DEFAULT_EVAL_OUTPUT_DIR` — default directory for evaluation outputs (required unless you always pass `-o/--output`).
- `HF_CACHE_REMOTE` — base path to your HuggingFace cache.
- `CODE_EVAL_FILE_PATH` - path to [HF code eval](https://github.com/huggingface/evaluate/tree/main/metrics/code_eval) path (you can ignore this path if you have internet connection else replace by corresponding local path)

## Running Evaluations

After having activated the environment, you will have access to the `amalia` command. To evaluate a single task, run:
```bash
amalia --tasks "amalia_arc_challenge_chat_mt_pt" --model_name_or_path amalia-llm/AMALIA-9B-0626-DPO --model_output_dir "./outputs" --slurm_config <slurm_config>
```

Note:
- To reproduce our runs, check the helper scripts in `infra/eval_scripts` - Additional Information [here](infra/eval_scripts/README.md).
- You will have to create or adapt a **SLURM config class** to run eval on your infrastructure. Go to `configs/slurm.py` and add/adapt a class there.

#### Loading models that require custom code (`trust_remote_code`)

By default, `trust_remote_code` is **disabled**. Some models ship custom modeling code on the Hugging Face Hub that must be executed on load — these will fail to load unless you explicitly opt in:

```bash
amalia "..." --harness-args.trust-remote-code True
```

> ⚠️ Enabling this executes arbitrary code from the model repository on load. Only enable it for repositories you trust.

### Using lm-eval-harness Directly

You can also use the original `lm-eval` command directly by specifying the tasks' folder:

```bash
lm_eval --model hf \
    --model_args pretrained=amalia-llm/AMALIA-9B-0626-DPO \
    --tasks amalia_arc_challenge_chat_mt_pt \
    --include_path tasks/amalia-bench \
    --output_path ./outputs
```

The `--include_path` flag points to the directory containing the AMALIA-Bench task definitions.

### Quick Visualization Script

Use [create_result_tables.py](scripts/create_result_tables.py) to aggregate JSON evaluation outputs into CSV tables and graphs:

```bash
python3 scripts/create_result_tables.py --folder_path </path/to/output> \
    --only_most_recent --output_path </path/to/results.csv> --plot_graphs
```

The script recursively scans folders for `results_*.json` files and produces:
- Full aggregated CSV with one row per (task, metric, model)
- Simplified pivot tables (`_simple.csv`, `_simple_with_subcategories.csv`)
- ProPor table artifacts
- Per-model/per-task graphs (when `--plot_graphs` is set)

**Key flags:** `--folder_path` (required), `--output_path` (optional, must end in `.csv`), `--only_most_recent`, `--plot_graphs`, `--include_models`, `--exclude_models`, `--checkpoints`, `--include_datasets`, `--exclude_datasets`.

To customize display names, edit [model_task_pretty_names.py](scripts/model_task_pretty_names.py).


## Inspect the Results

Our runner scripts generates results divided into model-specific folders. If you are using our scripts or a similar structure, you can inspect scores and model outputs with:

```bash
amalia inspect --root <root-dir>
```

Export a shareable leaderboard:

```bash
amalia inspect --root <results-dir> --export-html ./site
# open ./site/index.html, or publish to GitHub Pages / an HF Space
```

The export captures the default (latest-run) view; the interactive run-mode and
task-mode toggles are only available in the live server. Absolute local paths are
scrubbed from the exported HTML so the site is safe to publish.

## Citation

If you use this code for your research, please consider citing:

```bibtex
@inproceedings{simplicio-etal-2026-amalia,
    title = "{AMALIA}: A Fully Open Large Language Model for {E}uropean {P}ortuguese",
    author = "Simpl{\'i}cio, Afonso and Vinagre, Gon{\c{c}}alo and Ramos, Miguel Moura and Tavares, Diogo and Ferreira, Rafael and Attanasio, Giuseppe and Alves, Duarte M. and Calvo, In{\^e}s and Vieira, In{\^e}s and Guerra, Rui and Furtado, James and Canaverde, Beatriz and Paulo, Iago and Ramos, Vasco and Gl{\'o}ria-Silva, Diogo and Faria, Miguel and Treviso, Marcos and Gomes, Daniel and Gomes, Pedro and Semedo, David and Martins, Andr{\'e} and Magalh{\~a}es, Jo{\~a}o",
    booktitle = "Proceedings of the 17th International Conference on Computational Processing of {P}ortuguese ({PROPOR} 2026) - Vol. 1",
    month = apr,
    year = "2026",
    address = "Salvador, Brazil",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.propor-1.38/",
    pages = "380--391",
    isbn = "979-8-89176-387-6"
}

@misc{simplicio2026amaliatechnicalreportfully,
    title = {AMALIA Technical Report: A Fully Open Source Large Language Model for European Portuguese}, 
    author = {Afonso Simplício and Gonçalo Vinagre and Miguel Moura Ramos and Diogo Tavares and Rafael Ferreira and Giuseppe Attanasio and Duarte M. Alves and Inês Calvo and Inês Vieira and Rui Guerra and James Furtado and Beatriz Canaverde and Iago Paulo and Vasco Ramos and Diogo Glória-Silva and Miguel Faria and Marcos Treviso and Daniel Gomes and Pedro Gomes and David Semedo and André Martins and João Magalhães},
    year = {2026},
    eprint = {2603.26511},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url = {https://arxiv.org/abs/2603.26511}
}
```
