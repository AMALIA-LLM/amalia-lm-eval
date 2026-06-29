# P3B3: A Multi-Turn Conversational Benchmark for Measuring European and Brazilian Portuguese Variety Bias in LLMs

[![Code](https://img.shields.io/badge/GitHub-P3B3%20Benchmark-blue?logo=github)](https://github.com/AMALIA-LLM/p3b3-benchmark)
[![Paper: ACL Anthology](https://img.shields.io/badge/Paper-Link-red)](https://arxiv.org/abs/2606.16753)
[![MeLLM @ ACL 2026](https://img.shields.io/badge/Workshop-MeLLM%20%40%20ACL%202026-green)](https://www.mellm.org/)


This is an older version of the codebase, for evaluation with the P3B3 benchmark please refer to the **updated [repository](https://github.com/AMALIA-LLM/p3b3-benchmark)**.

P3B3 is a benchmark for evaluating language model biases toward Portuguese language variants (European Portuguese vs Brazilian Portuguese) through multi-turn conversation generation and automated assessment.

The benchmark is introduced in the paper: [P3B3: A Multi-Turn Conversational Benchmark for Measuring European and Brazilian Portuguese Variety Bias in LLMs](https://arxiv.org/abs/2606.16753), accepted at the MeLLM Workshop @ ACL 2026.


## Notes

- This is an **older** version of the P3B3 codebase, please refer to the **updated repository** for the latest code and instructions: [repository](https://github.com/AMALIA-LLM/p3b3-benchmark).
- All examples with model name used Qwen, but could be any hugging face or local model
- You'll need a .env with a Gemini api key for the llm scores


## Available Scripts

- **`generate.py`**: Generates model responses (normal, PT-PT, PT-BR)
- **`score.py`**: Classifies linguistic variant using HuggingFace models
- **`llm_score.py`**: Evaluates response quality using LLM (Gemini)
- **`run_pipeline.py`**: Orchestrates script execution

## Pipeline Usage

### Run Full Pipeline
```bash
# With pipeline
python run_pipeline.py --generate --class-score --llm-score \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --output-dir pt-pt-eval

# On cluster (requires GPU for generation)
srun --pty --gres=gpu:1 --mem 50G python run_pipeline.py --generate --class-score --llm-score \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --output-dir pt-pt-eval
```

### Generate Only
```bash
# With pipeline
python run_pipeline.py --generate \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --output-dir pt-pt-eval

# On cluster (requires GPU)
srun --pty --gres=gpu:1 --mem 50G python run_pipeline.py --generate \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --output-dir pt-pt-eval
```

### Classify with HuggingFace Only
```bash
# With pipeline
python run_pipeline.py --class-score --base-dir pt-pt-eval

# On cluster (requires GPU)
srun --pty --gres=gpu:1 --mem 50G python run_pipeline.py --class-score --base-dir pt-pt-eval
```

### Evaluate with LLM Only
```bash
# With pipeline (CPU only)
python run_pipeline.py --llm-score --base-dir pt-pt-eval

# On cluster (CPU only)
srun --pty --mem 10G python run_pipeline.py --llm-score --base-dir pt-pt-eval
```

### Generate + Classify
```bash
# With pipeline
python run_pipeline.py --generate --class-score \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --output-dir pt-pt-eval

# On cluster (requires GPU)
srun --pty --gres=gpu:1 --mem 50G python run_pipeline.py --generate --class-score \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --output-dir pt-pt-eval
```

### Export CSVs from LLM Database
```bash
# With pipeline (CPU only)
python run_pipeline.py --export-llm --base-dir pt-pt-eval

# On cluster (CPU only)
srun --pty --mem 10G python run_pipeline.py --export-llm --base-dir pt-pt-eval
```

## Individual Script Usage

### Response Generation
```bash
# Direct
python generate.py --model-name-or-path Qwen/Qwen2.5-7B-Instruct --base-output-dir pt-pt-eval

# On cluster (requires GPU)
srun --pty --gres=gpu:1 --mem 50G python generate.py \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --base-output-dir pt-pt-eval
```

### HuggingFace Classification
```bash
# Direct
python score.py pt-pt-eval

# On cluster (requires GPU)
srun --pty --gres=gpu:1 --mem 50G python score.py pt-pt-eval
```

### LLM Evaluation
```bash
# Direct (CPU only)
python llm_score.py pt-pt-eval

# On cluster (CPU only)
srun --pty --mem 10G python llm_score.py pt-pt-eval
```

## Output Structure

```
pt-pt-eval/
├── *.json                    # Model responses
├── class_scores/             # HF classifier scores
│   └── *_scored_v1.csv
└── llm_scores/               # LLM scores and DB
    ├── pt_pt_conversation_evaluations.db
    └── *_llm_scored.csv
```

## Supported Models

- **Local**: Any HuggingFace model compatible with vLLM
- **APIs**: 
  - `google/gemini-2.0-flash-exp`
  - `openai/gpt-4o-mini`

## Requirements

- GPU for local models (`--gres=gpu:1`)
- Environment variables: `GEMINI_API_KEY`, `OPENAI_API_KEY`
- Dependencies: `vllm`, `transformers`, `google-genai`, `openai`



## Citation

If you find this work relevant please cite:

```bibtex
@inproceedings{ferreira_p3b3,
      title={{P3B3}: A Multi-Turn Conversational Benchmark for Measuring European and Brazilian Portuguese Variety Bias in {LLMs}}, 
      author={Rafael Ferreira and Inês Vieira and Inês Calvo and James Furtado and Iago Paulo and Diogo Tavares and Diogo Glória-Silva and David Semedo and João Magalhães},
      booktitle={Proceedings of the 1st Workshop on Multilinguality in the Era of Large Language Models (MeLLM)},
      year={2026},
      publisher={Association for Computational Linguistics},
      eprint={2606.16753},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.16753}, 
}
```