# AMALIA-BENCH: European Portuguese LLM Evaluation Benchmarks

This directory contains Portuguese translated versions of popular and newly created benchmark datasets for evaluating large language models (LLMs) using the `llm-eval-harness` framework.

## Datasets Included in AMALIA-BENCH

| Dataset                    | Category              | Source lang    | Translation type        | Cultural Relevance | Translated with      | Note          | HF URL / Other URL                                                    |
|----------------------------|-----------------------|----------------|-------------------------|--------------|----------------------|---------------|-----------------------------------------------------------------------|
| aime_pt                    | Mathematics           | English        | MT into PT              | No           | Gemma-4 31B-It       | NEW           | https://huggingface.co/datasets/amalia-llm/aime-1983-2024-ptpt        |
| alba_mcq                   | General Knowledge     | PT             |                         | Portugal     |                      | NEW, ORIGINAL | https://huggingface.co/datasets/amalia-llm/alba_mcq                   |
| arc_mt                     | Commonsense Reasoning | English        | MT into PT              | No           | DeepL                |               | https://huggingface.co/datasets/LumiOpen/arc_challenge_mt             |
| bbh_mt*                    | Reasoning             | English        | MT and Rules into PT    | No           | Finetuned GemmaX2-9B | NEW           | https://huggingface.co/datasets/amalia-llm/bigbenchhard-mt-pt         |
| calame                     | Sentence Completion   | PT             |                         | No           |                      |               | https://huggingface.co/datasets/NOVA-vision-language/calame-pt        |
| commonsense_qa_mt**        | Commonsense Reasoning | English        | MT into PT              | No           | Finetuned GemmaX2-9B | NEW           | https://huggingface.co/datasets/amalia-llm/commonsense_qa-mt-pt       |
| cultura_viva_pt_mcq        | Cultural Knowledge    | PT             |                         | Portugal     |                      | NEW, ORIGINAL | https://huggingface.co/datasets/amalia-llm/cultura-viva-pt-mcq        |
| dsl_tl                     | PT-Variant ID         | PT & BR        | Human translation       | Portugal & Brazil |                      |               | https://huggingface.co/datasets/amalia-llm/dsl_tl_fewshot             |
| frmt                       | Translation           | PT & BR        | Human translation       | Portugal & Brazil |                      |               | https://huggingface.co/datasets/hugosousa/frmt                        |
| global_piqa                | Commonsense Reasoning | PT             |                         | Portugal     |                      |               | https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel |
| gsm8k_mt                   | Mathematics           | English        | MT into PT              | No           | DeepL                |               | https://huggingface.co/datasets/LumiOpen/opengpt-x_gsm8kx             |
| hellaswag_mt               | Commonsense Reasoning | English        | MT into PT              | No           | DeepL                |               | https://huggingface.co/datasets/LumiOpen/opengpt-x_hellaswagx         |
| humaneval_en               | Code Generation       | English        |                         | No           |                      |               | https://huggingface.co/datasets/openai/openai_humaneval               |
| humaneval_pt               | Code Generation       | English        |                         | No           | NLLB-200             |               | https://huggingface.co/datasets/amalia-llm/humaneval_mt_pt            |
| ifeval_mt                  | Instruction Following | English        | MT + Human verification | No           | DeepL                | NEW           | https://huggingface.co/datasets/amalia-llm/IFEval-mt-pt               |
| LegalBenchPT               | General Knowledge     | PT             |                         | Portugal     |                      | NEW, ORIGINAL | https://huggingface.co/datasets/BeatrizCanaverde/LegalBench.PT        |
| math-pt                    | Mathematics           | PT             |                         | No           |                      |               | https://huggingface.co/datasets/tiagoteixeira03/MATH-PT               |
| minerva_math_en            | Mathematics           | English        |                         | No           |                      |               | https://huggingface.co/datasets/EleutherAI/hendrycks_math             |
| minerva_math_pt            | Mathematics           | English        | MT into PT              | No           | Gemma-4 31B-It       | NEW           | https://huggingface.co/datasets/amalia-llm/hendrycks-math-ptpt        |
| mmlu_mt                    | General Knowledge     | English        | MT into PT              | No           | DeepL                |               | https://huggingface.co/datasets/LumiOpen/opengpt-x_mmlux              |
| mmlu_prox                  | General Knowledge     | English        | MT into PT              | No           | Claude Sonnet 3.7    |               | https://huggingface.co/datasets/li-lab/MMLU-ProX                      |
| openbookqa_mt              | General Knowledge     | English        | MT into PT              | No           | Finetuned GemmaX2-9B | NEW           | https://huggingface.co/datasets/amalia-llm/openbookqa-mt-pt           |
| piqa_mt                    | Commonsense Reasoning | English        | MT into PT              | No           | Finetuned GemmaX2-9B | NEW           | https://huggingface.co/datasets/amalia-llm/piqa-mt-pt                 |
| pt_completion              | PT-Variant ID         | PT             |                         | Portugal & Brazil |                      | NEW, ORIGINAL | https://huggingface.co/datasets/amalia-llm/pt_text_completion         |
| pt_exams                   | General Knowledge     | PT             |                         | Portugal     |                      | NEW, ORIGINAL | https://huggingface.co/datasets/amalia-llm/pt_exams                   |
| PtBrVID                    | PT-Variant ID         | PT & BR        |                         | Portugal & Brazil |                      |               | https://huggingface.co/datasets/liaad/PtBrVId                         |
| ruler_en                   | Long Context          | English        |                         | No           |                      |               | Multiple datasets (see folder)                                        |
| ruler_pt                   | Long Context          | English        | MT into PT              | No           | Mixed                |               | Multiple datasets (see folder)                                        |
| simple_safety_tests        | Safety                | English        |                         | No           |                      |               | https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests        |
| simple_safety_tests_pt     | Safety                | English        | MT into PT              | No           | Finetuned GemmaX2-9B | NEW           | https://huggingface.co/datasets/amalia-llm/sst_pt-pt                  |
| multilingual_advbench      | Safety                | Multi-language |                         | No           |                      |               | https://huggingface.co/datasets/simonycl/multilingual_advbench        |
| xstest                     | Safety                | English        |                         | No           |                      |               | https://huggingface.co/datasets/Paul/XSTest                           |
| xstest_pt                  | Safety                | English        | MT into PT              | No           | Finetuned GemmaX2-9B | NEW           | https://huggingface.co/datasets/amalia-llm/xstest_ptpt                |
| wildguardtest              | Safety                | English        |                         | No           |                      |               | https://huggingface.co/datasets/walledai/WildGuardTest                |
| wild_guard_test_pt         | Safety                | English        | MT into PT              | No           | Gemma-4 31B-It       | NEW           | https://huggingface.co/datasets/amalia-llm/wildguardmix-ptpt          |
| harmbench_standard         | Safety                | English        |                         | No           |                      |               | https://huggingface.co/datasets/walledai/HarmBench                    |
| harmbench_standard_pt      | Safety                | English        | MT into PT              | No           | Gemma-4 31B-It       | NEW           | https://huggingface.co/datasets/amalia-llm/harmbench-ptpt             |
| harmbench_contextual       | Safety                | English        |                         | No           |                      |               | https://huggingface.co/datasets/walledai/HarmBench                    |
| harmbench_contextual_pt    | Safety                | English        | MT into PT              | No           | Gemma-4 31B-It       | NEW           | https://huggingface.co/datasets/amalia-llm/harmbench-ptpt             |
| toxic_chat                 | Safety                | English        |                         | No           |                      |               | https://huggingface.co/datasets/lmsys/toxic-chat                      |
| toxic_chat_pt              | Safety                | English        | MT into PT              | No           | Gemma-4 31B-It       | NEW           | https://huggingface.co/datasets/amalia-llm/toxichat-ptpt              |
| saudade-pt                 | Temporal Reasoning    | PT             |                         | Portugal     |                      | NEW, ORIGINAL | https://huggingface.co/datasets/amalia-llm/saudade-pt                 |
| social_iqa-mt-pt           | Commonsense Reasoning | English        | MT into PT              | No           | Finetuned GemmaX2-9B | NEW           | https://huggingface.co/datasets/amalia-llm/siqa-mt-pt                 |
| truthfulqa_mt              | NLU                   | English        | MT into PT              | No           | DeepL                |               | https://huggingface.co/datasets/LumiOpen/opengpt-x_truthfulqax        |
| winogrande_mt*             | NLU                   | English        | MT into PT              | No           | Finetuned GemmaX2-9B | NEW           | https://huggingface.co/datasets/amalia-llm/winogrande-mt-pt           |

**Notes:**
- NEW - newly translated version of an existing dataset
- ORIGINAL - original dataset created specifically for AMALIA-BENCH
- \* Partial translations - some tasks are not transferable to Portuguese
- \*\* Not advised for use due to evaluation issues



## Usage with `llm-eval-harness`

The use should be the same as to the original datasets so for more information check the README of each task.

## Contributing

If you want to contribute with a new dataset or a new translation of an existing dataset, please do a pull request with the dataset and a README with the information about the dataset, translation method and source.

Follow the same best practices as in the LM-Eval-Harness repository.
