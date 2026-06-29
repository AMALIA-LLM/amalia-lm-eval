# MATH-PT

Adaptation of MATH-PT MCQ set to llm-eval-harness.

Original data location: https://huggingface.co/datasets/tiagoteixeira03/MATH-PT

## Tasks

This directory contains evaluation tasks for the MATH-PT dataset, which provides Portuguese translations (Brazilian and European) of mathematics competition problems.

### Multiple Choice (MCQ) Tasks

#### Available Tasks

- `math_pt_br_mcq`: Brazilian Portuguese multiple choice questions (638 examples)
- `math_pt_pt_mcq`: European Portuguese multiple choice questions (419 examples)
- `math_pt_pt_mcq_no_figures`: European Portuguese MCQ excluding examples with figures (277 examples)

All tasks use chain-of-thought prompting where the model is asked to show its reasoning before selecting the final answer in the format `\boxed{X}`.

**Note on filtering**: Only the `ptpt_multiple_choice` dataset includes `question_has_figure` and `solution_has_figure` metadata fields. The `math_pt_pt_mcq_no_figures` task filters out 142 examples (34%) where either field is `True`, keeping only 277 examples with no figures. The `ptbr_multiple_choice` dataset does not include figure metadata, so only the full version is available.

### Metrics

- `exact_match`: Measures if the extracted answer matches the correct letter exactly

### Notes

- Tasks use a custom answer extractor (`ExtractAnswerMathPT`) that looks for answers in `\boxed{}` format or common answer patterns
- The open-ended versions (`ptbr_open_ended` and `ptpt_open_ended`) are not yet implemented in harness check the original paper