# Calame-PT evaluation

This folder contains scripts and helpers to evaluate next-word generation on the [Calame-PT dataset ](https://huggingface.co/datasets/NOVA-vision-language/calame-pt)(Portuguese) from [GlórIA — A Generative and Open Large Language Model for Portuguese](https://arxiv.org/abs/2402.12969).

High-level summary
- Task: given a sentence with a masked/omitted last word, prompt a model with the sentence and consider the model's first generated word as its prediction for the missing last word; compute exact-match (EM) between the normalized predicted first word and the dataset's target last word.
- Dataset: `NOVA-vision-language/calame-pt` (Hugging Face dataset). The code evaluates two subsets: `generated` and `handwritten`.
