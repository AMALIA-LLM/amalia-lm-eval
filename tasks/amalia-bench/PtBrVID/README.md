# Portuguese Variant Identification Task using the PtBrVID Dataset

## Task Overview

This task aims to evaluate the ability of Large Language Models (LLMs) to identify the specific Portuguese variant of a given sentence. We utilize the `PtBrVID` dataset from Hugging Face which contains sentences annotated with their respective Portuguese dialectal variations.

## Dataset

The dataset used for this evaluation is:

-   **Name:** `PtBrVID` [original location](https://huggingface.co/datasets/liaad/PtBrVId)
-   **Description:** This dataset comprises sentences in Portuguese, each labeled with PT-PT or PT-BR.

## Task Definition

We use a multiple choice format to evaluate the model's performance. The task is to select the correct Portuguese variant (PT-PT or PT-BR) for a given sentence.
We use the `validation` set of the dataset since it has 1000 samples for each topic category with a approximately balanced distribution of labels.

Each dataset instance consists of:
-  **text**: A sentence in a variant of Portuguese.
-  **label**: The label for the variant 0 - PT-PT, 1 - PT-BR

### Example prompt
```
Qual é a variante de português utilizada neste texto: {{text}} 
PT-PT para a variante de português de Portugal e PT-BR para a variante de português do Brasil.
Escolha a opção correta:
{{choices_str}}
Resposta:
```

### Target Output
The model should return `A`, `B` corresponding to the correct variant of portuguese in use.

### Task Version

Two versions of the task are available:
- **amalia_ptbrvid_mcq_shuffle** - for base models
- **amalia_ptbrvid_mcq_shuffle_chat** - use this version for instruction-tuned models with a chat template.

## Processing Logic
The answer choices are shuffled for each instance using the `process_brvid_docs` function from `utils.py`, ensuring that the order does not remain static across the dataset.

## Evaluation Metrics
The task is evaluated based on:
-   **Accuracy (`acc`)**: Measures the proportion of correct answers.
