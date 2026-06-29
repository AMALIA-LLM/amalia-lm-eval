# Portuguese Variant Identification Task Using the `dsl_tl` Dataset


## Task Overview

This task aims to evaluate the ability of Large Language Models (LLMs) to identify the specific Portuguese variant of a given sentence. We utilize the `dsl_tl` dataset from Hugging Face (`LCA-PORVID/dsl_tl`), which contains sentences annotated with their respective Portuguese dialectal variations.


## Dataset

The dataset used for this evaluation is:

-   **Name:** `dsl_tl` [original location](https://github.com/LanguageTechnologyLab/DSL-TL/tree/main)
-   **Source:** [LCA-PORVID/dsl_tl](https://huggingface.co/datasets/LCA-PORVID/dsl_tl)
-   **Paper:** [Language Variety Identification with True Labels](https://aclanthology.org/2024.lrec-main.882/)
-   **Description:** This dataset comprises sentences in Portuguese, each labeled with PT-PT, PT-BR or N/A if not possible to determine the dialectal variant.


## Task Definition

There are 3 different tasks in a multiple choice format:

| Task Name                          | Number of Labels | Labels               | Shuffled Order |   Number of fewshot examples |
|------------------------------------|------------------|----------------------|----------------|------------------------------|
| amalia_dsl_tl_pt_mcq               | 3                | PT-BR, PT-PT, N/A    | No             |  0                           |
| amalia_dsl_tl_pt_mcq_shuffled      | 3                | PT-BR, PT-PT, N/A    | Yes            |  0                           |
| amalia_dsl_tl_pt_mcq_shuffled_no_na| 2                | PT-BR, PT-PT         | Yes            |  0                           |
| amalia_dsl_tl_2_labels_zeroshot    | 2                | PT-BR, PT-PT         | No             |  0                           |
| amalia_dsl_tl_2_labels_fewshot     | 2                | PT-BR, PT-PT         | NO             |  4                           |
| amalia_dsl_tl_3_labels_zeroshot    | 3                | PT-BR, PT-PT, PT     | No             |  0                           |
| amalia_dsl_tl_3_labels_fewshot     | 3                | PT-BR, PT-PT, PT     | NO             |  6                           |


Each dataset instance consists of:
-  **text**: A sentence in a variant of Portuguese.
-  **label**: The label for the variant 0 - PT-PT, 1 - PT-BR, 2 - N/A


### Fewshot Examples

We used the `train` split from the [original dataset](https://github.com/LanguageTechnologyLab/DSL-TL/tree/main) as fewshot examples. We paired instances with different labels and saved the resulting dataset in a [new location](https://huggingface.co/datasets/amalia-llm/dsl_tl_fewshot). Each row in this new dataset consists of:
-  **examples**: A tuple of text-labels pairs containing exactly one text per label.

For the `2-shot` scenario, each instance looks like this:
```
Texto: {{example_text_1}}
Resposta: PT-PT  

Texto: {{example_text_2}} 
Resposta: PT-BR
```

For the `3-shot` scenario, each instance looks like this:
```
Texto: {{example_text_1}}
Resposta: PT-PT  

Texto: {{example_text_2}} 
Resposta: PT-BR

Texto: {{example_text_3}}
Resposta: PT
```



### Example Prompts

For ```amalia_dsl_tl_pt_mcq```, ```amalia_dsl_tl_pt_mcq_shuffled```, ```amalia_dsl_tl_pt_mcq_shuffled_no_na```:
```
Qual é a variante de português utilizada neste texto: {{text}} 
PT-PT para a variante de português de Portugal e PT-BR para a variante de português do Brasil.
Escolha a opção correta:
{{choices_str}}
Resposta: "
```

For ```amalia_dsl_tl_3_labels_fewshot``` (similar structure for ```amalia_dsl_tl_2_labels_zeroshot```, ```amalia_dsl_tl_2_labels_fewshot```, ```amalia_dsl_tl_3_labels_zeroshot```): 
```
Identifica a variante de português a que cada texto pertence. As opções são: 'PT-PT' para Português Europeu, 'PT-BR' para Português do Brasil, apenas 'PT' se não for possível distinguir a variante do português.

Texto: {{example_text_1}}
Resposta: PT-PT  

Texto: {{example_text_2}} 
Resposta: PT-BR

Texto: {{example_text_3}}
Resposta: PT 

Texto: {{example_text_4}}
Resposta: PT-PT  

Texto: {{example_text_5}} 
Resposta: PT-BR

Texto: {{example_text_6}}
Resposta: PT 

Texto: {{test_text}} 
Resposta: 
```



### Target Outputs

For ```amalia_dsl_tl_pt_mcq```, ```amalia_dsl_tl_pt_mcq_shuffled```, ```amalia_dsl_tl_pt_mcq_shuffled_no_na```: the model should return `A`, `B` (or `C` when number of labels is 3), corresponding to the correct variant of portuguese in use.

For ```amalia_dsl_tl_2_labels_zeroshot```, ```amalia_dsl_tl_2_labels_fewshot```, ```amalia_dsl_tl_3_labels_zeroshot```, and ```amalia_dsl_tl_3_labels_fewshot```: we consider the length-normalized log-probabilities of the answer options and choose the most likely.



## Processing Logic

For ```amalia_dsl_tl_pt_mcq```, ```amalia_dsl_tl_pt_mcq_shuffled```, ```amalia_dsl_tl_pt_mcq_shuffled_no_na```: the answer choices are shuffled for each instance using the `process_dsl_tl_docs` function from `utils.py`, ensuring that the order does not remain static across the dataset.



## Evaluation Metrics

The task is evaluated based on:
-   **Accuracy (`acc`)**: Measures the proportion of correct answers.
-   **Normalized Accuracy (`acc_norm`)**: Measures the proportion of correct answers. For each question, the ``correct'' answer is the option with the highest length-normalized log-probability.


<!-- 
normalized by byte-length of the target string. 
-   **Normalized Accuracy (`acc_norm`)**: Measures the proportion of correct answers. For each question, the ``correct'' answer is the option with the highest length-normalized log-probability.
-->


