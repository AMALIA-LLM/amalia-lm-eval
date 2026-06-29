# Portuguese Text Completion (Multiple Choice) Task

## Task Overview
This task evaluates a language model’s preference when completing a sentence in Portuguese. Each example is formatted as a multiple-choice question with two possible completions: one in European Portuguese (pt-PT) and the other in Brazilian Portuguese (pt-BR).

Given that both variants are valid, the task is intended to capture the model’s tendency or bias toward one variant over the other, rather than to assess correctness in an absolute sense. The results therefore reflect which variety of Portuguese the model systematically favors.

## Dataset
- **Name** : `pt_text_completion`
- **Dataset Source**: [amalia-llm/pt_text_completion](https://huggingface.co/datasets/amalia-llm/pt_text_completion)
- **Description**: This dataset contains sentences in Portuguese with a blank space where a word should be inserted. Each sentence has two possible completions: one in European Portuguese and the other in Brazilian Portuguese. A higher value indicates a higher tendency towards pt-pt.
- **Validation Split**: `valid`
- **Test Split**: `test`

Each dataset instance consists of:
- **id**: A unique identifier for the instance.
- **text**: A sentence with a blank space where a word should be inserted.
- **pt-pt**: The European Portuguese completion option.
- **pt-br**: The Brazilian Portuguese completion option.

## Task Definition
For each instance, the model is given a sentence and must choose the most appropriate completion from the given choices.

### Example Prompt
```
Qual a palavra mais adequada para completar o espaço em branco na seguinte frase: {text}
A opção correta é:
A. {choice_1}
B. {choice_2}
Resposta:
```

### Target Output
The model should return `A` or `B`, corresponding to the correct choice which in this case is the European Portuguese completion.

## Processing Logic
The answer choices are shuffled for each instance using the `process_pt_text_completion_docs` function from `utils.py`, ensuring that the order does not remain static across the dataset.

## Evaluation Metrics
The task is evaluated based on:
- **Accuracy (`acc`)**: Measures the proportion of pt-pt answers.
