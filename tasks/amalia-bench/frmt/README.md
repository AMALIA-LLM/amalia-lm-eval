# FRMT

FRMT is a machine translation benchmark for evaluating English-to-Portuguese translation quality, with separate variants for European Portuguese (PT) and Brazilian Portuguese (BR).

## Dataset

**Source:** `hugosousa/frmt` on HuggingFace

The dataset includes test examples categorized into three difficulty buckets:
- **lexical**: Tests lexical variation and word choice
- **entity**: Tests proper handling of named entities
- **random**: Random sampling across translation phenomena

## Task Variants

### Zero-shot Tasks
- `entity_zeroshot_br` / `entity_zeroshot_pt`
- `lexical_zeroshot_br` / `lexical_zeroshot_pt`
- `random_zeroshot_br` / `random_zeroshot_pt`

### Few-shot Tasks
- `entity_fewshot_br` / `entity_fewshot_pt`
- `lexical_fewshot_br` / `lexical_fewshot_pt`
- `random_fewshot_br` / `random_fewshot_pt`

### Task Groups
- `amalia_frmt_fewshot_br` - All few-shot Brazilian Portuguese tasks
- `amalia_frmt_fewshot_pt` - All few-shot European Portuguese tasks
- `amalia_frmt_zeroshot_br` - All zero-shot Brazilian Portuguese tasks
- `amalia_frmt_zeroshot_pt` - All zero-shot European Portuguese tasks

## Evaluation

**Metric:** chrF (character n-gram F-score)
- Aggregation: chrF score
- Higher is better

**Generation stops at:**
- `</s>`
- `<|im_end|>`
- `\n\nInglês:`

## Prompt Format

**Zero-shot (Brazilian):**
```
Traduz o seguinte texto de Inglês para Português do Brasil.
Inglês: {input}

Português do Brasil:
```

**Few-shot (with exemplars):**
```
Inglês: {example_en}
Português: {example_target}

Inglês: {input}
Português:
```
