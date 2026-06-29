# ALBA MCQ: Automated Linguistics Benchmark for baseline Assessment

[![Paper](https://img.shields.io/badge/Paper-ACL_Anthology-blue)](https://aclanthology.org/2026.propor-1.69/)
[![Original Repo](https://img.shields.io/badge/Main_Repo-AMALIA--LLM-green)](https://github.com/AMALIA-LLM/alba-benchmark)

**ALBA MCQ** is the multiple-choice adaptation of the ALBA benchmark, a suite of linguistically informed tasks designed to assess the capabilities of Large Language Models (LLMs) in **European Portuguese**.

While the original ALBA benchmark focuses on generative outputs, this MCQ version is designed for **quicker evaluation** of linguistic concepts by providing predefined choices.

> The official ALBA benchmark is primarily intended for a **generative format**, where models generate answers from scratch. These MCQ versions are intended as a baseline or for environments requiring rapid, automated scoring. For the generative setup, refer to the [official repository](https://github.com/AMALIA-LLM/alba-benchmark).

---

## 🧩 Dataset Overview

ALBA includes multiple subtasks, each targeting a distinct linguistic domain in European Portuguese:

| Subset | Domain | Focus Area |
| :--- | :--- | :--- |
| **`alba_semantics_mcq`** | Semantics | Meaning, inference, and context understanding |
| **`alba_discourse_mcq`** | Discourse | Coherence, cohesion, and pragmatic relations |
| **`alba_variety_mcq`** | Variety | Regional and dialectal variation |
| **`alba_lexicology_mcq`** | Lexicology | Vocabulary and word usage |
| **`alba_morphology_mcq`** | Morphology | Word formation and inflection |
| **`alba_phonetics_mcq`** | Phonetics | Sound-based linguistic reasoning |
| **`alba_syntax_mcq`** | Syntax | Grammatical structure and composition |
| **`alba_word_play_mcq`** | Word Play | Idioms, metaphors, and creative language |

---

## 📊 Data Format

Each entry in the dataset follows the structure below:

```json
{
    "id": 1,
    "subject": "Culture-bound Semantics",
    "question": "Dá-me exemplos de provérbios onde conste a palavra \"carro\".",
    "choices": [
        "Opção A (Pontuação: 1)...",
        "Opção B (Pontuação: 5)...",
        "Opção C (Pontuação: 3)..."
    ],
    "scores": [1, 5, 3],
    "correct_choice": 1
}
```

### Fields
* **`id`**: Unique identifier for the question.
* **`subject`**: Specific linguistic category or theme.
* **`question`**: The prompt/question presented to the model.
* **`choices`**: List of possible answers.
* **`scores`**: Expert or annotator ratings (1–5) for each respective choice.
* **`correct_choice`**: Index of the highest-scoring answer (the ground truth).

---

## ⚙️ Task Variants

There are two versions of the MCQ task available:

* **`alba_mcq`**: The standard version featuring **three answer options** per question.
* **`alba_two_options_mcq`**: A simplified version featuring **only two options** (the best-rated vs. the worst-rated).

---

## 📖 Citation

If you use this dataset in your research, please cite the following paper:

```bibtex
@inproceedings{vieira-etal-2026-alba,
    title = "{ALBA}: A {E}uropean {P}ortuguese Benchmark for Evaluating Language and Linguistic Dimensions in Generative {LLM}s",
    author = "Vieira, In{\^e}s  and      Calvo, In{\^e}s  and      Paulo, Iago  and      Furtado, James  and      Ferreira, Rafael  and      Tavares, Diogo  and      Gl{\'o}ria-Silva, Diogo  and      Semedo, David  and      Magalh{\~a}es, Jo{\~a}o",
    booktitle = "Proceedings of the 17th International Conference on Computational Processing of {P}ortuguese ({PROPOR} 2026) - Vol. 1",
    month = apr,
    year = "2026",
    address = "Salvador, Brazil",
    publisher = "Association for Computational Linguistics",
    url = "[https://aclanthology.org/2026.propor-1.69/](https://aclanthology.org/2026.propor-1.69/)",
    pages = "697--707",
    ISBN = "979-8-89176-387-6"
}
```