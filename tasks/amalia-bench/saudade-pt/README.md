# Saudade PT


Saudade PT is a benchmark for evaluating temporal reasoning capabilities of Portuguese language models. The benchmark consists of questions designed to test models' understanding of temporal relationships between events, particularly focusing on Portuguese historical events, personalities, and cultural contexts. Each question asks the model to determine which of two events related to a specific entity occurred first.

https://huggingface.co/datasets/amalia-llm/saudade-pt


#### Categories

The benchmark covers diverse categories of Portuguese events and entities:

* **Biography (Biografia)**: Temporal events in individuals' lives
* **Politics (Politico)**: Political events, careers, and historical milestones
* **Football (Futebol)**: Football-related events, clubs, and sporting achievements
* **Television (Televisao)**: Television programs, broadcasts, and media events
* **Music (Musica)**: Musical events, album releases, and artist careers
* **Actors (Ator)**: Acting careers, film releases, and theatrical events
* **Nobility (Nobre)**: Historical nobility events and lineages
* **Historical Governments (Governo_historico)**: Government formations and political regimes
* **Business (Empresa)**: Company foundations, mergers, and business events
* **Religion (Religioso)**: Religious events and ecclesiastical history
* **Military (Militar)**: Military events and conflicts
* **Film (Filme)**: Film releases and cinema events
* **Science (Cientista)**: Scientific achievements and discoveries
* **Writers (Escritor)**: Literary events and publications
* **Elections (Eleição)**: Electoral events and democratic processes
* **Political Parties (Partido)**: Party foundations and political movements
* **Wars (Guerra)**: Military conflicts and wars
* **Extinct States (Estado_extinto)**: Historical states and their events
* **Plastic Artists (Artista_plástico)**: Visual arts events and exhibitions
* **Architects (Arquiteto)**: Architectural achievements and building constructions


#### Tasks

* `saudade_chat_pt`: Chat-formatted variant
* `saudade_chat_cot_pt`: Chat-formatted with chain-of-thought reasoning
* `saudade_pt`: Multiple-choice evaluation task for base models.

All tasks follow a binary choice format where models must determine which of two events occurred first.

### Dataset preparation

The three tasks load the `pt_binary` configuration at dataset revision
`ef67be3c1400acb9526627ad8c2a825c77ffbf37`. The `process_docs` hook prepares the
raw annotations during normal harness loading. No offline preparation is required.

The pinned file contains 7,444 records. Preparation retains the same 7,435 records,
in the same order, with the same prompts and labels as `process_temp_reasoning.py`.
The hook logs each canonical exclusion and reference correction, then reports:

```text
SAUDADE prepared 7435 of 7444 records; excluded 9 canonical prefix failures
```

The nine exclusions fail the original script's question-prefix extraction rules;
they do have answer labels. They remain excluded to preserve the canonical subset.
The two spelling discrepancies are resolved by explicit reference mappings rather
than fuzzy matching. The harness can invoke preparation repeatedly, so the same
accounting lines can appear more than once in a run. `binary_track_manifest.json`
records the exact question,
reference and options for each decision. `source_line` is one-based in the pinned
JSONL file; decisions match content, so reordering does not change the labels.

| Source line | Entity | Preparation decision |
|---:|---|---|
| 131 | António José de Melo Breyner Teles da Silva | Exclude: no prefix |
| 3216 | Canal Q | Exclude: no prefix |
| 3316 | Buéréré | Exclude: no prefix |
| 3317 | Buéréré | Exclude: no prefix |
| 3479 | Cante se Puder | Map the known reference spelling discrepancy to option B |
| 3534 | Taça da Liga de Futsal | Exclude: no prefix |
| 3840 | Revolução Iraniana | Exclude: no prefix |
| 4972 | Noel Rosa | Exclude: no prefix |
| 5299 | Inês de Medeiros | Map the known reference spelling discrepancy to option A |
| 5375 | António Campos | Exclude: no prefix |
| 6220 | Sporting Clube da Covilhã | Exclude: no prefix |

Unknown prefix failures, unmatched or ambiguous reference answers, and conflicting
prepared fields raise an error before inference. Already-prepared input is validated
against the raw annotations. Repeated preparation preserves the records and labels.

### Run the tasks

From the repository root, with the project dependencies installed, run:

```bash
python -m lm_eval --model hf --model_args pretrained=<model> \
  --include_path tasks/amalia-bench \
  --tasks saudade_pt,saudade_chat_pt,saudade_chat_cot_pt \
  --log_samples --output_path outputs/saudade
```

For a CPU pipeline smoke test, use the command below. This tests loading, requests,
inference and scoring; the two-item results are not benchmark estimates.

```bash
python -m lm_eval --model hf \
  --model_args pretrained=sshleifer/tiny-gpt2,revision=5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be,dtype=float32 \
  --include_path tasks/amalia-bench \
  --tasks saudade_pt,saudade_chat_pt,saudade_chat_cot_pt \
  --device cpu --batch_size 2 --limit 2 \
  --gen_kwargs max_gen_toks=8,do_sample=False \
  --log_samples --output_path outputs/saudade-smoke
```

Run the regression suite without downloading a model or dataset:

```bash
python -m unittest discover -s tests
```

### Dataset Structure

Raw Portuguese examples contain `question`, `entity`, `facts_pair`,
`expected_answers` and `category`. Preparation adds `prompt` and `answer_index`.
Each prepared example contains:
- `question`: Full question text in Portuguese asking which event occurred first
- `prompt`: Question stem (format: "Qual dos dois eventos relacionados com {entity} ocorreu primeiro:")
- `entity`: The entity (person, organization, event) that both facts relate to
- `facts_pair`: Array containing two event descriptions
- `answer_index`: Index (0 or 1) indicating which event in `facts_pair` occurred first
- `expected_answers`: Array with the full text of the correct answer
- `category`: Category label indicating the domain (e.g., "Biografia", "Futebol", "Politico")

Example:
```json
{
  "question": "Qual dos dois eventos relacionados com António Nicolau d'Almeida ocorreu primeiro: a fundação do Foot-Ball Club do Porto por António Nicolau d'Almeida ou o nascimento de António Nicolau d'Almeida?",
  "prompt": "Qual dos dois eventos relacionados com António Nicolau dAlmeida ocorreu primeiro:",
  "entity": "António Nicolau d'Almeida",
  "facts_pair": [
    "Fundação do Foot-Ball Club do Porto por António Nicolau d'Almeida",
    "Nascimento de António Nicolau d'Almeida"
  ],
  "answer_index": 1,
  "expected_answers": ["Nascimento de António Nicolau d'Almeida"],
  "category": "Biografia"
}
```

### Evaluation Metrics

#### Chat Tasks (chat and chat_cot)
The benchmark uses custom answer extraction (`ExtractAnswerR`) that attempts to parse the model's response using multiple patterns:
1. LaTeX-style boxed answers: `\boxed{A}`
2. Explicit answer statements: "A resposta é A", "The answer is A"
3. Final letter at end of response
4. Any A-D letter found in the response


#### Multiple Choice Tasks (mcq)
* `acc`: Accuracy (exact match)
* `acc_norm`: Normalized accuracy (probability-based)