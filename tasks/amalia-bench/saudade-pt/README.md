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

### Dataset Structure

Each example contains:
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