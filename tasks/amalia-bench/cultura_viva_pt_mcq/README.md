# CulturaVivaPT MCQ

CulturaVivaPT is a culturally-grounded evaluation benchmark for Portuguese language models, covering ten key domains of Portuguese cultural knowledge. This subset of the benchmark consists of multiple-choice questions designed to test models' understanding of Portuguese culture, traditions, and local knowledge. The benchmark emphasizes culturally-specific elements including audiovisual culture, traditional festivals, traditional gastronomy, regional geography, cultural heritage, public holidays, Portuguese literature, notable personalities, proverbs, and sports culture that reflect Portuguese linguistic heritage.

https://huggingface.co/datasets/amalia-llm/cultura-viva-pt-mcq


### Groups, Tags, and Tasks

#### Groups

* `cultura_viva_pt`: Evaluation group covering various cultural domains
* `cultura_viva_pt_chat`: Chat-formatted variant of the `cultura_viva_pt` group for instruction-tuned models
* `cultura_viva_pt_chat_cot`: Chain-of-thought prompting variant for the `cultura_viva_pt_chat` group, encouraging models to generate reasoning steps before answering

#### Domains

The benchmark covers ten distinct cultural domains:

* **Audiovisual**: Portuguese cinema, television, music, and multimedia cultural productions
* **Festivals**: Traditional celebrations, local festivals, religious festivities, and cultural events across Portugal
* **Gastronomy**: Traditional Portuguese foods, regional specialties, culinary heritage
* **Geography**: Portuguese administrative divisions, regional characteristics, historical geography, and local landmarks
* **Heritage**: Cultural heritage, historical monuments, UNESCO sites, and Portuguese architectural traditions
* **Holidays**: Public holidays, municipal commemorations, patron saints, and liturgical calendar events
* **Literature**: Portuguese literary works, authors, movements, and linguistic heritage
* **Personalities**: Notable Portuguese historical figures, artists, writers, scientists, and cultural icons
* **Proverbs**: Traditional Portuguese sayings, folk wisdom, and linguistic expressions
* **Sports**: Portuguese sports culture, notable athletes, sporting achievements, and traditional sports

#### Tasks

* `cultura_viva_pt`: Main multiple-choice evaluation task (0-shot)
* `cultura_viva_pt_audiovisual`: Audiovisual domain subset
* `cultura_viva_pt_festivals`: Festivals domain subset
* `cultura_viva_pt_gastronomy`: Gastronomy domain subset
* `cultura_viva_pt_geography`: Geography domain subset
* `cultura_viva_pt_heritage`: Heritage domain subset
* `cultura_viva_pt_holidays`: Holidays domain subset
* `cultura_viva_pt_literature`: Literature domain subset
* `cultura_viva_pt_personalities`: Personalities domain subset
* `cultura_viva_pt_proverbs`: Proverbs domain subset
* `cultura_viva_pt_sports`: Sports domain subset

Variants available for tasks:
- `default`: Completion-based multiple choice
- `chat`: Chat-formatted multiple choice for instruction-tuned models
- `cot`: Chain-of-thought prompting variant for the chat format, encouraging models to generate reasoning steps before answering

### Dataset Structure

Each example contains:
- `id`: Unique identifier (format: `ptculcov_{domain_abbr}_{number}`)
- `question`: The question text in Portuguese
- `choices`: Object containing:
  - `text`: Array of four answer options
  - `label`: Array of labels ["A", "B", "C", "D"]
- `answerKey`: The correct answer label ("A", "B", "C", or "D")
- `domain`: One of ["audiovisual", "festivals", "gastronomy", "geography", "heritage", "holidays", "literature", "personalities", "proverbs", "sports"]
- `_seed_id`: Original source identifier
- `_reviewed_at`: Timestamp of review

### Evaluation Metrics

#### Multiple Choice Tasks (default and chat)
* `acc`: Accuracy (exact match)
* `acc_norm`: Normalized accuracy (probability-based)
