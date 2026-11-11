# LicenciamentoPT

LicenciamentoPT checks whether a model answers a question about Portuguese urban licensing by pointing to the provision of the law
that answers it, and by quoting that provision word for word. The same 160 questions are asked at three levels of help: five
candidate passages, the whole part of the law the answer sits in, and nothing.

https://huggingface.co/datasets/Paulo-FOAAP/licenciamento-pt

## Data

Each item is a question in the words of a home owner, an architect or a municipal official ("A câmara pode exigir o plano de
segurança e saúde?", "Preciso de licença para mudar as janelas?"). Questions about the elements of a comunicação prévia name the
procedure, since Annex V lists similar elements for other procedures. The 160 questions ask about 101 provisions.

The law is the text in force published in the Diário da República, held in the dataset's `law` configuration (1,431 passages):

* Portaria n.º 320/2026/1, Annexes V to VII (the elements that instruct each procedure, their format, and the model forms)
* Regime Jurídico da Urbanização e da Edificação (Decreto-Lei n.º 555/99, as amended by Decreto-Lei n.º 108/2026): what a
  municipality may not ask for (art. 9.º, n.º 12), exempt works (arts. 6.º and 6.º-A), and the comunicação prévia procedure

For the five-passage tasks, the four wrong passages are the ones a lexical search ranks closest to the question, leaving out the
answer's own parent and sub-items, any passage with the same words, the model forms of Annex VII, and passages listed as also
answering it. That list was completed by reading the multiple-choice errors of AMALIA-9B-0626-DPO and two outside reviews, and moving
out wrong options that also answered (art. 97.º, n.º 4 for "A câmara pode pedir o livro de obra com o pedido?", for example).
Because the errors read were that model's, its `mcq` score may be slightly favoured over other models'. A word-overlap baseline
(the passage sharing most words with the question, no model) scores 58.9 % on `mcq`; chance is 20 %.

Each passage carries its reference and the sentences that introduce it, as the law reads: "i) O livro de obra digitalizado;" only
answers anything under "… nem os municípios podem determinar a apresentação de:".

## Tags and Tasks

### Tags

* `licenciamento_pt`: the three tasks that fit an 8k context (`mcq`, `cite`, `closedbook`)
* `licenciamento_pt_long`: the law-in-context task, which needs a 32k context

### Tasks

With five passages (the answer and four close ones):

* `licenciamento_pt_mcq_chat`: choose the passage that answers (A to E). Metric: `exact_match`.
* `licenciamento_pt_cite_chat`: copy the text of the passage that answers. Metrics: the quote metrics below, with the five
  passages as the source.

Without passages:

* `licenciamento_pt_law_chat`: the prompt holds the whole part of the law the answer sits in, as published: Annexes V to VII of
  the Portaria (21.9k tokens, 510 passages), or one of three runs of RJUE articles (1 to 20, 21 to 70, 71 to 130; 15.5k to 20.3k
  tokens, 247 to 362 passages). The part shown depends on the answer, so the model is told which of the four parts to read,
  not where in it. The model gives the provision's reference and its text.
* `licenciamento_pt_closedbook_chat`: the same question and answer format, with no law in the prompt.

Both are scored on `right_reference`: the reference names the provision with all its parts and no other (annex or article,
number, alínea, subalínea; act numbers such as "Decreto-Lei n.º 555/99" are ignored, and "Anexo V, II, 7" reads as n.º 7). Another
provision with the same words, or listed in `also_answers` as answering the question as well (arts. 34.º, n.º 2 and 80.º, n.º 2
on when works may start, for example), counts as right.

### Quote metrics (all quote tasks)

Quotes are compared word by word, ignoring wrapping (markdown, quotation marks, a leading "Texto:", the clause's own label),
punctuation and letter case. The source is the five passages (`cite`) or the whole law (`law`, `closedbook`).

* `exact_quote`: the words of the answer's text, in order, and no others.
* `not_verbatim` (lower is better): the quote is not a stretch of the source; a single word changed, dropped or added counts.
* `added_words` (lower is better): at least four words of the quote belong to no five-word sequence of the source, i.e. text the
  source does not have, as opposed to a slip.

Answers under three words ("Sim.", "Não sei.") are not quotes and count in neither; a quote cut with "[...]" is judged piece by
piece.

## Categories

| Category | Items | What it covers |
|---|---|---|
| `elementos_comuns` | 23 | Annex V, n.os 5 to 17 |
| `comunicacao_previa_edificacao` | 59 | Annex V, n.º 27, including each specialty project |
| `formato` | 7 | Annex VI |
| `nao_pode_ser_exigido` | 17 | RJUE art. 9.º, n.º 12 |
| `isencoes` | 26 | RJUE arts. 6.º, n.º 1 and 6.º-A, n.º 1 |
| `procedimento` | 28 | RJUE arts. 4.º, 10.º, 34.º, 35.º, 58.º, 62.º-A, 71.º, 80.º, 80.º-A, 97.º, 102.º |
| **Total** | **160** | |

## Usage

```bash
lm_eval --model vllm --model_args pretrained=amalia-llm/AMALIA-9B-0626-DPO \
  --tasks licenciamento_pt --include_path tasks/amalia-bench --apply_chat_template

lm_eval --model vllm --model_args pretrained=amalia-llm/AMALIA-9B-0626-DPO,max_model_len=32768 \
  --tasks licenciamento_pt_long --include_path tasks/amalia-bench --apply_chat_template
```

## Results

amalia-llm/AMALIA-9B-0626-DPO, zero-shot, chat template, vLLM 0.11.0, bfloat16, greedy decoding, one L40S (2026-11-11). The
standard errors below are clustered by provision (160 items, 101 provisions), so they are wider than the ones the harness prints.

| Task | What the model is given | `right_reference` | `exact_quote` | `not_verbatim` ↓ | `added_words` ↓ |
|---|---|---|---|---|---|
| `licenciamento_pt_law_chat` | the whole part of the law (15k–22k tokens) | 8.1 ± 2.3 | 24.4 ± 4.1 | 51.2 ± 4.5 | 15.6 ± 2.8 |
| `licenciamento_pt_closedbook_chat` | nothing | 0.0 | 0.0 | 100.0 | 99.4 ± 0.6 |
| `licenciamento_pt_cite_chat` | five passages | | 61.3 ± 3.8 | 15.0 ± 3.1 | 6.2 ± 1.9 |

| Task | What the model is given | `exact_match` | word-overlap baseline | chance |
|---|---|---|---|---|
| `licenciamento_pt_mcq_chat` | five passages | 87.5 ± 2.6 | 58.9 | 20.0 |

Given the whole part of the law the answer sits in, the model names the right provision in 8.1 % of answers and quotes it word for
word in 24.4 %. Half its quotes (51.2 %) are not word for word, and 15.6 % carry text the law does not have: after "seguro de
responsabilidade civil dos técnicos, nos termos da Lei n.º 31/2009, de 3 de julho" it adds "assegurando a responsabilidade por
danos decorrentes de erros ou omissões profissionais", and to Annex V, n.º 9, alínea d) it adds "nomeadamente no Plano Diretor
Municipal (PDM)". Its references miss mostly by number: elements of n.º 27 are cited as "ponto 1" or "ponto 24". With no law in the
prompt it never reproduces a provision. With the answer among five passages, it picks it in 87.5 % of items and copies it word for
word in 61.3 %: finding the provision in the law is the hard part, not choosing among a few.

## Checklist

* [x] Is the task an existing benchmark in the literature? No, it is new.
* [x] Is the dataset original Portuguese (PT-PT)? Yes: questions written for this benchmark, passages quoted from the Diário da República.
* [x] Is the scoring deterministic, with no judge model? Yes.
* [x] Is there a test? `tests/test_licenciamento_pt.py`.
