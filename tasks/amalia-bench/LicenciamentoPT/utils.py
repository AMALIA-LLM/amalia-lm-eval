"""LicenciamentoPT: prompts and scoring.

Each item is a question about Portuguese urban licensing. With five passages (the answer and four close ones), the multiple-choice
task asks for the letter and the citation task for the passage's text, word for word. Without passages, the law-in-context task gives
the whole part of the law the answer sits in (15k to 22k tokens, 247 to 510 passages) and the closed-book task gives nothing; both
ask for the provision's reference and its text, word for word.
"""
import functools
import json
import os
import re

CONTEXT_PREFIX = "Contexto: "
TEXT_PREFIX = "Texto: "


def _candidates(doc):
    blocks = []
    for letter, text, context, passage in zip(doc["choices"]["label"], doc["choices"]["text"], doc["contexts"], doc["passages"]):
        lines = [f"{letter}. {passage['reference']}, {passage['act']}"]
        if context:
            lines.append(f"   {CONTEXT_PREFIX}{context}")
        lines.append(f"   {TEXT_PREFIX}{text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def doc_to_text_mcq(doc):
    return (
        "Dada a seguinte pergunta sobre licenciamento urbanístico e cinco passagens da lei (A, B, C, D e E), escolha a passagem "
        "que responde à pergunta.\n"
        f"Pergunta: {doc['question'].strip()}\n\n"
        f"{_candidates(doc)}\n\n"
        'A sua resposta deve terminar com "A melhor resposta é [letra_da_resposta]" onde [letra_da_resposta] é A, B, C, D ou E.'
    )


def doc_to_text_cite(doc):
    return (
        "Dada a seguinte pergunta sobre licenciamento urbanístico e cinco passagens da lei, copie palavra por palavra o texto da "
        "passagem que responde à pergunta.\n"
        f"Pergunta: {doc['question'].strip()}\n\n"
        f"{_candidates(doc)}\n\n"
        f'Responda apenas com o texto da passagem, tal como aparece depois de "{TEXT_PREFIX.strip()}", sem a letra, sem a '
        "referência e sem o contexto."
    )


def _normalize(text):
    """Wrapping only: markdown emphasis, surrounding quotes, a leading «Texto:», the clause's own label («6 - », «a) », «iv) »),
    punctuation closing the clause («;», «.», «:», «,»), letter case and runs of spaces. The words inside are compared as they are."""
    text = re.sub(r"\*\*|__|`", "", text).strip()
    text = re.sub(r"^\s*texto\s*:\s*", "", text, flags=re.IGNORECASE)
    for _ in range(2):
        text = text.strip().strip("\"'«»“”").strip().rstrip(".;:,").strip()
    text = re.sub(r"^(\d+\s*-\s*|[a-z]{1,2}\)\s*|[ivxl]+\)\s*)", "", text)
    return " ".join(text.split()).casefold()


def _tokens(text):
    return re.findall(r"\w+", text.casefold())


class _Source:
    """A text a quote should come from (the five passages, or the whole law): its words in order, and its 5-word sequences."""

    def __init__(self, texts):
        words = []
        for t in texts:
            words += _tokens(_normalize(t))
            words.append("§")   # no stretch runs from one passage into the next unless the quote runs on
        self.stream = " " + " ".join(w for w in words if w != "§") + " "
        self.grams = {tuple(words[k:k + 5]) for k in range(len(words) - 4) if "§" not in words[k:k + 5]}
        runs = [w for w in words if w != "§"]
        self.grams |= {tuple(runs[k:k + 5]) for k in range(len(runs) - 4)}

    def has(self, words):
        return f" {' '.join(words)} " in self.stream

    def uncovered(self, words):
        """How many of the quote's words belong to no 5-word sequence of the source."""
        if len(words) < 5:
            return 0 if self.has(words) else len(words)
        covered = [False] * len(words)
        for k in range(len(words) - 4):
            if tuple(words[k:k + 5]) in self.grams:
                for m in range(k, k + 5):
                    covered[m] = True
        return covered.count(False)


ELLIPSIS = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]|\(\s*(?:\.\.\.|…)\s*\)|…")
MIN_QUOTE_WORDS = 3   # shorter answers («Sim.», «Não sei.») are not quotes
ADDED_WORDS = 4


def _quote_scores(quote, gold_texts, source):
    """exact_quote: the quote has the words of one of the gold texts, in order, and no others (wrapping, punctuation and case
    aside). not_verbatim: it is a quote and some piece of it is not a stretch of the source (a word changed, dropped or added).
    added_words: it is a quote and at least four of its words belong to no 5-word sequence of the source, i.e. text the source
    does not have, as opposed to a slip. A quote cut with «[...]» is judged piece by piece."""
    words = _tokens(_normalize(quote))
    exact = any(words == _tokens(_normalize(t)) for t in gold_texts)
    if len(words) < MIN_QUOTE_WORDS:
        return {"exact_quote": float(exact), "not_verbatim": 0.0, "added_words": 0.0}
    pieces = [w for w in (_tokens(p) for p in ELLIPSIS.split(_normalize(quote))) if len(w) >= MIN_QUOTE_WORDS] or [words]
    return {
        "exact_quote": float(exact),
        "not_verbatim": float(any(not source.has(p) for p in pieces)),
        "added_words": float(sum(source.uncovered(p) for p in pieces) >= ADDED_WORDS),
    }


def process_results_cite(doc, results):
    """Scored against the five passages shown (see _quote_scores)."""
    return _quote_scores(results[0] if results else "", [doc["answer_text"]], _Source(doc["choices"]["text"]))


# Law-in-context and closed-book tasks.

LAW_DATASET = "Paulo-FOAAP/licenciamento-pt"
LAW_REVISION = "ef2162977464f104dd031550dfd380f7ff7562e9"   # same revision as the task configs
ACTS = "o Regime Jurídico da Urbanização e da Edificação (Decreto-Lei n.º 555/99, na redação do Decreto-Lei n.º 108/2026) e a Portaria n.º 320/2026/1"
ANSWER_FORMAT = (
    "Responda exatamente neste formato, em duas linhas:\n"
    "Referência: <o artigo, número e alínea, ou o anexo, número e alínea, da disposição que responde>\n"
    f"{TEXT_PREFIX}<o texto dessa disposição, palavra por palavra>"
)


@functools.lru_cache(maxsize=None)
def _law():
    """Every passage of the law in publication order (the dataset's «law» configuration). LICENCIAMENTO_PT_LAW points at a local
    copy of law.jsonl."""
    local = os.environ.get("LICENCIAMENTO_PT_LAW")
    if local:
        with open(local, encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    import datasets

    return list(datasets.load_dataset(LAW_DATASET, "law", split="law", revision=LAW_REVISION))


def _header(path):
    m = re.match(r"anexo-(V|VI|VII)\.", path)
    if m:
        return f"ANEXO {m.group(1)}"
    m = re.match(r"anexo\.art(\d+)(?:-([A-Z]))?\.", path)
    return f"Artigo {m.group(1)}.º" + (f"-{m.group(2)}" if m.group(2) else "") if m else None


@functools.lru_cache(maxsize=None)
def _section_text(section):
    rows = [p for p in _law() if p["section"] == section]
    lines, last = [rows[0]["act"]], None
    for p in rows:
        head = _header(p["path"])
        if head and head != last:
            lines += ["", head]
            last = head
        lines.append(p["text"])
    return "\n".join(lines)


def doc_to_text_law(doc):
    return (
        f"{_section_text(doc['section'])}\n\n"
        "Acima está o texto em vigor de uma parte da lei do licenciamento urbanístico. Indique a disposição que responde à pergunta "
        "e copie o seu texto palavra por palavra.\n"
        f"Pergunta: {doc['question'].strip()}\n\n"
        f"{ANSWER_FORMAT}"
    )


def doc_to_text_closed(doc):
    return (
        f"Sobre o licenciamento urbanístico em Portugal, segundo {ACTS}, indique a disposição da lei que responde à pergunta e "
        "copie o seu texto palavra por palavra.\n"
        f"Pergunta: {doc['question'].strip()}\n\n"
        f"{ANSWER_FORMAT}"
    )


def _split_answer(text):
    """(reference, quote) from «Referência: …» and «Texto: …». Without a «Texto:» line the whole answer, minus a reference
    line, is read as the quote."""
    ref = re.search(r"refer[êe]ncia\s*:\s*(.*)", text, flags=re.IGNORECASE)
    quote = re.search(r"texto\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    reference = ref.group(1).strip() if ref else ""
    if quote:
        body = quote.group(1)
    else:
        body = re.sub(r"^.*refer[êe]ncia\s*:.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    body = re.split(r"\n\s*\n|\n\s*refer[êe]ncia\s*:|\n\s*\(?\s*nota\b", body.strip(), flags=re.IGNORECASE)[0]
    body = re.sub(r"\s*\((?:nota|obs)[^)]*\)\s*$", "", body, flags=re.IGNORECASE)
    return reference, body


def _parts_of_path(path):
    """anexo-V.27.i.iii → {annex: V, number: 27, letter: i, sub: iii}; anexo.art6-A.1.e → {article: 6-A, number: 1, letter: e}."""
    m = re.match(r"anexo-(V|VI|VII)\.(\d+)(?:\.([a-z]{1,2}))?(?:\.([ivxl]+))?$", path)
    keys = ("annex", "number", "letter", "sub")
    if not m:
        m = re.match(r"anexo\.art(\d+(?:-[A-Z])?)\.(\d+)(?:\.([a-z]{1,2}))?(?:\.([ivxl]+))?$", path)
        keys = ("article", "number", "letter", "sub")
    return {k: v for k, v in zip(keys, m.groups()) if v} if m else None


ACT = re.compile(r"\b(?:decreto-lei|dl|portaria|lei)\s*(?:n\.?\s*º\s*)?\d+(?:\s*/\s*\d+)*", re.IGNORECASE)


def _parts_of_reference(text):
    """The provision a free-text reference names: «artigo 9.º, n.º 12, alínea i)», «Anexo V, n.º 27, alínea i), subalínea iii)»,
    «art. 6.º-A, n.º 1, al. e)», «ANEXO V, II, 7» (a bare number after the annex or its section). Act numbers («Decreto-Lei
    n.º 555/99», «Portaria n.º 320/2026/1») are ignored. An alínea given as a digit is read as the number when no number is named,
    and is wrong otherwise. Only the first of each part is read."""
    t = ACT.sub(" ", text.replace("°", "º"))
    out = {}
    m = re.search(r"\banexo\s+(VII|VI|V)\b", t, flags=re.IGNORECASE)
    if m:
        out["annex"] = m.group(1).upper()
    m = re.search(r"\bart(?:igo|\.)?\s*(\d+)\s*\.?\s*º?\s*(?:-\s*([A-Z])\b)?", t, flags=re.IGNORECASE)
    if m:
        out["article"] = m.group(1) + (f"-{m.group(2).upper()}" if m.group(2) else "")
    m = re.search(r"\b(?:n\.?\s*º|número|ponto)\s*(\d+)", t, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\banexo\s+(?:VII|VI|V)\b[\s,]*(?:(?:sec[çc][ãa]o\s+)?[IVX]+\b[\s,.-]*)?(\d+)\b", t, flags=re.IGNORECASE)
    if m:
        out["number"] = m.group(1)
    m = re.search(r"\bsubal[íi]nea\s*([ivxl]+)\b", t, flags=re.IGNORECASE)
    if m:
        out["sub"] = m.group(1).lower()
    m = re.search(r"(?<!sub)\bal(?:[íi]nea|\.)\s*([a-z]{1,2}|\d+)\b", t, flags=re.IGNORECASE)
    if m:
        if not m.group(1).isdigit():
            out["letter"] = m.group(1).lower()
        elif "number" not in out:
            out["number"] = m.group(1)
        else:
            out["letter"] = "?"
    return out


@functools.lru_cache(maxsize=None)
def _law_source():
    return _Source(tuple(p["text"] for p in _law()))


def process_results_recall(doc, results):
    """Another provision that answers the same question (also_answers) counts as right.
    right_reference: the reference names the answer's provision, all its parts and no other (for the annexes: annex, number,
    alínea, subalínea; for the RJUE: article, number, alínea, subalínea). The quote is scored against the whole law (see
    _quote_scores)."""
    reference, quote = _split_answer(results[0] if results else "")
    answers = [(doc["answer_path"], doc["answer_text"])] + [(a["path"], a["text"]) for a in doc.get("also_answers") or []]
    named = _parts_of_reference(reference)

    def names(path):
        """Every part of the provision, and no part it does not have (a sibling or an invented alínea is wrong)."""
        gold, seen = _parts_of_path(path), dict(named)
        if not gold:
            return False
        if "annex" in gold:
            seen.pop("article", None)   # «artigo 2.º da Portaria, Anexo V, …»
        return seen == gold

    scores = _quote_scores(quote, [t for _, t in answers], _law_source())
    return {"right_reference": float(any(names(p) for p, _ in answers)), **scores}
