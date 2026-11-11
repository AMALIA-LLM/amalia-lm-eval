import json
import os
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks/amalia-bench/LicenciamentoPT"
FIXTURE = REPO / "tests/fixtures/licenciamento_pt.jsonl"
LAW = REPO / "tests/fixtures/licenciamento_pt_law.jsonl"  # arts. 34.º and 35.º of the RJUE, from the dataset's «law» configuration

os.environ["LICENCIAMENTO_PT_LAW"] = str(LAW)

from lm_eval.api.task import ConfigurableTask  # noqa: E402
from lm_eval.utils import load_yaml_config  # noqa: E402


def records():
    with open(FIXTURE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class LicenciamentoPTTaskTest(unittest.TestCase):
    def test_records_load_in_all_tasks(self):
        for filename, target in [
            ("licenciamento_pt_mcq_chat.yaml", ["C", "A", "B"]),
            ("licenciamento_pt_cite_chat.yaml", None),
            ("licenciamento_pt_law_chat.yaml", None),
            ("licenciamento_pt_closedbook_chat.yaml", None),
        ]:
            with self.subTest(task=filename):
                config = load_yaml_config(str(TASKS / filename))
                self.assertEqual(config["dataset_path"], "Paulo-FOAAP/licenciamento-pt")
                self.assertEqual(config["dataset_kwargs"]["revision"], "ef2162977464f104dd031550dfd380f7ff7562e9")
                config.update(dataset_path="json", dataset_name=None, dataset_kwargs={"data_files": {"test": str(FIXTURE)}})
                task = ConfigurableTask(config=config)
                docs = task.eval_docs
                self.assertEqual(len(docs), 3)
                if target:
                    self.assertEqual([task.doc_to_target(d) for d in docs], target)
                prompt = task.doc_to_text(docs[0])
                self.assertIn("O que é uma comunicação prévia?", prompt)
                if filename == "licenciamento_pt_law_chat.yaml":
                    self.assertIn("Artigo 34.º", prompt)
                    self.assertIn(docs[0]["answer_text"], prompt)


class LicenciamentoPTScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        sys.path.insert(0, str(TASKS))
        import utils

        cls.utils = utils
        cls.doc = records()[0]

    def test_cite_ignores_wrapping_not_words(self):
        u, doc = self.utils, self.doc
        text = doc["answer_text"]
        self.assertEqual(u.process_results_cite(doc, [f'Texto: «{text.rstrip(";.")}»'])["exact_quote"], 1.0)
        self.assertEqual(u.process_results_cite(doc, [text.replace("comunicação", "declaração", 1)])["exact_quote"], 0.0)
        self.assertEqual(u.process_results_cite(doc, ["Não sei."]), {"exact_quote": 0.0, "not_verbatim": 0.0, "added_words": 0.0})

    def test_recall_reference_and_quote(self):
        u, doc = self.utils, self.doc
        right = u.process_results_recall(doc, [f"Referência: artigo 34.º, n.º 2\nTexto: {doc['answer_text']}"])
        self.assertEqual(right, {"right_reference": 1.0, "exact_quote": 1.0, "not_verbatim": 0.0, "added_words": 0.0})
        act_first = u.process_results_recall(doc, ["Referência: Decreto-Lei n.º 555/99, artigo 34.º, n.º 2\nTexto: x"])
        self.assertEqual(act_first["right_reference"], 1.0)
        extra = u.process_results_recall(doc, ["Referência: artigo 34.º, n.º 2, alínea a)\nTexto: x"])
        self.assertEqual(extra["right_reference"], 0.0)
        slip = u.process_results_recall(doc, [f"Referência: artigo 34.º, n.º 2\nTexto: {doc['answer_text'].replace('declaração', 'declarações')}"])
        self.assertEqual((slip["not_verbatim"], slip["added_words"]), (1.0, 0.0))
        added = u.process_results_recall(doc, [f"Referência: artigo 34.º, n.º 2\nTexto: {doc['answer_text']} e nos termos da lei geral"])
        self.assertEqual((added["not_verbatim"], added["added_words"]), (1.0, 1.0))
        cut = doc["answer_text"].split(", ")
        abridged = u.process_results_recall(doc, [f"Referência: artigo 34.º, n.º 2\nTexto: {cut[0]} [...] {cut[-1]}"])
        self.assertEqual((abridged["not_verbatim"], abridged["added_words"]), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
