import unittest
from pathlib import Path

from lm_eval.api.task import ConfigurableTask
from lm_eval.utils import load_yaml_config

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks/amalia-bench/saudade-pt"
FIXTURE = REPO / "tests/fixtures/saudade_pt_binary.jsonl"


class SaudadeTaskTest(unittest.TestCase):
    def test_released_records_load_in_all_task_variants(self):
        for filename, targets in [
            ("saudade_mcq_pt.yaml", [0, 0, 1, 0]),
            ("saudade_chat_pt.yaml", ["A", "A", "B", "A"]),
            ("saudade_chat_cot_pt.yaml", ["A", "A", "B", "A"]),
        ]:
            with self.subTest(task=filename):
                config = load_yaml_config(str(TASKS / filename))
                config.update(
                    dataset_path="json",
                    dataset_name=None,
                    dataset_kwargs={"data_files": {"test": str(FIXTURE)}},
                )
                task = ConfigurableTask(config=config)
                docs = task.eval_docs
                self.assertEqual(len(docs), 4)
                self.assertEqual([task.doc_to_target(doc) for doc in docs], targets)
                self.assertIn("Rio Minho", task.doc_to_text(docs[0]))
                self.assertIn("Sacalina", task.doc_to_text(docs[1]))
                self.assertEqual(
                    [doc["entity"] for doc in docs],
                    ["Rio Minho", "Sacalina", "Cante se Puder", "Inês de Medeiros"],
                )
                if filename == "saudade_mcq_pt.yaml":
                    self.assertEqual(
                        task.doc_to_choice(docs[1]),
                        [
                            "Criação da Província da Sacalina do Sul",
                            "Extinção da Província da Sacalina do Sul e integração ao "
                            "Oblast de Sacalina",
                        ],
                    )


class SaudadePreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json

        from datasets import Dataset

        cls.records = [
            json.loads(line)
            for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        ]
        cls.dataset_type = Dataset
        cls.prepare = staticmethod(
            load_yaml_config(str(TASKS / "saudade_mcq_pt.yaml"))["process_docs"]
        )

    def dataset(self, records):
        return self.dataset_type.from_list(records)

    def test_canonical_decisions_are_logged_and_survive_reordering(self):
        with self.assertLogs(level="INFO") as captured:
            result = self.prepare(self.dataset(list(reversed(self.records))))
        self.assertEqual(
            result["entity"],
            ["Inês de Medeiros", "Cante se Puder", "Sacalina", "Rio Minho"],
        )
        self.assertEqual(result["answer_index"], [0, 1, 0, 0])
        self.assertIn("prepared 4 of 5 records; excluded 1", "\n".join(captured.output))
        self.assertIn("manifest line 131", "\n".join(captured.output))
        self.assertIn("manifest line 3479", "\n".join(captured.output))
        self.assertIn("manifest line 5299", "\n".join(captured.output))

    def test_prepared_input_is_idempotent(self):
        result = self.prepare(self.dataset(self.records))
        self.assertEqual(self.prepare(result).to_list(), result.to_list())
        self.assertEqual(
            result[0]["prompt"],
            "Qual dos dois acontecimentos sobre o Rio Minho ocorreu primeiro:",
        )

    def test_normalized_exact_answer_can_select_second_option(self):
        doc = {
            **self.records[0],
            "expected_answers": [
                "  INSCRIÇÃO DAS PESQUEIRAS DO RIO MINHO NO INVENTÁRIO NACIONAL "
                "DO PATRIMÓNIO IMATERIAL  "
            ],
        }
        result = self.prepare(self.dataset([doc]))
        self.assertEqual(result[0]["answer_index"], 1)
        self.assertEqual(result[0]["expected_answers"], doc["expected_answers"])

    def test_malformed_or_ambiguous_records_fail_before_evaluation(self):
        invalid = [
            ({"question": ""}, "question must be a nonempty string"),
            ({"question": None}, "question must be a nonempty string"),
            ({"facts_pair": ["one"]}, "facts_pair must contain two"),
            ({"facts_pair": ["one", ""]}, "facts_pair must contain two"),
            ({"expected_answers": []}, "expected_answers must contain one"),
            ({"expected_answers": None}, "expected_answers must contain one"),
            ({"expected_answers": [""]}, "expected_answers must contain one"),
            ({"expected_answers": ["unmatched answer"]}, "exactly one option"),
            (
                {"facts_pair": ["one", " ONE "], "expected_answers": ["one"]},
                "exactly one option",
            ),
            ({"question": "Which happened first?"}, "not a canonical exclusion"),
            ({"prompt": "incorrect prefix"}, "prepared prompt disagrees"),
            ({"answer_index": 1}, "prepared answer_index disagrees"),
            ({"answer_index": False}, "prepared answer_index disagrees"),
        ]
        for changes, message in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(
                    ValueError, "SAUDADE source row 1.*" + message
                ):
                    self.prepare(self.dataset([{**self.records[0], **changes}]))

    def test_prefix_rules_match_original_preparation(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "saudade_original", TASKS / "process_temp_reasoning.py"
        )
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        for question, prefix in [
            (
                "Qual evento ocorreu primeiro: a fundação ou o nascimento?",
                "Qual evento ocorreu primeiro:",
            ),
            ("Compare os eventos: a fundação ou o nascimento?", "Compare os eventos:"),
            (
                "Qual evento ocorreu primeiro entre a fundação e o nascimento?",
                "Qual evento ocorreu primeiro",
            ),
        ]:
            with self.subTest(question=question):
                self.assertEqual(
                    original.extract_prompt_prefix(question), (prefix, True)
                )
                result = self.prepare(
                    self.dataset([{**self.records[0], "question": question}])
                )
                self.assertEqual(result[0]["prompt"], prefix)
                self.assertEqual(result[0]["answer_index"], 0)
        question = "Qual evento foi mais antigo?"
        self.assertEqual(original.extract_prompt_prefix(question), (None, False))
        with self.assertRaisesRegex(ValueError, "not a canonical exclusion"):
            self.prepare(self.dataset([{**self.records[0], "question": question}]))

    def test_policy_requires_exact_source_content(self):
        for index in [2, 3, 4]:
            with self.subTest(record=index):
                doc = {
                    **self.records[index],
                    "question": self.records[index]["question"] + " ",
                }
                with self.assertRaises(ValueError):
                    self.prepare(self.dataset([doc]))

    def test_duplicate_policy_record_is_not_silently_counted_twice(self):
        with self.assertRaisesRegex(ValueError, "occurs more than once"):
            self.prepare(self.dataset([self.records[3], self.records[3]]))

    def test_empty_evaluation_set_fails_explicitly(self):
        for records in [[], [self.records[2]]]:
            with self.subTest(records=records):
                with self.assertRaisesRegex(ValueError, "no evaluation records"):
                    self.prepare(self.dataset(records))


if __name__ == "__main__":
    unittest.main()
