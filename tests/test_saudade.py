import unittest
from pathlib import Path

from lm_eval.api.task import ConfigurableTask
from lm_eval.utils import load_yaml_config

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / 'tasks/amalia-bench/saudade-pt'
FIXTURE = REPO / 'tests/fixtures/saudade_pt_binary.jsonl'


class SaudadeTaskTest(unittest.TestCase):
    def test_released_records_load_in_all_task_variants(self):
        for filename, targets in [
            ('saudade_mcq_pt.yaml', [0, 0, 1, 0]),
            ('saudade_chat_pt.yaml', ['A', 'A', 'B', 'A']),
            ('saudade_chat_cot_pt.yaml', ['A', 'A', 'B', 'A']),
        ]:
            with self.subTest(task=filename):
                config = load_yaml_config(str(TASKS / filename))
                config.update(
                    dataset_path='json',
                    dataset_name=None,
                    dataset_kwargs={'data_files': {'test': str(FIXTURE)}},
                )
                task = ConfigurableTask(config=config)
                docs = task.eval_docs
                self.assertEqual(len(docs), 4)
                self.assertEqual([task.doc_to_target(doc) for doc in docs], targets)
                self.assertIn('Rio Minho', task.doc_to_text(docs[0]))
                self.assertIn('Sacalina', task.doc_to_text(docs[1]))
                self.assertEqual(
                    [doc['entity'] for doc in docs],
                    ['Rio Minho', 'Sacalina', 'Cante se Puder', 'Inês de Medeiros'],
                )
                if filename == 'saudade_mcq_pt.yaml':
                    self.assertEqual(
                        task.doc_to_choice(docs[1]),
                        [
                            'Criação da Província da Sacalina do Sul',
                            'Extinção da Província da Sacalina do Sul e integração ao '
                            'Oblast de Sacalina',
                        ],
                    )


if __name__ == '__main__':
    unittest.main()
