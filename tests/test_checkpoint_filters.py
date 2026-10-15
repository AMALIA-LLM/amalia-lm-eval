import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from create_result_tables import extract_results


class CheckpointFilterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        for i, model in enumerate(("checkpoint-100", "checkpoint-500", "EuroLLM-9B")):
            path = Path(self.temp_dir.name) / f"results_2026-10-15T00-00-00.{i:06d}.json"
            path.write_text(json.dumps({
                "model_name": model,
                "configs": {"arc_challenge_mt_pt": {"num_fewshot": 0}},
                "results": {"arc_challenge_mt_pt": {"acc,none": 0.5}},
            }), encoding="utf-8")

    def models(self, **kwargs):
        results = extract_results(
            [self.temp_dir.name],
            only_most_recent=False,
            ignore_subcategories=False,
            **kwargs,
        )
        return set(results["model_name"].astype(str))

    def test_checkpoint_filters_without_excluded_models(self):
        self.assertEqual(
            self.models(min_checkpoint=500),
            {"checkpoint-500", "EuroLLM-9B"},
        )
        self.assertEqual(
            self.models(max_checkpoint=100),
            {"checkpoint-100", "EuroLLM-9B"},
        )
        self.assertEqual(
            self.models(checkpoints="500"),
            {"checkpoint-500", "EuroLLM-9B"},
        )

    def test_checkpoint_filters_with_excluded_models(self):
        self.assertEqual(
            self.models(min_checkpoint=500, exclude_models=["EuroLLM-9B"]),
            {"checkpoint-500"},
        )


if __name__ == "__main__":
    unittest.main()
