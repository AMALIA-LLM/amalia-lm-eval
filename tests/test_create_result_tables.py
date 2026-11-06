import argparse
import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest.mock
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
create_result_tables = importlib.import_module("create_result_tables")


class ParseCheckpointsTests(unittest.TestCase):
    def test_ignores_empty_items_and_whitespace(self):
        parse = create_result_tables.parse_checkpoints
        self.assertEqual(parse("100,"), "100")
        self.assertEqual(parse(" 100 , ,300, "), "100,300")

    def test_rejects_non_integers(self):
        parse = create_result_tables.parse_checkpoints
        for value in ("abc", "100,x", ",", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse(value)

    def test_cli_reports_bad_checkpoints_as_usage_error(self):
        argv = ["create_result_tables.py", "--folder_path", ".", "--checkpoints", "100,x"]
        stderr = io.StringIO()
        with unittest.mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                create_result_tables.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--checkpoints", stderr.getvalue())

    def test_cli_accepts_trailing_comma(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i, model in enumerate(("checkpoint-100", "checkpoint-500")):
                path = Path(tmp) / f"results_2026-10-15T00-00-00.{i:06d}.json"
                path.write_text(json.dumps({
                    "model_name": model,
                    "configs": {"arc_challenge_mt_pt": {"num_fewshot": 0}},
                    "results": {"arc_challenge_mt_pt": {"acc,none": 0.5}},
                }), encoding="utf-8")
            output = Path(tmp) / "out" / "results.csv"
            # an empty --exclude_models makes main apply the checkpoint filter
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "create_result_tables.py"), "--folder_path", tmp,
                 "--output_path", str(output), "--exclude_models", "--checkpoints", "500,"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set(pd.read_csv(output)["model_name"]), {"checkpoint-500"})


class TagOrderTests(unittest.TestCase):
    def test_tags_keep_first_seen_order(self):
        Tags = create_result_tables.Tags
        self.assertEqual(
            create_result_tables.add_tags_to_datasets("global_piqa_pt"),
            [Tags.COMMONSENSE, Tags.MCQ, Tags.PT_SPECIFIC],
        )

    def test_tags_do_not_depend_on_hash_seed(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from create_result_tables import add_tags_to_datasets; "
            "print(add_tags_to_datasets('global_piqa_pt'))"
        )
        outputs = set()
        for seed in range(1, 7):
            env = dict(os.environ, PYTHONHASHSEED=str(seed), MPLBACKEND="Agg")
            result = subprocess.run(
                [sys.executable, "-c", code, str(SCRIPTS)],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.add(result.stdout)
        self.assertEqual(len(outputs), 1, outputs)


if __name__ == "__main__":
    unittest.main()
