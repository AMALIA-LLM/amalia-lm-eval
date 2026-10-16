from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "inspect" / "app.py"
RESULT_STAMP = "2026-07-02T01-02-03"


def load_inspect_app() -> ModuleType:
    spec = importlib.util.spec_from_file_location("amalia_inspect_app_test", APP_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load inspection app from {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_result_fixture(root: Path) -> None:
    task = "task-one"
    model_dir = root / "model alpha"
    model_dir.mkdir(parents=True)
    payload = {
        "results": {
            task: {
                "acc,none": 0.75,
                "acc_stderr,none": 0.01,
            }
        },
        "configs": {
            task: {
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "none",
                    }
                ]
            }
        },
    }
    (model_dir / f"results_{RESULT_STAMP}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (model_dir / f"samples_{task}_{RESULT_STAMP}.jsonl").write_text(
        json.dumps({"prompt": "Question?", "target": "Answer"}) + "\n",
        encoding="utf-8",
    )


class InspectStaticExportTest(unittest.TestCase):
    def test_export_static_writes_relative_site(self) -> None:
        inspect_app = load_inspect_app()
        with TemporaryDirectory(dir=Path.home()) as tmpdir:
            tmp_path = Path(tmpdir)
            results_root = tmp_path / "results"
            out_dir = tmp_path / "site"
            (tmp_path / "alternate").mkdir()
            write_result_fixture(results_root)

            previous_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                index_path = inspect_app.export_static(Path("results"), Path("site"))
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(index_path, (out_dir / "index.html").resolve())
            self.assertTrue((out_dir / "index.html").is_file())
            self.assertTrue((out_dir / "overview.html").is_file())
            self.assertTrue((out_dir / "overview" / "export.csv").is_file())
            self.assertTrue((out_dir / "static" / "style.css").is_file())

            model_path = out_dir / "model" / "model%20alpha.html"
            task_path = (
                out_dir / "model" / "model%20alpha" / "task" / "task-one.html"
            )
            stamp_task_path = (
                out_dir
                / "model"
                / "model%20alpha"
                / "task"
                / "task-one"
                / f"{RESULT_STAMP}.html"
            )
            self.assertTrue(model_path.is_file())
            self.assertTrue(task_path.is_file())
            self.assertTrue(stamp_task_path.is_file())

            index_html = (out_dir / "index.html").read_text(encoding="utf-8")
            model_html = model_path.read_text(encoding="utf-8")
            task_html = task_path.read_text(encoding="utf-8")
            stamp_task_html = stamp_task_path.read_text(encoding="utf-8")
            exported_html = [
                index_html,
                model_html,
                task_html,
                stamp_task_html,
            ]
            for html in exported_html:
                self.assertNotIn("http://localhost", html)
                self.assertNotIn("http://127.0.0.1", html)
                self.assertNotIn('href="/', html)
                self.assertNotIn('src="/', html)
                self.assertNotIn('action="/', html)

            # Filenames on disk are percent-encoded ("model%20alpha.html"); the
            # emitted hrefs double-encode the "%" so a browser decodes them back
            # to the on-disk names instead of 404ing on "model alpha.html".
            self.assertIn('href="model/model%2520alpha.html"', index_html)
            self.assertIn('href="static/style.css"', index_html)
            self.assertIn(
                f'href="model%2520alpha/task/task-one/{RESULT_STAMP}.html"',
                model_html,
            )

            absolute_root = str(results_root.resolve())
            home_path = str(Path.home())
            for html_path in out_dir.rglob("*.html"):
                html = html_path.read_text(encoding="utf-8")
                self.assertNotIn(absolute_root, html)
                if home_path not in {"", "/"}:
                    self.assertNotIn(home_path, html)


if __name__ == "__main__":
    unittest.main()
