"""The CLI must import (and show --help) even when HF_CACHE_REMOTE is unset."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env_without_hf_cache():
    env = {k: v for k, v in os.environ.items() if k != "HF_CACHE_REMOTE"}
    env["PYTHONPATH"] = str(REPO)
    return env


def _env_with_empty_hf_cache():
    # An empty value counts as "not configured", and load_dotenv() never
    # overrides a variable that is already set, so a developer's .env cannot
    # leak a real HF_CACHE_REMOTE into the test.
    env = _env_without_hf_cache()
    env["HF_CACHE_REMOTE"] = ""
    return env


class ImportWithoutHfCacheRemoteTest(unittest.TestCase):
    def test_cli_help_works_without_hf_cache_remote(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.cli", "--help"],
            cwd=REPO / "tests",
            env=_env_without_hf_cache(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("model-name-or-path", result.stdout)

    def test_marenostrum_profile_reports_missing_hf_cache_remote(self):
        code = (
            "from scripts.cli import EvalConfig, _apply_slurm_defaults\n"
            "cfg = EvalConfig(model_name_or_path='m', model_output_dir='o',\n"
            "                 slurm_config='marenostrum')\n"
            "_apply_slurm_defaults(cfg)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO / "tests",
            env=_env_with_empty_hf_cache(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError", result.stderr)
        self.assertIn("needs HF_CACHE_REMOTE", result.stderr)

    def test_marenostrum_profile_uses_hf_cache_remote_when_set(self):
        env = _env_without_hf_cache()
        env["HF_CACHE_REMOTE"] = "/scratch/cache"
        code = (
            "from configs.slurm import SLURM_CONFIGS\n"
            "print(SLURM_CONFIGS['marenostrum'].hf_hub_cache)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO / "tests",
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "/scratch/cache/huggingface/hub")


if __name__ == "__main__":
    unittest.main()
