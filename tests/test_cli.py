import importlib
import importlib.util
import os
import sys
import types
import unittest

os.environ.setdefault("HF_CACHE_REMOTE", "/tmp/hf-cache")

if importlib.util.find_spec("dotenv") is None:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv_stub

if importlib.util.find_spec("jinja2") is None:
    jinja2_stub = types.ModuleType("jinja2")
    jinja2_stub.Environment = object
    jinja2_stub.FileSystemLoader = object
    jinja2_stub.StrictUndefined = object
    sys.modules["jinja2"] = jinja2_stub

if importlib.util.find_spec("tyro") is None:
    tyro_stub = types.ModuleType("tyro")
    tyro_stub.cli = lambda *_args, **_kwargs: None
    sys.modules["tyro"] = tyro_stub

cli_module = importlib.import_module("scripts.cli")
EvalConfig = cli_module.EvalConfig
_build_lm_eval_argv = cli_module._build_lm_eval_argv


class BuildLmEvalArgvTest(unittest.TestCase):
    def test_flattens_harness_args_without_hf_logging(self) -> None:
        config = EvalConfig(
            model_name_or_path="amalia-llm/AMALIA-9B-0626-DPO",
            model_output_dir="./outputs",
        )
        template_args = {
            "log_results_hf": False,
            "harness_args": {
                "model": "hf",
                "model_args": "pretrained=amalia-llm/AMALIA-9B-0626-DPO",
                "include_path": "./tasks/amalia-bench",
                "apply_chat_template": True,
                "log_samples": False,
                "limit": None,
            },
        }

        argv = _build_lm_eval_argv("calame", template_args, config)

        self.assertEqual(
            argv[:8],
            [
                "lm_eval",
                "--tasks",
                "calame",
                "--output_path",
                "./outputs",
                "--seed",
                "42",
                "--trust_remote_code",
            ],
        )
        self.assertIn("--model", argv)
        self.assertIn("--include_path", argv)
        self.assertIn("--apply_chat_template", argv)
        self.assertNotIn("--hf_hub_log_args", argv)


if __name__ == "__main__":
    unittest.main()
