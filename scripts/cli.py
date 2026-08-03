"""Tyro-based CLI for orchestrating AMALIA evaluations."""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import jinja2
import tyro

from configs.models import HARNESS_MODEL_CONFIGS
from configs.slurm import SLURM_CONFIGS
from tasks.configs import HARNESS_TASK_CONFIGS, TASK_GROUPS

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass(frozen=True)
class HarnessArgs:
    """Arguments specific to the evaluation harness.

    Parameter precedence in AMALIA Eval:
    - Task defaults (HARNESS_TASK_CONFIGS) provide the initial values for a given task.
    - Model overrides (HARNESS_MODEL_CONFIGS) refine those defaults for a specific model.
    - User CLI overrides (values passed via --harness-args-*) take final precedence.

    Implementation detail:
    - We start from these dataclass defaults only as a safe base so required keys exist.
    - The effective order of application is: dataclass defaults -> HARNESS_TASK_CONFIGS -> HARNESS_MODEL_CONFIGS -> CLI overrides.
    - Only values the user explicitly sets on the CLI override prior layers; unspecified CLI fields do not reset values.
    """

    model: str = "vllm"   # vllm or hf
    apply_chat_template: bool = False
    num_fewshot: int | None = None
    fewshot_as_multiturn: bool = False
    cache_requests: bool = False
    local_base_dir: str | None = None
    # path to amalia-bench tasks
    include_path: str = "./tasks/amalia-bench"
    limit: int | None = None
    gen_kwargs: str | None = None
    batch_size: str = "auto"
    use_cache: str | None = None
    trust_remote_code: bool = False
    log_samples: bool = True
    system_instruction: str | None = None

    metadata: str | None = None
    confirm_run_unsafe_code: bool = False

@dataclass
class EvalConfig:
    """CLI arguments for running evaluations."""

    model_name_or_path: str
    model_output_dir: str

    max_length: int = 4096
    think_end_token: str = "</think>"
    enable_thinking: str = "False"

    # Optional top-level LoRA adapter path
    lora_local_path: str | None = None

    tasks: list[str] | None = None
    task_group: str | None = None
    slurm_config: str | None = None

    # SLURM / cluster configuration
    job_name: str | None = "amalia-eval"
    time: str = "01:00:00"
    cpus_per_task: int = 16
    gpus_per_node: int = 1
    gres: str | None = None
    mem: str = "32G"
    partition: str | None = None
    qos: str | None = None
    account: str | None = None

    # Conda-specific variables
    activate_env: str | None = None
    conda_home: str | None = None

    harness_args: HarnessArgs = HarnessArgs()

    # Template / environment configuration
    templates_file: str = "./templates/launch.sbatch.j2"
    hf_home: str | None = None
    hf_hub_cache: str | None = None
    hf_assets_cache: str | None = None
    hf_datasets_cache: str | None = None
    hf_hub_offline: bool | None = None

    y: bool = False  # auto-confirm submission


def _resolve_tasks(config: EvalConfig) -> Sequence[str]:
    """Resolve and normalize task list from CLI config.

    Accepts these common forms for `config.tasks`:
    - None (use `task_group`)
    - List[str] like ["task1", "task2"]
    - Single-element list where the string contains whitespace or commas, e.g. ["task1 task2"] or ["task1,task2"]

    The function normalizes to a flat list of task names, splitting on commas and whitespace when necessary.
    """
    if config.tasks:
        normalized: list[str] = []
        for t in config.tasks:
            if not isinstance(t, str):
                normalized.append(str(t))
                continue
            # Split on commas first, then on whitespace in each piece
            for part in t.split(","):
                for token in part.strip().split():
                    if token:
                        normalized.append(token)
        return normalized

    if config.task_group not in TASK_GROUPS:
        raise ValueError(
            f"Unknown task group: {config.task_group}. "
            f"Available groups: {sorted(TASK_GROUPS.keys())}"
        )
    return TASK_GROUPS[config.task_group]


def _apply_slurm_defaults(config: EvalConfig) -> None:
    if not config.slurm_config:
        return

    if config.slurm_config not in SLURM_CONFIGS:
        raise ValueError(
            f"Unknown SLURM profile '{config.slurm_config}'. "
            f"Available profiles: {sorted(SLURM_CONFIGS.keys())}"
        )

    selected_config = SLURM_CONFIGS[config.slurm_config]
    logging.info(f"Selected config: {selected_config}")

    for key, value in asdict(selected_config).items():
        setattr(config, key, value)


def _resolve_model_config(model_name_or_path: str) -> dict:
    """Retrieve model-specific configuration if available."""
    if model_name_or_path in HARNESS_MODEL_CONFIGS:
        return HARNESS_MODEL_CONFIGS[model_name_or_path]
    # Check for model name without path components
    model_basename = os.path.basename(model_name_or_path.rstrip("/"))
    if model_basename in HARNESS_MODEL_CONFIGS:
        return HARNESS_MODEL_CONFIGS[model_basename]
    return {}


def _build_harness_args(task: str, config: EvalConfig) -> dict:
    """Build the merged harness args with precedence.

    Precedence (lowest to highest):
    - Dataclass defaults (safe base)
    - HARNESS_TASK_CONFIGS (task defaults)
    - HARNESS_MODEL_CONFIGS (model overrides)
    - CLI overrides (only explicitly set values)
    """
    # 1) Dataclass defaults as safe base
    default_harness = asdict(HarnessArgs())
    harness_cfg: dict = deepcopy(default_harness)

    # 2) Task defaults
    task_cfg = deepcopy(HARNESS_TASK_CONFIGS.get(task, {}))
    for key, value in task_cfg.items():
        if (
            key in harness_cfg
            and isinstance(harness_cfg[key], dict)
            and isinstance(value, dict)
        ):
            harness_cfg[key].update(value)
        else:
            harness_cfg[key] = value

    # 3) Model overrides
    model_config = _resolve_model_config(config.model_name_or_path)
    for key, value in model_config.items():
        if (
            key in harness_cfg
            and isinstance(harness_cfg[key], dict)
            and isinstance(value, dict)
        ):
            harness_cfg[key].update(value)
        else:
            harness_cfg[key] = value

    # 4) User-provided CLI overrides only (avoid overriding with dataclass defaults)
    current_harness = asdict(config.harness_args)
    user_overrides: dict = {}
    for key, value in current_harness.items():
        if key not in default_harness:
            # Unknown key from CLI, treat as override
            user_overrides[key] = value
            continue
        if value != default_harness[key]:
            user_overrides[key] = value

    for key, value in user_overrides.items():
        if (
            key in harness_cfg
            and isinstance(harness_cfg[key], dict)
            and isinstance(value, dict)
        ):
            harness_cfg[key].update(value)
        else:
            harness_cfg[key] = value

    return harness_cfg


def _render_and_submit(config: EvalConfig) -> None:
    _apply_slurm_defaults(config)

    selected_tasks = _resolve_tasks(config)
    logger.info(f"Selected tasks: {selected_tasks}!")

    templates_dir = os.path.dirname(config.templates_file)
    template_name = os.path.basename(config.templates_file)
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(templates_dir),
        undefined=jinja2.StrictUndefined,
    )
    template = jinja_env.get_template(template_name)

    output_dir = os.path.abspath(config.model_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    base_args = asdict(config)
    base_args.pop("tasks", None)
    base_args.pop("slurm", None)

    for task in selected_tasks:
        # we are not using this
        """
        if task not in HARNESS_TASK_CONFIGS:
            available = ", ".join(sorted(HARNESS_TASK_CONFIGS.keys()))
            print(f"WARNING: {task} not found. Available tasks: {available}")
        """

        template_args = deepcopy(dict(base_args))
        template_args["tasks"] = [task]

        # Build and assign the merged harness args
        template_args["harness_args"] = _build_harness_args(task, config)

        max_length = config.max_length

        # check if lora was passed
        is_using_lora = template_args.get("lora_local_path", None)

        # special case for some models which are not compatible with the version of VLLM
        hf_exception_models = [
            "gemma-2",
            "ministral",
            "gemma-3-12b-it",
            "Qwen3.5",
        ]

        # check incompatible with the ones above using a contains relationship
        is_incompatible = any(incompatible_model.lower() in config.model_name_or_path.lower() for incompatible_model in hf_exception_models)
        if is_incompatible and template_args["harness_args"]["model"] == "vllm":
            logger.warning(f"Detected incompatible model {config.model_name_or_path} with VLLM version switching to HF")
            template_args["harness_args"]["model"] = "hf"

        # use trust_remote_code flag
        trust_remote_code = bool(template_args["harness_args"].get("trust_remote_code", False))
        if trust_remote_code:
            logger.warning(
                "trust_remote_code is ENABLED: custom code from "
                f"'{config.model_name_or_path}' will execute on model load."
            )

        # check if is hf or vllm and use a different set of model args
        if template_args["harness_args"]["model"] == "vllm":
            model_args = {
                "pretrained": config.model_name_or_path,
                "max_model_len": max_length,
                "gpu_memory_utilization": "0.8",  # Fraction of GPU memory allocated to vLLM's engine/KV Cache. Remaining is reserved for runtime overhead (e.g., 0.70 reserves 30% for safety).
                "dtype": "bfloat16",  # all models using the same dtype (e.g. gemma is fp32 by default)
                "think_end_token": config.think_end_token,  # to parse thinking steps
                "enable_thinking": config.enable_thinking,  # whether to enable thinking or not
                "trust_remote_code": str(trust_remote_code),  # gated by operator flag
            }
            if is_using_lora:
                model_args["enable_lora"] = "True"  # whether to enable lora or not
                model_args["lora_local_path"] = template_args["lora_local_path"]
        elif template_args["harness_args"]["model"] == "hf":
            model_args = {
                "pretrained": config.model_name_or_path,
                "max_length": max_length,
                "dtype": "bfloat16",
                "think_end_token": config.think_end_token,
                "enable_thinking": config.enable_thinking,
                "trust_remote_code": str(trust_remote_code),  # gated by operator flag
            }
            if is_using_lora:
                model_args["peft"] = template_args["lora_local_path"]

        else:
            raise ValueError(f"Unknown model type, use 'hf' or 'vllm': {template_args['harness_args']['model']}")

        # Format the model args string as expected by lm_eval
        template_args["harness_args"]["model_args"] = [
            f"{k}={v}"
            for k, v in model_args.items()
        ]
        template_args["harness_args"]["model_args"] = ",".join(
            template_args["harness_args"]["model_args"]
        )

        # Extract short model name from path (e.g., "Meta-Llama-3.1-8B" from full path)
        model_basename = os.path.basename(config.model_name_or_path.rstrip("/"))
        short_model_name = model_basename.replace("_", "-")[:30]
        template_args["job_name"] = f"{short_model_name}_{task}"

        script_path = os.path.join(output_dir, f"{template_name}_{task}")

        # print("Template arguments:")
        # print("=" * 40)
        # for key, value in sorted(template_args.items()):
        #     print(f"{key}: {value}")
        # print("=" * 40)

        rendered = template.render(template_args)
        # print(f"Rendering template for task: {task}")
        # print(f"Script path: {script_path}")
        # print("=" * 40)
        # print(rendered)
        # print("=" * 40)

        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(rendered)

        # print(f"Evaluating checkpoint {config.model_name_or_path}")

        if config.y or input(f"Submit job for task '{task}'? [y/N]: ").lower() == "y":
            result = subprocess.run(["sbatch", "-v", script_path], capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"Error submitting job: {result.stderr}")
            else:
                print(result.stdout)
        else:
            print(f"Skipped submission for task: {task}")


def launch_inspect(argv: Sequence[str]) -> None:
    """Load the Flask inspection app without shadowing stdlib inspect."""

    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "inspect" / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Inspection app not found at {app_path}")

    spec = importlib.util.spec_from_file_location("amalia_inspect_app", app_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load inspection app module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise AttributeError("Inspection app module missing main() function.")
    module.main(argv)


def main() -> None:
    """Entry point for the AMALIA CLI."""

    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        launch_inspect(sys.argv[2:])
        return

    config = tyro.cli(EvalConfig)
    _render_and_submit(config)


def inspect_main() -> None:
    """Console-script entry point for the inspection UI."""

    launch_inspect(sys.argv[1:])


if __name__ == "__main__":
    main()
