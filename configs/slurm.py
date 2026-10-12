"""SLURM configuration presets for common infrastructures."""

from dataclasses import dataclass
from typing import Any
import os
import dotenv

# load the env variables from a .env file
dotenv.load_dotenv()


def _remote_hf_cache(subdir: str) -> str | None:
    """Path under $HF_CACHE_REMOTE, or None when it is not configured.

    Evaluated at import time, so it must not raise: otherwise every CLI
    invocation (including ``amalia --help``) fails when HF_CACHE_REMOTE is unset.
    """
    base = os.getenv("HF_CACHE_REMOTE")
    return os.path.join(base, subdir) if base else None


@dataclass(frozen=True)
class SardineConfig:
    """Container for default SLURM-related launch parameters."""

    cpus_per_task: int = 20
    mem: str | None = "32G"
    partition: str | None = "a6000"
    time: str = "04:00:00"
    qos: str = "gpu-short"
    job_name: str | None = None
    templates_file: str = "./templates/launch.sbatch.j2"
    hf_hub_offline: bool = False
    conda_home: str = "/mnt/data-poseidon/giuseppe/miniconda3"


@dataclass(frozen=True)
class SardineH100(SardineConfig):
    partition: str = "h100"
    qos: str = "gpu-h100"


@dataclass(frozen=True)
class MarenostrumConfig:
    """Container for SLURM-related launch defaults."""

    cpus_per_task: int = 20
    gpus_per_node: int = 1
    mem: str | None = None
    gres: str = "gpu:1"
    partition: str | None = None
    time: str = "16:00:00"  # "1:00:00"
    qos: str = "acc_ehpc"  # acc_ehpc or acc_debug
    account: str = "epor32"
    job_name: str | None = None
    activate_env: str = (
        os.getenv("ACTIVATE_ENV", "py312_amalia")
    )
    templates_file: str = (
        "./templates/launch.sbatch.j2"
    )
    hf_home: str | None = _remote_hf_cache("huggingface")
    hf_hub_cache: str | None = _remote_hf_cache("huggingface/hub")
    hf_assets_cache: str | None = _remote_hf_cache("huggingface/assets")
    hf_datasets_cache: str | None = _remote_hf_cache("huggingface/datasets")
    hf_hub_offline: bool = True


@dataclass(frozen=True)
class NovaConfig:
    """Container for SLURM-related launch defaults."""

    cpus_per_task: int = 8
    gpus_per_node: int = 1  # if specifying a specific gpu (i.e. using pcie or sxm4) set this to 0
    mem: str | None = "32GB"
    gres: str = "gpu:1"  # "gpu:h200:1" # "gpu:nvidia_a100-pcie-40gb:1"  # gpu:nvidia_a100-sxm4-40gb:1
    partition: str | None = None # "high_priority"
    time: str = "12:00:00"
    job_name: str | None = None
    activate_env: str = os.getenv("ACTIVATE_ENV", "py312_amalia")
    templates_file: str = "./templates/launch.sbatch.j2"

    hf_home: str = "/data/huggingface_cache"
    hf_hub_cache: str = "/data/huggingface_cache/hub"
    hf_assets_cache: str = "/data/huggingface_cache/assets"
    hf_datasets_cache: str = "/data/huggingface_cache/datasets"
    hf_hub_offline: bool = False

    conda_home: str = "/opt/Anaconda-2021.05/condabin/conda"



SLURM_CONFIGS: dict[str, Any] = {
    "marenostrum": MarenostrumConfig(),
    "sardine": SardineConfig(),
    "sardine-h100": SardineH100(),
    "nova": NovaConfig(),
}