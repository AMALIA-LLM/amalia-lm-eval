from pathlib import Path

BENCHMARK_PATH = str(Path(__file__).resolve().parents[1] / "tasks" / "amalia-bench")
LAUNCH_TEMPLATE = str(Path(__file__).resolve().parent / "templates" / "launch.sbatch.j2")
