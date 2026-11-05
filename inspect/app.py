"""Flask application for browsing lm-eval-harness results."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import posixpath
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from html import escape as html_escape
from html import unescape as html_unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

from flask import Flask, Response, abort, g, redirect, render_template, request, url_for

TIMESTAMP_PATTERN = re.compile(
    r"results_(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?)"
)
HTML_LINK_ATTR_PATTERN = re.compile(
    r"(?P<attr>\b(?:href|src|action)\s*=\s*)(?P<quote>['\"])(?P<url>.*?)(?P=quote)",
    re.IGNORECASE,
)

DEFAULT_TASK_DISPLAY_NAMES: Dict[str, str] = {
    "amalia_arc_challenge_mt_pt": "ARC-C",
    "commonsense_qa-mt-pt": "CommQA",
    "openbookqa-mt-pt": "OpenQA",
    "piqa-mt-pt": "PiQA",
    "amalia_hellaswag_mt_pt": "HSwG",
    "amalia_mmlu_continuation_mt_pt": "MMLU",
    "winogrande-mt-pt": "WG",
}

DISPLAY_NAME_FILENAME = ".amalia_inspect_display_names.json"
VISIBILITY_FILENAME = ".amalia_inspect_visibility.json"
EXPORT_MARKER = ".amalia_inspect_export"


class DisplayNameStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.task_overrides: Dict[str, str] = {}
        self.model_overrides: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        tasks = data.get("tasks", {})
        if isinstance(tasks, dict):
            for key, value in tasks.items():
                if isinstance(key, str) and isinstance(value, str):
                    trimmed = value.strip()
                    if trimmed and trimmed != key:
                        self.task_overrides[key] = trimmed
        models = data.get("models", {})
        if isinstance(models, dict):
            for key, value in models.items():
                if isinstance(key, str) and isinstance(value, str):
                    trimmed = value.strip()
                    if trimmed and trimmed != key:
                        self.model_overrides[key] = trimmed

    def save(self) -> None:
        payload = {
            "tasks": self.task_overrides,
            "models": self.model_overrides,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def get_task_display_name(self, task: str) -> str:
        if task in self.task_overrides:
            override = self.task_overrides[task].strip()
            if override:
                return override
        return DEFAULT_TASK_DISPLAY_NAMES.get(task, task)

    def set_task_display_name(self, task: str, display: str) -> None:
        trimmed = display.strip()
        if not trimmed or trimmed == task:
            self.task_overrides.pop(task, None)
            return
        self.task_overrides[task] = trimmed

    def get_model_display_name(self, model: str) -> str:
        if model in self.model_overrides:
            override = self.model_overrides[model].strip()
            if override:
                return override
        return self._default_model_display_name(model)

    def set_model_display_name(self, model: str, display: str) -> None:
        trimmed = display.strip()
        if not trimmed or trimmed == model:
            self.model_overrides.pop(model, None)
            return
        self.model_overrides[model] = trimmed

    def task_keys(self) -> List[str]:
        return sorted(
            set(self.task_overrides.keys()) | set(DEFAULT_TASK_DISPLAY_NAMES.keys())
        )

    def model_keys(self) -> List[str]:
        return sorted(self.model_overrides.keys())

    @staticmethod
    def _default_model_display_name(model: str) -> str:
        parts = [segment.replace("_", " ").strip() for segment in model.split("__")]
        cleaned = [segment for segment in parts if segment]
        if cleaned:
            return " / ".join(cleaned)
        return model

    def task_overrides_copy(self) -> Dict[str, str]:
        return dict(self.task_overrides)

    def model_overrides_copy(self) -> Dict[str, str]:
        return dict(self.model_overrides)


class VisibilityStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._data = {}
            return
        if isinstance(payload, dict):
            self._data = payload
        else:
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _root_key(root: Path) -> str:
        try:
            return str(root.resolve())
        except FileNotFoundError:
            return str(root)

    def _ensure_root(self, root: Path) -> Dict[str, Any]:
        key = self._root_key(root)
        existing = self._data.get(key)
        if isinstance(existing, dict):
            return existing
        created: Dict[str, Any] = {}
        self._data[key] = created
        return created

    def hidden_models(self, root: Path) -> set[str]:
        bucket = self._ensure_root(root)
        raw = bucket.get("hidden_models", [])
        if not isinstance(raw, list):
            return set()
        return {item for item in raw if isinstance(item, str) and item}

    def hidden_tasks(self, root: Path) -> set[str]:
        bucket = self._ensure_root(root)
        raw = bucket.get("hidden_tasks", [])
        if not isinstance(raw, list):
            return set()
        return {item for item in raw if isinstance(item, str) and item}

    def set_hidden_models(self, root: Path, hidden: Iterable[str]) -> None:
        bucket = self._ensure_root(root)
        cleaned = sorted({item for item in hidden if isinstance(item, str) and item})
        bucket["hidden_models"] = cleaned

    def set_hidden_tasks(self, root: Path, hidden: Iterable[str]) -> None:
        bucket = self._ensure_root(root)
        cleaned = sorted({item for item in hidden if isinstance(item, str) and item})
        bucket["hidden_tasks"] = cleaned

    def is_model_visible(self, root: Path, model: str) -> bool:
        return model not in self.hidden_models(root)

    def is_task_visible(self, root: Path, task: str) -> bool:
        return task not in self.hidden_tasks(root)


_display_name_store: Optional[DisplayNameStore] = None
_visibility_store: Optional[VisibilityStore] = None


def set_display_name_store(store: DisplayNameStore) -> None:
    global _display_name_store
    _display_name_store = store


def get_display_name_store() -> DisplayNameStore:
    global _display_name_store
    if _display_name_store is None:
        fallback_path = Path.cwd() / DISPLAY_NAME_FILENAME
        _display_name_store = DisplayNameStore(fallback_path)
    return _display_name_store


def set_visibility_store(store: VisibilityStore) -> None:
    global _visibility_store
    _visibility_store = store


def get_visibility_store() -> VisibilityStore:
    global _visibility_store
    if _visibility_store is None:
        fallback_path = Path.cwd() / VISIBILITY_FILENAME
        _visibility_store = VisibilityStore(fallback_path)
    return _visibility_store


def get_task_display_name(task: str) -> str:
    return get_display_name_store().get_task_display_name(task)


def get_model_display_name(model: str) -> str:
    return get_display_name_store().get_model_display_name(model)


@dataclass
class MetricEntry:
    model: str
    task: str
    metric_key: str
    value: Any
    stderr: Optional[float]
    timestamp: datetime
    timestamp_raw: str
    result_path: Path
    samples_path: Optional[Path]

    @property
    def formatted_value(self) -> str:
        if isinstance(self.value, float):
            return f"{self.value:.3f}"
        if isinstance(self.value, int):
            return str(self.value)
        return json.dumps(self.value)

    @property
    def formatted_stderr(self) -> Optional[str]:
        if self.stderr is None:
            return None
        return f"±{self.stderr:.3f}"

    @property
    def formatted_timestamp(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def display_task_name(self) -> str:
        return get_task_display_name(self.task)


@dataclass
class StaticExportRoute:
    route_path: str
    relative_output: Path
    query: Optional[Dict[str, str]] = None


class ResultStore:
    def __init__(self, root: Path, *, base_dir: Path) -> None:
        self.root = root
        self.base_dir = base_dir
        self._validate_root()
        self.results: Dict[str, Dict[str, List[MetricEntry]]] = {}
        self.primary_tasks: Dict[str, str] = {}
        self.display_tasks_map: Dict[str, List[str]] = {}
        self._index_results()

    def _validate_root(self) -> None:
        try:
            root_resolved = self.root.resolve()
            base_resolved = self.base_dir.resolve()
        except FileNotFoundError:
            abort(404, description="Selected root directory does not exist.")
        if not root_resolved.is_dir():
            abort(404, description="Selected root is not a directory.")
        if (
            base_resolved not in root_resolved.parents
            and root_resolved != base_resolved
        ):
            abort(
                403, description="Root directory must be inside the launch directory. Try going in the root folder and running 'amalia inspect --root .'"
            )

    def _index_results(self) -> None:
        for model_dir in sorted(self.root.iterdir()):
            if not model_dir.is_dir():
                continue
            entries: Dict[str, List[MetricEntry]] = {}
            for result_file in sorted(model_dir.glob("results_*.json")):
                timestamp, raw_stamp = parse_timestamp(result_file)
                with result_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                results_payload = payload.get("results", {})
                if not results_payload:
                    continue
                task_names = list(results_payload.keys())
                display_list = self.display_tasks_map.setdefault(model_dir.name, [])
                primary_task = task_names[0]
                if model_dir.name not in self.primary_tasks:
                    self.primary_tasks[model_dir.name] = primary_task
                if len(task_names) == 1:
                    for name in task_names:
                        if name not in display_list:
                            display_list.append(name)
                else:
                    if primary_task not in display_list:
                        display_list.append(primary_task)
                for task_name, task_payload in results_payload.items():
                    entry = build_entry(
                        model=model_dir.name,
                        task=task_name,
                        timestamp=timestamp,
                        timestamp_raw=raw_stamp,
                        result_path=result_file,
                        payload=payload,
                        task_payload=task_payload,
                    )
                    if entry is None:
                        continue
                    entries.setdefault(task_name, []).append(entry)
            if entries:
                for runs in entries.values():
                    runs.sort(key=lambda item: item.timestamp)
                self.results[model_dir.name] = entries

    @property
    def tasks(self) -> List[str]:
        all_tasks = set()
        for task_entries in self.results.values():
            all_tasks.update(task_entries.keys())
        return sorted(all_tasks)

    def display_tasks(self, tasks_mode: str) -> List[str]:
        if tasks_mode != "first":
            return self.tasks
        ordered: List[str] = []
        for model in self.results.keys():
            for task in self.display_tasks_map.get(model, []):
                if task not in ordered:
                    ordered.append(task)
        return ordered

    def latest_only(self) -> Dict[str, Dict[str, MetricEntry]]:
        latest: Dict[str, Dict[str, MetricEntry]] = {}
        for model, task_entries in self.results.items():
            latest[model] = {}
            for task, runs in task_entries.items():
                latest[model][task] = runs[-1]
        return latest

    def all_runs(self) -> Dict[str, Dict[str, List[MetricEntry]]]:
        return self.results

    def get_model(self, model: str) -> Dict[str, List[MetricEntry]]:
        if model not in self.results:
            abort(404, description="Model not found.")
        return self.results[model]

    def get_entry(self, model: str, task: str, stamp: Optional[str]) -> MetricEntry:
        model_entries = self.get_model(model)
        if task not in model_entries:
            abort(404, description="Task not found for model.")
        runs = model_entries[task]
        if stamp:
            for run in runs:
                if run.timestamp_raw == stamp:
                    return run
            abort(404, description="Result timestamp not found.")
        return runs[-1]


def parse_timestamp(path: Path) -> Tuple[datetime, str]:
    match = TIMESTAMP_PATTERN.search(path.stem)
    if match:
        raw = match.group("stamp")
        for fmt in ("%Y-%m-%dT%H-%M-%S.%f", "%Y-%m-%dT%H-%M-%S"):
            try:
                return datetime.strptime(raw, fmt), raw
            except ValueError:
                continue
    stat = path.stat()
    fallback = datetime.fromtimestamp(stat.st_mtime)
    return fallback, fallback.strftime("%Y-%m-%dT%H-%M-%S")


def build_entry(
    *,
    model: str,
    task: str,
    timestamp: datetime,
    timestamp_raw: str,
    result_path: Path,
    payload: Dict[str, Any],
    task_payload: Dict[str, Any],
) -> Optional[MetricEntry]:
    main_key = resolve_primary_metric_key(payload, task, task_payload)
    if not main_key:
        return None
    value = task_payload.get(main_key)
    if value is None:
        return None
    stderr: Optional[float] = None
    metric_base = main_key.split(",", 1)[0]
    stderr_candidates = []
    if "," in main_key:
        metric_suffix = main_key.split(",", 1)[1]
        stderr_candidates.append(f"{metric_base}_stderr,{metric_suffix}")
    stderr_candidates.extend(
        [
            f"{main_key}_stderr",
            f"{metric_base}_stderr",
        ]
    )
    for candidate in stderr_candidates:
        if candidate in task_payload and isinstance(
            task_payload[candidate], (int, float)
        ):
            stderr = float(task_payload[candidate])
            break
    if stderr is None:
        for key, val in task_payload.items():
            if "stderr" not in key:
                continue
            if key.startswith(metric_base) and isinstance(val, (int, float)):
                stderr = float(val)
                break
    samples_path = locate_samples(result_path.parent, task, timestamp_raw)
    return MetricEntry(
        model=model,
        task=task,
        metric_key=main_key,
        value=value,
        stderr=stderr,
        timestamp=timestamp,
        timestamp_raw=timestamp_raw,
        result_path=result_path,
        samples_path=samples_path,
    )


def resolve_primary_metric_key(
    payload: Dict[str, Any], task: str, task_payload: Dict[str, Any]
) -> Optional[str]:
    configs = payload.get("configs", {}).get(task, {})
    metric_list = configs.get("metric_list")
    if metric_list:
        first = metric_list[0]
        metric = first.get("metric")
        aggregation = first.get("aggregation")
        candidate = f"{metric},{aggregation}" if aggregation else metric
        if metric and candidate in task_payload:
            return candidate
    for key, value in task_payload.items():
        if "stderr" in key or key == "alias":
            continue
        if isinstance(value, (int, float)):
            return key
    return None


def locate_samples(model_dir: Path, task: str, stamp: str) -> Optional[Path]:
    sample_name = f"samples_{task}_{stamp}.jsonl"
    sample_path = model_dir / sample_name
    if sample_path.exists():
        return sample_path
    candidates = sorted(model_dir.glob(f"samples_{task}_*.jsonl"))
    if candidates:
        return candidates[-1]
    return None


def load_sample_rows(samples_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"raw": line})
    return rows


def _stats_from_values(
    values: Iterable[float],
) -> Tuple[Optional[float], Optional[float]]:
    numeric = [float(v) for v in values]
    if not numeric:
        return None, None
    mean = sum(numeric) / len(numeric)
    variance = sum((v - mean) ** 2 for v in numeric) / len(numeric)
    stddev = math.sqrt(variance)
    return mean, stddev


def compute_latest_stats(
    table: Dict[str, Dict[str, MetricEntry]],
    display_map: Optional[Dict[str, List[str]]],
    visible_tasks: Optional[set[str]],
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    averages: Dict[str, Optional[float]] = {}
    stddevs: Dict[str, Optional[float]] = {}
    for model, entries in table.items():
        allowed: Optional[set[str]] = None
        if display_map is not None:
            allowed_list = display_map.get(model, [])
            if allowed_list:
                allowed = set(allowed_list)
        values = [
            float(entry.value)
            for task, entry in entries.items()
            if entry
            and isinstance(entry.value, (int, float))
            and (allowed is None or task in allowed)
            and (visible_tasks is None or task in visible_tasks)
        ]
        mean, stddev = _stats_from_values(values)
        averages[model] = mean
        stddevs[model] = stddev
    return averages, stddevs


def compute_all_stats(
    table: Dict[str, Dict[str, List[MetricEntry]]],
    display_map: Optional[Dict[str, List[str]]],
    visible_tasks: Optional[set[str]],
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    averages: Dict[str, Optional[float]] = {}
    stddevs: Dict[str, Optional[float]] = {}
    for model, task_entries in table.items():
        values: List[float] = []
        allowed: Optional[set[str]] = None
        if display_map is not None:
            allowed_list = display_map.get(model, [])
            if allowed_list:
                allowed = set(allowed_list)
        for task, runs in task_entries.items():
            if allowed is not None and task not in allowed:
                continue
            if visible_tasks is not None and task not in visible_tasks:
                continue
            for run in runs:
                if isinstance(run.value, (int, float)):
                    values.append(float(run.value))
        mean, stddev = _stats_from_values(values)
        averages[model] = mean
        stddevs[model] = stddev
    return averages, stddevs


def build_overview_data(store: ResultStore, mode: str, tasks_mode: str) -> Tuple[
    Any,
    Dict[str, Optional[float]],
    Dict[str, Optional[float]],
    List[str],
]:
    visibility = get_visibility_store()
    hidden_tasks = visibility.hidden_tasks(store.root)
    hidden_models = visibility.hidden_models(store.root)

    display_tasks = store.display_tasks(tasks_mode)
    display_tasks = [task for task in display_tasks if task not in hidden_tasks]
    display_map = store.display_tasks_map if tasks_mode == "first" else None
    visible_tasks = set(display_tasks)
    if mode == "latest":
        table = {
            model: entries
            for model, entries in store.latest_only().items()
            if model not in hidden_models
        }
        averages, stddevs = compute_latest_stats(table, display_map, visible_tasks)
    else:
        table = {
            model: entries
            for model, entries in store.all_runs().items()
            if model not in hidden_models
        }
        averages, stddevs = compute_all_stats(table, display_map, visible_tasks)
    return table, averages, stddevs, display_tasks


def _format_decimal(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _format_entry_value(entry: Optional[MetricEntry]) -> str:
    if not entry or entry.value is None:
        return ""
    if isinstance(entry.value, float):
        return f"{entry.value:.6f}"
    return str(entry.value)


def _format_runs_value(runs: Iterable[MetricEntry]) -> str:
    parts: List[str] = []
    for run in runs:
        value = run.value
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{value:.6f}")
        else:
            parts.append(str(value))
    return "; ".join(parts)


def create_app(default_root: Path) -> Flask:
    base_dir = Path.cwd()
    default_root = (
        default_root if default_root.is_absolute() else base_dir / default_root
    ).resolve()
    available_roots = sorted(p.resolve() for p in base_dir.iterdir() if p.is_dir())
    if default_root not in available_roots:
        available_roots.append(default_root)
        available_roots.sort()

    set_display_name_store(DisplayNameStore(base_dir / DISPLAY_NAME_FILENAME))
    set_visibility_store(VisibilityStore(base_dir / VISIBILITY_FILENAME))

    app = Flask(__name__, template_folder="templates", static_folder="static")

    def resolve_root(raw: Optional[str]) -> Path:
        if raw:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = base_dir / raw
        else:
            candidate = default_root
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            abort(404, description="Selected root directory does not exist.")
        return resolved

    def get_store() -> ResultStore:
        if hasattr(g, "result_store"):
            return g.result_store
        root_param = request.args.get("root")
        store = ResultStore(resolve_root(root_param), base_dir=base_dir)
        g.result_store = store
        return store

    @app.context_processor
    def inject_shared() -> Dict[str, Any]:
        store = get_store()
        mode = request.args.get("mode", "latest")
        tasks_mode = request.args.get("tasks_mode", "first")
        return {
            "available_roots": available_roots,
            "selected_root_path": store.root,
            "selected_root": str(store.root),
            "active_mode": mode,
            "active_tasks_mode": tasks_mode,
        }

    @app.route("/")
    def home() -> Any:
        mode = request.args.get("mode", "latest")
        tasks_mode = request.args.get("tasks_mode", "first")
        return redirect(
            url_for(
                "overview",
                root=str(default_root),
                mode=mode,
                tasks_mode=tasks_mode,
            )
        )

    @app.route("/overview")
    def overview() -> Any:
        mode = request.args.get("mode", "latest")
        if mode not in {"latest", "all"}:
            mode = "latest"
        tasks_mode = request.args.get("tasks_mode", "first")
        if tasks_mode not in {"first", "all"}:
            tasks_mode = "first"
        store = get_store()
        table, averages, stddevs, display_tasks = build_overview_data(
            store, mode, tasks_mode
        )
        name_store = get_display_name_store()
        display_names = {
            task: name_store.get_task_display_name(task) for task in display_tasks
        }
        model_display_names = {
            model: name_store.get_model_display_name(model) for model in table.keys()
        }
        return render_template(
            "overview.html",
            mode=mode,
            tasks=display_tasks,
            display_names=display_names,
            table=table,
            averages=averages,
            stddevs=stddevs,
            tasks_mode=tasks_mode,
            active_mode=mode,
            active_tasks_mode=tasks_mode,
            model_display_names=model_display_names,
        )

    @app.route("/overview/export.csv")
    def overview_csv() -> Response:
        mode = request.args.get("mode", "latest")
        if mode not in {"latest", "all"}:
            mode = "latest"
        tasks_mode = request.args.get("tasks_mode", "first")
        if tasks_mode not in {"first", "all"}:
            tasks_mode = "first"
        store = get_store()
        table, averages, stddevs, display_tasks = build_overview_data(
            store, mode, tasks_mode
        )
        tasks = display_tasks
        name_store = get_display_name_store()
        task_headers = [name_store.get_task_display_name(task) for task in tasks]

        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(["Model", "Avg Score", "Std Dev", *task_headers])

        if mode == "latest":
            for model, task_entries in table.items():
                model_display = name_store.get_model_display_name(model)
                row = [
                    model_display,
                    _format_decimal(averages.get(model)),
                    _format_decimal(stddevs.get(model)),
                ]
                for task in tasks:
                    row.append(_format_entry_value(task_entries.get(task)))
                writer.writerow(row)
        else:
            for model, task_entries in table.items():
                model_display = name_store.get_model_display_name(model)
                row = [
                    model_display,
                    _format_decimal(averages.get(model)),
                    _format_decimal(stddevs.get(model)),
                ]
                for task in tasks:
                    runs = task_entries.get(task, [])
                    row.append(_format_runs_value(runs))
                writer.writerow(row)

        csv_data = buffer.getvalue()
        filename = f"overview_{mode}.csv"
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )

    @app.route("/model/<model_name>")
    def model_detail(model_name: str) -> Any:
        mode = request.args.get("mode", "latest")
        tasks_mode = request.args.get("tasks_mode", "first")
        store = get_store()
        entries = store.get_model(model_name)
        name_store = get_display_name_store()
        display_names = {
            task: name_store.get_task_display_name(task) for task in entries.keys()
        }
        model_display_name = name_store.get_model_display_name(model_name)
        return render_template(
            "model.html",
            mode=mode,
            model_name=model_name,
            model_display_name=model_display_name,
            entries=entries,
            display_names=display_names,
            tasks_mode=tasks_mode,
        )

    @app.route("/model/<model_name>/task/<task_name>")
    def task_detail(model_name: str, task_name: str) -> Any:
        stamp = request.args.get("stamp")
        tasks_mode = request.args.get("tasks_mode", "first")
        store = get_store()
        selection = store.get_entry(model_name, task_name, stamp)
        rows = []
        if selection.samples_path and selection.samples_path.exists():
            rows = load_sample_rows(selection.samples_path)
        name_store = get_display_name_store()
        task_display_name = name_store.get_task_display_name(task_name)
        model_display_name = name_store.get_model_display_name(model_name)
        return render_template(
            "task.html",
            model_name=model_name,
            task_name=task_name,
            task_display_name=task_display_name,
            model_display_name=model_display_name,
            entry=selection,
            rows=rows,
            mode=request.args.get("mode", "latest"),
            root=str(store.root),
            tasks_mode=tasks_mode,
        )

    @app.route("/display-names", methods=["GET", "POST"])
    def edit_display_names() -> Any:
        mode = request.args.get("mode", "latest")
        tasks_mode = request.args.get("tasks_mode", "first")
        store = get_store()
        name_store = get_display_name_store()
        task_keys = set(store.tasks) | set(name_store.task_keys())
        model_keys = set(store.results.keys()) | set(name_store.model_keys())
        tasks = sorted(task_keys)
        models = sorted(model_keys)
        if request.method == "POST":
            task_names = request.form.getlist("task_name")
            task_labels = request.form.getlist("task_display")
            for task, label in zip(task_names, task_labels):
                name_store.set_task_display_name(task, label)
            model_names = request.form.getlist("model_name")
            model_labels = request.form.getlist("model_display")
            for model, label in zip(model_names, model_labels):
                name_store.set_model_display_name(model, label)
            name_store.save()
            return redirect(
                url_for(
                    "edit_display_names",
                    root=str(store.root),
                    mode=mode,
                    tasks_mode=tasks_mode,
                    updated="1",
                )
            )
        task_overrides = name_store.task_overrides_copy()
        model_overrides = name_store.model_overrides_copy()
        task_effective = {
            task: name_store.get_task_display_name(task) for task in tasks
        }
        model_effective = {
            model: name_store.get_model_display_name(model) for model in models
        }
        updated = request.args.get("updated") == "1"
        return render_template(
            "display_names.html",
            mode=mode,
            tasks_mode=tasks_mode,
            tasks=tasks,
            models=models,
            task_overrides=task_overrides,
            model_overrides=model_overrides,
            task_defaults=DEFAULT_TASK_DISPLAY_NAMES,
            task_effective=task_effective,
            model_effective=model_effective,
            updated=updated,
        )

    @app.route("/models", methods=["GET", "POST"])
    def models_page() -> Any:
        mode = request.args.get("mode", "latest")
        tasks_mode = request.args.get("tasks_mode", "first")
        store = get_store()
        name_store = get_display_name_store()
        visibility = get_visibility_store()

        all_models = sorted(store.results.keys())
        if request.method == "POST":
            visible_models = set(request.form.getlist("model_visible"))
            hidden_models = {m for m in all_models if m not in visible_models}
            visibility.set_hidden_models(store.root, hidden_models)
            visibility.save()
            return redirect(
                url_for(
                    "models_page",
                    root=str(store.root),
                    mode=mode,
                    tasks_mode=tasks_mode,
                    updated="1",
                )
            )

        model_display_names = {
            model: name_store.get_model_display_name(model) for model in all_models
        }
        model_visible = {
            model: visibility.is_model_visible(store.root, model) for model in all_models
        }
        updated = request.args.get("updated") == "1"
        return render_template(
            "models.html",
            mode=mode,
            tasks_mode=tasks_mode,
            models=all_models,
            model_display_names=model_display_names,
            model_visible=model_visible,
            updated=updated,
        )

    @app.route("/benchmarks", methods=["GET", "POST"])
    def benchmarks_page() -> Any:
        mode = request.args.get("mode", "latest")
        tasks_mode = request.args.get("tasks_mode", "first")
        store = get_store()
        name_store = get_display_name_store()
        visibility = get_visibility_store()

        all_tasks = store.display_tasks(tasks_mode)
        if request.method == "POST":
            visible_tasks = set(request.form.getlist("task_visible"))
            hidden_tasks = {t for t in all_tasks if t not in visible_tasks}
            visibility.set_hidden_tasks(store.root, hidden_tasks)
            visibility.save()
            return redirect(
                url_for(
                    "benchmarks_page",
                    root=str(store.root),
                    mode=mode,
                    tasks_mode=tasks_mode,
                    updated="1",
                )
            )

        task_display_names = {
            task: name_store.get_task_display_name(task) for task in all_tasks
        }
        task_visible = {
            task: visibility.is_task_visible(store.root, task) for task in all_tasks
        }
        updated = request.args.get("updated") == "1"
        return render_template(
            "benchmarks.html",
            mode=mode,
            tasks_mode=tasks_mode,
            tasks=all_tasks,
            task_display_names=task_display_names,
            task_visible=task_visible,
            updated=updated,
        )

    return app


# Top-level entries the static export generates. Re-export removes only these
# (never the whole output directory), preserving VCS metadata and user files.
_GENERATED_EXPORT_ENTRIES: Tuple[str, ...] = (
    "index.html",
    "overview.html",
    "models.html",
    "benchmarks.html",
    "display-names.html",
    "overview",
    "model",
    "static",
    EXPORT_MARKER,
)


def _safe_path_segment(value: str) -> str:
    encoded = quote(value, safe="")
    if encoded in {"", ".", ".."}:
        return "_"
    return encoded


def _model_static_path(model: str) -> Path:
    return Path("model") / f"{_safe_path_segment(model)}.html"


def _task_static_path(model: str, task: str) -> Path:
    return (
        Path("model")
        / _safe_path_segment(model)
        / "task"
        / f"{_safe_path_segment(task)}.html"
    )


def _task_stamp_static_path(model: str, task: str, stamp: str) -> Path:
    return (
        Path("model")
        / _safe_path_segment(model)
        / "task"
        / _safe_path_segment(task)
        / f"{_safe_path_segment(stamp)}.html"
    )


def _route_path_for_model(model: str) -> str:
    return f"/model/{quote(model, safe='')}"


def _route_path_for_task(model: str, task: str) -> str:
    return f"{_route_path_for_model(model)}/task/{quote(task, safe='')}"


def _export_routes(
    store: ResultStore, visibility: VisibilityStore, root: Path
) -> List[StaticExportRoute]:
    routes = [
        StaticExportRoute("/overview", Path("index.html")),
        StaticExportRoute("/overview", Path("overview.html")),
        StaticExportRoute("/overview/export.csv", Path("overview") / "export.csv"),
        StaticExportRoute("/models", Path("models.html")),
        StaticExportRoute("/benchmarks", Path("benchmarks.html")),
        StaticExportRoute("/display-names", Path("display-names.html")),
    ]
    # Skip models/tasks the user hid with the visibility controls; the overview
    # and CSV already exclude them, so a published site must not expose their
    # detail pages either.
    hidden_models = visibility.hidden_models(root)
    hidden_tasks = visibility.hidden_tasks(root)
    for model in sorted(store.results.keys()):
        if model in hidden_models:
            continue
        routes.append(
            StaticExportRoute(_route_path_for_model(model), _model_static_path(model))
        )
        for task in sorted(store.results[model].keys()):
            if task in hidden_tasks:
                continue
            routes.append(
                StaticExportRoute(
                    _route_path_for_task(model, task),
                    _task_static_path(model, task),
                )
            )
            for run in store.results[model][task]:
                routes.append(
                    StaticExportRoute(
                        _route_path_for_task(model, task),
                        _task_stamp_static_path(model, task, run.timestamp_raw),
                        {"stamp": run.timestamp_raw},
                    )
                )
    return routes


def _export_request_path(
    route_path: str, root: Path, extra_query: Optional[Dict[str, str]] = None
) -> str:
    query_values = {
        "root": str(root),
        "mode": "latest",
        "tasks_mode": "first",
    }
    if extra_query:
        query_values.update(extra_query)
    query = urlencode(query_values)
    return f"{route_path}?{query}"


def _static_path_for_url(url: str) -> Optional[Path]:
    if not url or url.startswith(("#", "mailto:", "javascript:")):
        return None

    parsed = urlsplit(html_unescape(url))
    if parsed.scheme or parsed.netloc:
        return None

    path = parsed.path or "/"
    if path == "/":
        return Path("index.html")
    if path == "/overview":
        return Path("overview.html")
    if path == "/overview/export.csv":
        return Path("overview") / "export.csv"
    if path == "/models":
        return Path("models.html")
    if path == "/benchmarks":
        return Path("benchmarks.html")
    if path == "/display-names":
        return Path("display-names.html")
    if path.startswith("/static/"):
        asset = unquote(path.removeprefix("/static/"))
        parts = [part for part in asset.split("/") if part]
        if not parts or any(part in {".", ".."} for part in parts):
            return None
        return Path("static", *parts)

    parts = path.strip("/").split("/")
    if len(parts) == 2 and parts[0] == "model":
        return _model_static_path(unquote(parts[1]))
    if len(parts) == 4 and parts[0] == "model" and parts[2] == "task":
        query = parse_qs(parsed.query)
        stamp_values = query.get("stamp")
        model = unquote(parts[1])
        task = unquote(parts[3])
        if stamp_values:
            return _task_stamp_static_path(model, task, stamp_values[0])
        return _task_static_path(model, task)

    return None


def _relative_static_url(target: Path, current_output: Path) -> str:
    start = current_output.parent.as_posix()
    if start == ".":
        start = "."
    return posixpath.relpath(target.as_posix(), start)


def _markupsafe_escape(value: str) -> str:
    # Mirror how Jinja/MarkupSoup escapes text, which differs from Python's
    # html.escape for quotes ("&#39;"/"&#34;" vs "&#x27;"/"&quot;").
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("'", "&#39;")
        .replace('"', "&#34;")
    )


def _scrub_export_html(html: str, roots: Sequence[Tuple[str, str]]) -> str:
    # Neutralize absolute local paths that the templates render: the results
    # root and paths under it (result/samples files), the launch directory (the
    # overview renders sibling result dirs as <option value="/abs/path">), and
    # the home directory. Replace the raw string plus its escaped forms, since a
    # path such as /mnt/R&D or one containing quotes is rendered as HTML
    # entities (and Jinja and html.escape disagree on which entities to emit).
    scrubbed = html
    for raw, token in roots:
        if not raw:
            continue
        for variant in {raw, html_escape(raw), _markupsafe_escape(raw)}:
            scrubbed = scrubbed.replace(variant, token)
    return scrubbed


def _rewrite_html_links(
    html: str, current_output: Path, scrub_roots: Sequence[Tuple[str, str]]
) -> str:
    def replace(match: re.Match[str]) -> str:
        target = _static_path_for_url(match.group("url"))
        if target is None:
            return match.group(0)
        quote_char = match.group("quote")
        # Percent-encode the relative link so a browser decodes it back to the
        # exact on-disk filename. Export filenames are themselves percent-encoded
        # (see _safe_path_segment), so a raw href like "model%20a.html" would be
        # decoded to "model a.html" and 404. Encoding the "%" (safe="/") yields
        # "model%2520a.html", which decodes to the real "model%20a.html".
        relative = quote(_relative_static_url(target, current_output), safe="/")
        return f"{match.group('attr')}{quote_char}{relative}{quote_char}"

    rewritten = HTML_LINK_ATTR_PATTERN.sub(replace, html)
    return _scrub_export_html(rewritten, scrub_roots)


def _write_export_response(
    response: Response,
    out_path: Path,
    relative_output: Path,
    scrub_roots: Sequence[Tuple[str, str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if response.mimetype == "text/html":
        html = response.get_data(as_text=True)
        out_path.write_text(
            _rewrite_html_links(html, relative_output, scrub_roots),
            encoding="utf-8",
        )
        return
    out_path.write_bytes(response.get_data())


def export_static(root: Path, out_dir: Path) -> Path:
    base_dir = Path.cwd()
    resolved_root = (root if root.is_absolute() else base_dir / root).resolve()
    out_dir = out_dir.resolve()

    # Remove pages for models or tasks that were removed or renamed since a
    # previous export, without touching anything this tool did not generate.
    # We delete only the known generated entries (never the whole tree), so a
    # published directory keeps its VCS metadata and user files such as .git and
    # CNAME. On the first export into a non-empty directory that this tool did
    # not create (no marker), refuse so a user's own files are not overwritten.
    if out_dir.exists():
        marker = out_dir / EXPORT_MARKER
        if any(out_dir.iterdir()) and not marker.exists():
            raise RuntimeError(
                f"Refusing to export into non-empty directory {out_dir} that is "
                f"not a previous amalia export (no {EXPORT_MARKER} marker). Choose "
                "an empty or new directory."
            )
        for name in _GENERATED_EXPORT_ENTRIES:
            entry = out_dir / name
            if entry.is_dir():
                shutil.rmtree(entry)
            elif entry.exists():
                entry.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / EXPORT_MARKER).write_text(
        "AMALIA inspect static export directory.\n", encoding="utf-8"
    )

    app = create_app(resolved_root)
    store = ResultStore(resolved_root, base_dir=base_dir)
    routes = _export_routes(store, get_visibility_store(), resolved_root)

    # Paths the templates may render as absolute strings: the results root and
    # anything under it, the launch directory (sibling result dirs shown in the
    # overview root selector), and the home directory. Longest paths first so a
    # nested path is replaced before its parent prefix.
    scrub_roots = sorted(
        (
            (str(resolved_root), "[results]"),
            (str(base_dir), "[cwd]"),
            (str(Path.home()), "~"),
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )

    with app.test_client() as client:
        for route in routes:
            response = client.get(
                _export_request_path(route.route_path, resolved_root, route.query)
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Could not export {route.route_path}: HTTP {response.status_code}"
                )
            _write_export_response(
                response,
                out_dir / route.relative_output,
                route.relative_output,
                scrub_roots,
            )

    static_folder = Path(app.static_folder) if app.static_folder else None
    if static_folder and static_folder.exists():
        shutil.copytree(static_folder, out_dir / "static", dirs_exist_ok=True)

    index_path = out_dir / "index.html"
    print(f"Exported AMALIA Inspect static site to {out_dir}")
    print(f"Open {index_path}")
    return index_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the AMALIA inspection UI.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Result directory containing per-model folders.",
    )
    parser.add_argument(
        "--export-html",
        type=Path,
        metavar="DIR",
        help="Export the inspection UI as a static HTML site instead of serving it.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    values = list(argv) if argv is not None else None
    return parser.parse_args(values)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.export_html:
        export_static(args.root, args.export_html)
        return
    app = create_app(args.root)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
