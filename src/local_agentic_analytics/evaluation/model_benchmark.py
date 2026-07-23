"""Multi-model benchmark harness for apple-to-apple SLM comparison.

This harness benchmarks any Ollama model registered under ``configs/models/``
against the existing evaluation suites (gold SQL v2, ablation, finance)
without changing workflow behavior. The only variable allowed to differ
between runs is the model itself: the harness validates that every locked
variable (context window, temperature, prompt template) matches the shared
``configs/model.yaml`` and swaps models purely through the existing
``OLLAMA_MODEL`` environment mechanism that ``OllamaTool.from_config`` reads.

It reuses (never reimplements):
- ``evaluation.sql_gold_eval`` / ``evaluation.ablation_eval`` for the suites,
- ``evaluation.resource_monitor`` for peak RAM/VRAM and tokens/sec,
- ``evaluation.gpu_cpu_benchmark`` helpers for WDDM-safe VRAM attribution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import psutil
import requests
import yaml
from dotenv import load_dotenv

from local_agentic_analytics.core.config import PROJECT_ROOT, load_config
from local_agentic_analytics.evaluation.ablation_eval import (
    ABLATION_EVAL_COLUMNS,
    run_ablation_evaluation,
)
from local_agentic_analytics.evaluation.gpu_cpu_benchmark import (
    read_ollama_ps_vram,
)
from local_agentic_analytics.evaluation.resource_monitor import (
    ResourceSampler,
    aggregate_tokens_per_second,
    query_gpu_vram_mb,
)
from local_agentic_analytics.evaluation.sql_gold_eval import (
    SQL_GOLD_EVAL_COLUMNS,
    load_gold_questions,
    run_sql_gold_evaluation,
)
from local_agentic_analytics.evaluation.statistics import (
    bootstrap_rate_ci,
    majority_vote,
    mcnemar_vs_baseline,
)
from local_agentic_analytics.tools.ollama_tool import OllamaTool


MODEL_CONFIG_DIR = PROJECT_ROOT / "configs" / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "experiments" / "model_benchmark"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

SUITE_SQL_GOLD_V2 = "sql_gold_v2"
SUITE_ABLATION = "ablation"
SUITE_FINANCE = "finance"
SUITES = (SUITE_SQL_GOLD_V2, SUITE_ABLATION, SUITE_FINANCE)

# The workflow defaults to the energy domain; the finance suite must build
# the workflow with the finance profile or every question mis-routes to the
# energy table.
SUITE_DOMAINS: dict[str, str] = {
    SUITE_SQL_GOLD_V2: "energy",
    SUITE_ABLATION: "energy",
    SUITE_FINANCE: "finance",
}

SUITE_QUESTION_PATHS: dict[str, Path] = {
    SUITE_SQL_GOLD_V2: (
        PROJECT_ROOT / "references" / "sql_gold" / "energy_gold_questions_v2.json"
    ),
    SUITE_ABLATION: (
        PROJECT_ROOT / "references" / "sql_gold" / "energy_gold_questions_v2.json"
    ),
    # v2 (36 soal) menggantikan v1 (8 soal) sejak seluruh butirnya lolos
    # verifikasi eksekusi; lihat references/sql_gold/finance_gold_review_v2.md.
    SUITE_FINANCE: (
        PROJECT_ROOT / "references" / "sql_gold" / "finance_gold_questions_v2.json"
    ),
}

TELEMETRY_COLUMNS = ("latency_total", "tokens_per_second", "unit_correct")

REQUIRED_MODEL_FIELDS = (
    "key",
    "ollama_tag",
    "param_count",
    "quantization",
    "context_window",
    "temperature",
    "prompt_template",
    "source",
)
NULLABLE_MODEL_FIELDS = ("modelfile_sha256", "kaggle_notebook_url")
VALID_MODEL_SOURCES = ("base", "finetuned")

SEED_NOTE = (
    "temperature 0.0; no explicit Ollama seed (existing workflow behavior is "
    "left unchanged); GPU inference is not bit-deterministic, mitigated by "
    "repeats >= 3"
)

Logger = Callable[[str], None]
WorkflowBuilder = Callable[..., Any]


def _default_workflow_builder(**kwargs: Any) -> Any:
    # Imported lazily so config/hash/dry-run helpers work without the full
    # workflow dependency chain (e.g. a live DuckDB database).
    from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow

    return SequentialAnalyticsWorkflow(**kwargs)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelBenchmarkConfig:
    """One benchmark candidate registered in ``configs/models/``."""

    key: str
    ollama_tag: str
    param_count: str
    quantization: str
    context_window: int
    temperature: float
    prompt_template: str
    source: str
    modelfile_sha256: str | None = None
    kaggle_notebook_url: str | None = None
    cpu_only: bool = False


def list_model_keys(config_dir: str | Path = MODEL_CONFIG_DIR) -> list[str]:
    """List registered model keys (YAML file stems) in sorted order."""
    directory = Path(config_dir)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def load_model_benchmark_config(
    key: str,
    config_dir: str | Path = MODEL_CONFIG_DIR,
) -> ModelBenchmarkConfig:
    """Load and validate one model registry YAML."""
    path = Path(config_dir) / f"{key}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found: {path}")

    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in model config: {path}") from exc

    if not isinstance(content, dict) or not isinstance(content.get("model"), dict):
        raise ValueError(f"Model config must contain a 'model' mapping: {path}")

    raw = content["model"]
    for field_name in REQUIRED_MODEL_FIELDS:
        value = raw.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(
                f"Model config {path} is missing required field '{field_name}'"
            )

    if str(raw["key"]) != key:
        raise ValueError(
            f"Model config key '{raw['key']}' does not match file name '{key}'"
        )
    if str(raw["source"]) not in VALID_MODEL_SOURCES:
        raise ValueError(
            f"Model config {path} has invalid source '{raw['source']}' "
            f"(expected one of {VALID_MODEL_SOURCES})"
        )

    return ModelBenchmarkConfig(
        key=str(raw["key"]),
        ollama_tag=str(raw["ollama_tag"]),
        param_count=str(raw["param_count"]),
        quantization=str(raw["quantization"]),
        context_window=int(raw["context_window"]),
        temperature=float(raw["temperature"]),
        prompt_template=str(raw["prompt_template"]),
        source=str(raw["source"]),
        modelfile_sha256=_optional_str(raw.get("modelfile_sha256")),
        kaggle_notebook_url=_optional_str(raw.get("kaggle_notebook_url")),
        cpu_only=bool(raw.get("cpu_only", False)),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_locked_variables(
    config: ModelBenchmarkConfig,
    shared_config_path: str = "model.yaml",
) -> list[str]:
    """Check the variables that must stay identical across all candidates.

    Returns a list of human-readable problems; an empty list means the config
    is apple-to-apple compatible with the shared ``configs/model.yaml``.
    """
    problems: list[str] = []

    shared = load_config(shared_config_path).get("model", {})
    if not isinstance(shared, dict):
        return [f"shared config '{shared_config_path}' has no 'model' mapping"]

    shared_context = shared.get("context_window")
    if isinstance(shared_context, int) and shared_context != config.context_window:
        problems.append(
            f"context_window {config.context_window} differs from shared "
            f"configs/model.yaml ({shared_context})"
        )

    shared_temperature = shared.get("temperature")
    if (
        isinstance(shared_temperature, (int, float))
        and float(shared_temperature) != config.temperature
    ):
        problems.append(
            f"temperature {config.temperature} differs from shared "
            f"configs/model.yaml ({shared_temperature})"
        )

    if config.prompt_template != "default":
        problems.append(
            "prompt_template must be 'default' (identical workflow prompts) "
            f"but is '{config.prompt_template}'"
        )

    if config.source == "finetuned" and not config.modelfile_sha256:
        problems.append("finetuned model must record modelfile_sha256")

    return problems


# ---------------------------------------------------------------------------
# Telemetry wrapper (adds latency/unit columns without touching eval modules)
# ---------------------------------------------------------------------------


class TelemetryRecorder:
    """Records per-question telemetry from wrapped workflow runs."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def wrap(self, workflow: Any) -> "_TelemetryWorkflow":
        return _TelemetryWorkflow(workflow, self.records)


class _TelemetryWorkflow:
    """Delegating wrapper that captures latency and throughput per question."""

    def __init__(self, inner: Any, records: list[dict[str, Any]]):
        self._inner = inner
        self._records = records

    def run(self, user_query: str) -> Any:
        try:
            state = self._inner.run(user_query)
        except Exception:
            self._records.append(
                {
                    "question": user_query,
                    "latency_total": None,
                    "tokens_per_second": None,
                    "final_answer": "",
                    "success": False,
                }
            )
            raise

        latency = getattr(state, "latency", {}) or {}
        tool_calls = getattr(state, "tool_calls", []) or []
        self._records.append(
            {
                "question": user_query,
                "latency_total": latency.get("total"),
                "tokens_per_second": aggregate_tokens_per_second(tool_calls),
                "final_answer": getattr(state, "final_answer", None) or "",
                "success": bool(getattr(state, "success", False)),
            }
        )
        return state


def unit_mentioned(expected_unit: str, final_answer: str) -> bool | None:
    """Heuristic unit correctness: the expected unit appears as a standalone
    token in the final answer (case-insensitive, so 'kW' never matches 'kWh').

    Returns ``None`` when either side is empty, meaning "not assessable".
    """
    if not expected_unit or not final_answer:
        return None
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(expected_unit)}(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    return bool(pattern.search(final_answer))


def attach_telemetry(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge telemetry records into suite rows by call order.

    Rows and telemetry records are produced in the same order (the eval
    modules call ``workflow.run`` once per row); rows whose workflow call never
    happened (e.g. gold-file load errors) simply get empty telemetry columns.
    """
    expected_units = {
        str(question.get("question", "")): str(question.get("expected_unit", ""))
        for question in questions
    }

    merged_rows: list[dict[str, Any]] = []
    record_index = 0
    for row in rows:
        merged = dict(row)
        record = records[record_index] if record_index < len(records) else None
        if record is not None and record["question"] == row.get("question"):
            record_index += 1
            latency_total = record.get("latency_total")
            tokens_per_second = record.get("tokens_per_second")
            unit = unit_mentioned(
                expected_units.get(str(row.get("question", "")), ""),
                str(record.get("final_answer", "")),
            )
            merged["latency_total"] = "" if latency_total is None else latency_total
            merged["tokens_per_second"] = (
                "" if tokens_per_second is None else tokens_per_second
            )
            merged["unit_correct"] = "" if unit is None else unit
        else:
            merged["latency_total"] = ""
            merged["tokens_per_second"] = ""
            merged["unit_correct"] = ""
        merged_rows.append(merged)

    return merged_rows


# ---------------------------------------------------------------------------
# Ollama / environment helpers
# ---------------------------------------------------------------------------


def resolve_ollama_base_url() -> str:
    """Resolve the Ollama base URL via the same dotenv chain as the workflow."""
    load_dotenv(PROJECT_ROOT / ".env.example", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return (
        os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
        or DEFAULT_OLLAMA_BASE_URL
    )


def list_local_ollama_tags(base_url: str) -> list[str] | None:
    """List locally available model tags; ``None`` when Ollama is unreachable."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    models = data.get("models")
    if not isinstance(models, list):
        return []
    return [str(item.get("name", "")) for item in models if isinstance(item, dict)]


def is_tag_available(tag: str, local_tags: list[str]) -> bool:
    """Check tag availability, treating ``name`` and ``name:latest`` as equal."""
    normalized = tag if ":" in tag else f"{tag}:latest"
    for local_tag in local_tags:
        candidate = local_tag if ":" in local_tag else f"{local_tag}:latest"
        if candidate == normalized:
            return True
    return False


def pull_ollama_tag(tag: str, logger: Logger = print) -> bool:
    """Pull a model tag via the Ollama CLI. Returns ``True`` on success."""
    logger(f"Pulling Ollama model '{tag}' (this may take a while)...")
    try:
        result = subprocess.run(
            ["ollama", "pull", tag],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger("ERROR: 'ollama' executable not found on PATH.")
        return False

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        logger(f"ERROR: ollama pull '{tag}' failed: {stderr[:500]}")
        return False
    return True


def get_ollama_version(base_url: str) -> str | None:
    """Read the Ollama server version; ``None`` when unreachable."""
    try:
        response = requests.get(f"{base_url}/api/version", timeout=5)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    version = data.get("version")
    return str(version) if version else None


def get_ollama_model_details(base_url: str, tag: str) -> dict[str, Any]:
    """Best-effort ``/api/show`` details (actual quantization, parameters)."""
    try:
        response = requests.post(
            f"{base_url}/api/show",
            json={"model": tag},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return {}

    details = data.get("details")
    return details if isinstance(details, dict) else {}


def collect_hardware_info() -> dict[str, Any]:
    """Snapshot the hardware the benchmark ran on (for the manifest)."""
    uname = platform.uname()
    info: dict[str, Any] = {
        "platform": f"{uname.system} {uname.release}",
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": _query_gpu_name(),
    }
    return info


def _query_gpu_name() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first_line = result.stdout.strip().splitlines()
    return first_line[0].strip() if first_line else None


def git_commit_hash() -> str | None:
    """Current repo commit hash; ``None`` outside a usable git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


# ---------------------------------------------------------------------------
# Dataset hashing
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """SHA256 of one file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_gold_dataset(
    questions_path: str | Path,
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Hash the questions JSON plus every referenced gold SQL file."""
    questions_path = Path(questions_path)
    gold_files = sorted(
        {str(question.get("gold_sql_file", "")) for question in questions}
    )
    combined = hashlib.sha256()
    missing: list[str] = []
    for gold_file in gold_files:
        if not gold_file:
            continue
        resolved = Path(gold_file)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        if not resolved.is_file():
            missing.append(gold_file)
            continue
        combined.update(gold_file.encode("utf-8"))
        combined.update(sha256_file(resolved).encode("utf-8"))

    return {
        "questions_path": _relative_to_project(questions_path),
        "questions_sha256": sha256_file(questions_path),
        "gold_sql_combined_sha256": combined.hexdigest(),
        "n_questions": len(questions),
        "n_gold_sql_files": len([name for name in gold_files if name]),
        "missing_gold_sql_files": missing,
    }


def _relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile (pct in [0, 100]); ``None`` when empty."""
    if not values:
        return None
    if not 0 <= pct <= 100:
        raise ValueError("pct must be between 0 and 100")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (pct / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Mean and sample std dev (std is 0.0 for a single value)."""
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# Suite execution
# ---------------------------------------------------------------------------


def suite_columns(suite: str) -> tuple[str, ...]:
    """CSV columns for one suite: original eval columns + telemetry columns."""
    base = ABLATION_EVAL_COLUMNS if suite == SUITE_ABLATION else SQL_GOLD_EVAL_COLUMNS
    return tuple(base) + TELEMETRY_COLUMNS


def load_suite_questions(
    suite: str,
    questions_path: str | Path | None = None,
    limit: int | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load the gold questions for one suite (path override for new sets)."""
    if suite not in SUITES:
        raise ValueError(f"Unknown suite '{suite}' (expected one of {SUITES})")
    path = Path(questions_path) if questions_path else SUITE_QUESTION_PATHS[suite]
    questions = load_gold_questions(path)
    if limit is not None:
        questions = questions[:limit]
    return path, questions


def run_suite_once(
    suite: str,
    questions: list[dict[str, Any]],
    recorder: TelemetryRecorder,
    workflow_builder: WorkflowBuilder = _default_workflow_builder,
) -> list[dict[str, Any]]:
    """Run one full pass of a suite with telemetry-wrapped workflows."""
    domain = SUITE_DOMAINS.get(suite, "energy")
    if suite == SUITE_ABLATION:
        def factory(toggles: Any) -> Any:
            return recorder.wrap(workflow_builder(toggles=toggles, domain=domain))

        return run_ablation_evaluation(questions, workflow_factory=factory)

    workflow = recorder.wrap(workflow_builder(domain=domain))
    return run_sql_gold_evaluation(questions, workflow=workflow)


def write_benchmark_rows(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    output_path: str | Path,
) -> Path:
    """Write suite rows (with telemetry columns) to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def _warm_up(workflow: Any, question_text: str, logger: Logger) -> None:
    try:
        workflow.run(question_text)
    except Exception as exc:  # Warm-up failures must not abort the benchmark.
        logger(f"Warm-up run failed (ignored for metrics): {exc}")


def _unload_model(tag: str) -> None:
    try:
        OllamaTool.from_config().unload(tag)
    except Exception:
        return


def run_model_suite(
    model_config: ModelBenchmarkConfig,
    suite: str,
    questions: list[dict[str, Any]],
    repeats: int,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    logger: Logger = print,
    workflow_builder: WorkflowBuilder = _default_workflow_builder,
    sampler_factory: Callable[[], ResourceSampler] | None = None,
    warm_up: bool = True,
) -> dict[str, Any]:
    """Run one suite ``repeats`` times for one model and persist all outputs.

    The model is selected exclusively through the ``OLLAMA_MODEL`` environment
    variable (the existing config mechanism); no workflow code path changes.
    """
    if repeats < 1:
        raise ValueError("repeats must be greater than 0")

    os.environ["OLLAMA_MODEL"] = model_config.ollama_tag
    if model_config.cpu_only:
        logger(
            f"NOTE: '{model_config.key}' is marked cpu_only; expect CPU "
            "inference latency."
        )

    model_dir = Path(output_dir) / model_config.key
    model_dir.mkdir(parents=True, exist_ok=True)
    columns = suite_columns(suite)
    sampler_factory = sampler_factory or (
        lambda: ResourceSampler(vram_reader=query_gpu_vram_mb)
    )
    warmup_text = next(
        (
            str(question.get("question", "")).strip()
            for question in questions
            if str(question.get("question", "")).strip()
        ),
        "",
    )

    run_entries: list[dict[str, Any]] = []
    for run_index in range(1, repeats + 1):
        logger(
            f"[{model_config.key}] suite={suite} run {run_index}/{repeats} "
            f"({len(questions)} questions)"
        )
        _unload_model(model_config.ollama_tag)
        if warm_up and warmup_text:
            _warm_up(
                workflow_builder(domain=SUITE_DOMAINS.get(suite, "energy")),
                warmup_text,
                logger,
            )

        recorder = TelemetryRecorder()
        sampler = sampler_factory()
        sampler.start()
        timestamp = _current_timestamp()
        started = perf_counter()
        try:
            rows = run_suite_once(
                suite,
                questions,
                recorder,
                workflow_builder=workflow_builder,
            )
        finally:
            usage = sampler.stop()
        duration_seconds = perf_counter() - started

        ps_vram = read_ollama_ps_vram(model_config.ollama_tag)
        merged_rows = attach_telemetry(rows, recorder.records, questions)

        csv_name = f"{suite}_run{run_index}.csv"
        write_benchmark_rows(merged_rows, columns, model_dir / csv_name)

        latencies = [
            float(row["latency_total"])
            for row in merged_rows
            if row.get("latency_total") not in ("", None)
        ]
        run_entries.append(
            {
                "run_index": run_index,
                "csv": csv_name,
                "timestamp": timestamp,
                "duration_seconds": round(duration_seconds, 3),
                "latency_seconds": {
                    "n": len(latencies),
                    "p50": percentile(latencies, 50),
                    "p95": percentile(latencies, 95),
                    "mean": statistics.fmean(latencies) if latencies else None,
                },
                "resources": {
                    "peak_rss_mb": usage.peak_rss_mb,
                    "peak_global_vram_mb": usage.peak_vram_mb,
                    "vram_available": usage.vram_available,
                    "model_vram_mb": ps_vram.vram_mb if ps_vram.found else None,
                    "processor_split": ps_vram.processor,
                },
            }
        )

    return {
        "repeats": repeats,
        "seed": None,
        "seed_note": SEED_NOTE,
        "runs": run_entries,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_environment_info(base_url: str, ollama_tag: str) -> dict[str, Any]:
    """Environment metadata recorded once per manifest update."""
    return {
        "commit": git_commit_hash(),
        "python_version": platform.python_version(),
        "ollama_version": get_ollama_version(base_url),
        "ollama_model_details": get_ollama_model_details(base_url, ollama_tag),
        "hardware": collect_hardware_info(),
        "updated_at": _current_timestamp(),
    }


def write_manifest(
    model_dir: str | Path,
    model_config: ModelBenchmarkConfig,
    suite: str,
    suite_entry: dict[str, Any],
    dataset_info: dict[str, Any],
    environment: dict[str, Any],
) -> Path:
    """Write/merge ``manifest.json`` for one model directory.

    Existing entries for other suites are preserved so a model accumulates one
    manifest across suites; rerunning a suite replaces that suite's entry.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "manifest.json"

    manifest: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                manifest = existing
        except (json.JSONDecodeError, OSError):
            manifest = {}

    suites = manifest.get("suites")
    if not isinstance(suites, dict):
        suites = {}
    suites[suite] = {"dataset": dataset_info, **suite_entry}

    manifest.update(
        {
            "model_key": model_config.key,
            "model_config": asdict(model_config),
            "environment": environment,
            "suites": suites,
        }
    )
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def build_dry_run_report(
    model_keys: list[str],
    suite: str,
    repeats: int,
    questions_path: str | Path | None = None,
    limit: int | None = None,
    config_dir: str | Path = MODEL_CONFIG_DIR,
) -> dict[str, Any]:
    """Validate everything a real run needs, without touching Ollama models.

    Checks per model: registry config loads, locked variables match the shared
    config, the gold dataset resolves and hashes, and (best-effort) whether the
    tag is already available locally.
    """
    base_url = resolve_ollama_base_url()
    local_tags = list_local_ollama_tags(base_url)

    try:
        path, questions = load_suite_questions(suite, questions_path, limit)
        dataset_info = hash_gold_dataset(path, questions)
        dataset_error = ""
    except Exception as exc:
        dataset_info = {}
        dataset_error = str(exc)

    models_report: list[dict[str, Any]] = []
    for key in model_keys:
        entry: dict[str, Any] = {"model_key": key}
        try:
            config = load_model_benchmark_config(key, config_dir=config_dir)
        except Exception as exc:
            entry["config_ok"] = False
            entry["error"] = str(exc)
            models_report.append(entry)
            continue

        entry["config_ok"] = True
        entry["ollama_tag"] = config.ollama_tag
        entry["locked_variable_problems"] = validate_locked_variables(config)
        if local_tags is None:
            entry["tag_available_locally"] = None
        else:
            entry["tag_available_locally"] = is_tag_available(
                config.ollama_tag, local_tags
            )
        models_report.append(entry)

    return {
        "suite": suite,
        "repeats": repeats,
        "ollama_base_url": base_url,
        "ollama_reachable": local_tags is not None,
        "ollama_version": get_ollama_version(base_url),
        "dataset": dataset_info,
        "dataset_error": dataset_error,
        "models": models_report,
        "environment": {
            "commit": git_commit_hash(),
            "hardware": collect_hardware_info(),
        },
    }


# ---------------------------------------------------------------------------
# Summarization (used by scripts/summarize_model_benchmark.py)
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = (
    "model",
    "suite",
    "config",
    "runs",
    "n_questions",
    "execution_success_mean",
    "execution_success_std",
    "numeric_match_compared_mean",
    "numeric_match_compared_std",
    "numeric_match_total_mean",
    "numeric_match_total_std",
    "unit_correct_mean",
    "unit_correct_std",
    "full_accuracy_mean",
    "full_accuracy_std",
    "latency_p50_mean",
    "latency_p50_std",
    "latency_p95_mean",
    "latency_p95_std",
    "tokens_per_second_mean",
    "tokens_per_second_std",
    "peak_rss_mb_max",
    "model_vram_mb_max",
    "execution_success_ci_low",
    "execution_success_ci_high",
    "full_accuracy_ci_low",
    "full_accuracy_ci_high",
    "baseline_model",
    "p_exec_vs_baseline",
    "p_full_vs_baseline",
)


def summarize_run_rows(
    rows: list[dict[str, str]],
    suite: str,
) -> dict[str, dict[str, float | int | None]]:
    """Per-run metrics grouped by ablation config ('-' for non-ablation)."""
    success_field = (
        "execution_success" if suite == SUITE_ABLATION else "agent_success"
    )

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if suite == SUITE_ABLATION:
            group = str(row.get("config", "")) or "-"
        else:
            group = "-"
        groups.setdefault(group, []).append(row)

    metrics: dict[str, dict[str, float | int | None]] = {}
    for group, group_rows in groups.items():
        total = len(group_rows)
        successes = sum(
            1 for row in group_rows if row.get(success_field) == "True"
        )
        compared = [row for row in group_rows if row.get("numeric_match") != ""]
        matched = sum(1 for row in compared if row.get("numeric_match") == "True")
        unit_rows = [row for row in group_rows if row.get("unit_correct") != ""]
        unit_correct = sum(
            1 for row in unit_rows if row.get("unit_correct") == "True"
        )
        latencies = [
            float(row["latency_total"])
            for row in group_rows
            if row.get("latency_total") not in ("", None)
        ]
        throughputs = [
            float(row["tokens_per_second"])
            for row in group_rows
            if row.get("tokens_per_second") not in ("", None)
        ]

        # Full accuracy: legacy scalar match OR the Fase 2 row-set match.
        # CSVs written before Fase 2 lack result_match_full and fall back to
        # the legacy metric alone.
        full_matches = sum(
            1
            for row in group_rows
            if row.get("numeric_match") == "True"
            or row.get("result_match_full") == "True"
        )

        metrics[group] = {
            "n_questions": total,
            "execution_success_rate": successes / total if total else None,
            "numeric_match_compared_rate": (
                matched / len(compared) if compared else None
            ),
            "numeric_match_total_rate": matched / total if total else None,
            "full_accuracy_rate": full_matches / total if total else None,
            "unit_correct_rate": (
                unit_correct / len(unit_rows) if unit_rows else None
            ),
            "latency_p50": percentile(latencies, 50),
            "latency_p95": percentile(latencies, 95),
            "tokens_per_second_mean": (
                statistics.fmean(throughputs) if throughputs else None
            ),
        }
    return metrics


def extract_question_outcomes(
    rows: list[dict[str, str]],
    suite: str,
) -> dict[str, dict[str, dict[str, bool]]]:
    """Per-question binary outcomes per group, for paired statistics.

    Returns ``{group: {question_id: {"exec": bool, "full": bool}}}`` where
    ``full`` is the combined accuracy (legacy scalar OR row-set match).
    """
    success_field = (
        "execution_success" if suite == SUITE_ABLATION else "agent_success"
    )
    outcomes: dict[str, dict[str, dict[str, bool]]] = {}
    for row in rows:
        if suite == SUITE_ABLATION:
            group = str(row.get("config", "")) or "-"
        else:
            group = "-"
        question_id = str(row.get("question_id", ""))
        if not question_id:
            continue
        outcomes.setdefault(group, {})[question_id] = {
            "exec": row.get(success_field) == "True",
            "full": (
                row.get("numeric_match") == "True"
                or row.get("result_match_full") == "True"
            ),
        }
    return outcomes


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a benchmark CSV back into string rows.

    Stored result columns (``agent_result``/``gold_result``) can hold very
    large JSON payloads when an agent SQL returns unaggregated rows, so the
    csv field limit is raised beyond its 128 KiB default.
    """
    _ensure_large_csv_fields()
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _ensure_large_csv_fields() -> None:
    # On Windows csv.field_size_limit(sys.maxsize) overflows a C long, so walk
    # down from 2**31 - 1 until a value is accepted.
    limit = 2**31 - 1
    while limit > 131072:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def summarize_model_benchmark(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    baseline_model: str = "gemma2_2b",
) -> list[dict[str, Any]]:
    """Aggregate every model/suite under ``output_dir`` into summary rows.

    Per-run metrics are aggregated as mean +/- sample std across repeats;
    resource peaks are reported as the max across runs. Rates additionally get
    bootstrap 95% CIs (over per-question majority-vote outcomes), and every
    non-baseline model gets exact McNemar p-values against ``baseline_model``
    paired on the same questions within the same suite/config.
    """
    output_dir = Path(output_dir)
    summary_rows: list[dict[str, Any]] = []
    # (suite, group) -> model -> {"exec": {qid: bool}, "full": {qid: bool}}
    outcome_registry: dict[tuple[str, str], dict[str, dict[str, dict[str, bool]]]] = {}

    for manifest_path in sorted(output_dir.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(manifest, dict):
            continue

        model_key = str(manifest.get("model_key", manifest_path.parent.name))
        suites = manifest.get("suites")
        if not isinstance(suites, dict):
            continue

        for suite_name, suite_entry in suites.items():
            if not isinstance(suite_entry, dict):
                continue
            runs = suite_entry.get("runs")
            if not isinstance(runs, list) or not runs:
                continue

            per_run_metrics: list[dict[str, dict[str, float | int | None]]] = []
            per_run_outcomes: list[dict[str, dict[str, dict[str, bool]]]] = []
            rss_values: list[float] = []
            vram_values: list[float] = []
            for run in runs:
                if not isinstance(run, dict):
                    continue
                csv_path = manifest_path.parent / str(run.get("csv", ""))
                if not csv_path.is_file():
                    continue
                rows = read_csv_rows(csv_path)
                per_run_metrics.append(
                    summarize_run_rows(rows, str(suite_name))
                )
                per_run_outcomes.append(
                    extract_question_outcomes(rows, str(suite_name))
                )
                resources = run.get("resources")
                if isinstance(resources, dict):
                    rss = resources.get("peak_rss_mb")
                    if isinstance(rss, (int, float)):
                        rss_values.append(float(rss))
                    vram = resources.get("model_vram_mb")
                    if isinstance(vram, (int, float)):
                        vram_values.append(float(vram))

            if not per_run_metrics:
                continue

            groups: list[str] = []
            for run_metrics in per_run_metrics:
                for group in run_metrics:
                    if group not in groups:
                        groups.append(group)

            for group in groups:
                group_runs = [
                    run_metrics[group]
                    for run_metrics in per_run_metrics
                    if group in run_metrics
                ]
                row = _build_summary_row(
                    model_key,
                    str(suite_name),
                    group,
                    group_runs,
                    rss_values,
                    vram_values,
                )

                # Majority-vote per-question outcomes across repeats feed the
                # bootstrap CIs and the paired McNemar comparisons.
                exec_majority = majority_vote(
                    [
                        {
                            qid: item["exec"]
                            for qid, item in run_outcomes.get(group, {}).items()
                        }
                        for run_outcomes in per_run_outcomes
                    ]
                )
                full_majority = majority_vote(
                    [
                        {
                            qid: item["full"]
                            for qid, item in run_outcomes.get(group, {}).items()
                        }
                        for run_outcomes in per_run_outcomes
                    ]
                )
                exec_ci = bootstrap_rate_ci(list(exec_majority.values()))
                full_ci = bootstrap_rate_ci(list(full_majority.values()))
                row["execution_success_ci_low"] = exec_ci["ci_low"]
                row["execution_success_ci_high"] = exec_ci["ci_high"]
                row["full_accuracy_ci_low"] = full_ci["ci_low"]
                row["full_accuracy_ci_high"] = full_ci["ci_high"]

                outcome_registry.setdefault(
                    (str(suite_name), group), {}
                )[model_key] = {"exec": exec_majority, "full": full_majority}
                summary_rows.append(row)

    _attach_mcnemar_p_values(summary_rows, outcome_registry, baseline_model)
    return summary_rows


def _attach_mcnemar_p_values(
    summary_rows: list[dict[str, Any]],
    outcome_registry: dict[tuple[str, str], dict[str, dict[str, dict[str, bool]]]],
    baseline_model: str,
) -> None:
    for row in summary_rows:
        key = (str(row.get("suite", "")), str(row.get("config", "")))
        models = outcome_registry.get(key, {})
        baseline = models.get(baseline_model)
        row["baseline_model"] = baseline_model
        if baseline is None or row.get("model") == baseline_model:
            row["p_exec_vs_baseline"] = None
            row["p_full_vs_baseline"] = None
            continue
        model_outcomes = models.get(str(row.get("model", "")))
        if model_outcomes is None:
            row["p_exec_vs_baseline"] = None
            row["p_full_vs_baseline"] = None
            continue
        row["p_exec_vs_baseline"] = mcnemar_vs_baseline(
            model_outcomes["exec"], baseline["exec"]
        )["p_value"]
        row["p_full_vs_baseline"] = mcnemar_vs_baseline(
            model_outcomes["full"], baseline["full"]
        )["p_value"]


def _build_summary_row(
    model_key: str,
    suite: str,
    group: str,
    group_runs: list[dict[str, float | int | None]],
    rss_values: list[float],
    vram_values: list[float],
) -> dict[str, Any]:
    def collect(metric: str) -> list[float]:
        return [
            float(run[metric])
            for run in group_runs
            if run.get(metric) is not None
        ]

    exec_mean, exec_std = mean_std(collect("execution_success_rate"))
    num_cmp_mean, num_cmp_std = mean_std(collect("numeric_match_compared_rate"))
    num_tot_mean, num_tot_std = mean_std(collect("numeric_match_total_rate"))
    unit_mean, unit_std = mean_std(collect("unit_correct_rate"))
    full_mean, full_std = mean_std(collect("full_accuracy_rate"))
    p50_mean, p50_std = mean_std(collect("latency_p50"))
    p95_mean, p95_std = mean_std(collect("latency_p95"))
    tps_mean, tps_std = mean_std(collect("tokens_per_second_mean"))

    n_questions = max(
        (int(run["n_questions"]) for run in group_runs if run.get("n_questions")),
        default=0,
    )

    return {
        "model": model_key,
        "suite": suite,
        "config": group,
        "runs": len(group_runs),
        "n_questions": n_questions,
        "execution_success_mean": exec_mean,
        "execution_success_std": exec_std,
        "numeric_match_compared_mean": num_cmp_mean,
        "numeric_match_compared_std": num_cmp_std,
        "numeric_match_total_mean": num_tot_mean,
        "numeric_match_total_std": num_tot_std,
        "unit_correct_mean": unit_mean,
        "unit_correct_std": unit_std,
        "full_accuracy_mean": full_mean,
        "full_accuracy_std": full_std,
        "latency_p50_mean": p50_mean,
        "latency_p50_std": p50_std,
        "latency_p95_mean": p95_mean,
        "latency_p95_std": p95_std,
        "tokens_per_second_mean": tps_mean,
        "tokens_per_second_std": tps_std,
        "peak_rss_mb_max": max(rss_values) if rss_values else None,
        "model_vram_mb_max": max(vram_values) if vram_values else None,
    }


def write_summary_csv(
    summary_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write the cross-model summary CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    column: _format_summary_value(row.get(column))
                    for column in SUMMARY_COLUMNS
                }
            )
    return path


def render_summary_markdown(summary_rows: list[dict[str, Any]]) -> str:
    """Render the summary as a markdown table with mean +/- std cells."""
    lines = [
        "# Model Benchmark Summary",
        "",
        f"Generated: {_current_timestamp()}",
        "",
        "Rates are mean ± sample std across repeats; latency in seconds.",
        "`numeric_match(cmp)` uses only numerically comparable questions "
        "(legacy metric); `full_accuracy` counts legacy scalar OR row-set "
        "matches over all questions, with a bootstrap 95% CI over "
        "majority-vote per-question outcomes.",
        "`p (exec/full)` are exact McNemar p-values paired against the "
        "baseline model on the same questions ('-' = baseline itself or "
        "baseline missing).",
        "",
        "| model | suite | config | runs | n | exec_success | "
        "numeric_match(cmp) | full_accuracy (95% CI) | unit_correct | "
        "latency p50 (s) | latency p95 (s) | tokens/s | peak RSS (MB) | "
        "model VRAM (MB) | p exec | p full |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {suite} | {config} | {runs} | {n} | {exec} | {ncmp} "
            "| {full} | {unit} | {p50} | {p95} | {tps} | {rss} | {vram} "
            "| {p_exec} | {p_full} |".format(
                model=row.get("model", ""),
                suite=row.get("suite", ""),
                config=row.get("config", ""),
                runs=row.get("runs", ""),
                n=row.get("n_questions", ""),
                exec=_format_pct_pair(
                    row.get("execution_success_mean"),
                    row.get("execution_success_std"),
                ),
                ncmp=_format_pct_pair(
                    row.get("numeric_match_compared_mean"),
                    row.get("numeric_match_compared_std"),
                ),
                full=_format_pct_with_ci(
                    row.get("full_accuracy_mean"),
                    row.get("full_accuracy_ci_low"),
                    row.get("full_accuracy_ci_high"),
                ),
                unit=_format_pct_pair(
                    row.get("unit_correct_mean"),
                    row.get("unit_correct_std"),
                ),
                p50=_format_float_pair(
                    row.get("latency_p50_mean"), row.get("latency_p50_std")
                ),
                p95=_format_float_pair(
                    row.get("latency_p95_mean"), row.get("latency_p95_std")
                ),
                tps=_format_float_pair(
                    row.get("tokens_per_second_mean"),
                    row.get("tokens_per_second_std"),
                ),
                rss=_format_number(row.get("peak_rss_mb_max")),
                vram=_format_number(row.get("model_vram_mb_max")),
                p_exec=_format_p_value(row.get("p_exec_vs_baseline")),
                p_full=_format_p_value(row.get("p_full_vs_baseline")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _format_pct_with_ci(mean: Any, ci_low: Any, ci_high: Any) -> str:
    if mean is None:
        return "-"
    text = f"{float(mean):.1%}"
    if ci_low is not None and ci_high is not None:
        text += f" [{float(ci_low):.1%}, {float(ci_high):.1%}]"
    return text


def _format_p_value(value: Any) -> str:
    if value is None:
        return "-"
    number = float(value)
    if number < 0.001:
        return "<0.001"
    return f"{number:.3f}"


def _format_summary_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _format_pct_pair(mean: Any, std: Any) -> str:
    if mean is None:
        return "-"
    if std is None:
        return f"{float(mean):.1%}"
    return f"{float(mean):.1%} ± {float(std):.1%}"


def _format_float_pair(mean: Any, std: Any) -> str:
    if mean is None:
        return "-"
    if std is None:
        return f"{float(mean):.2f}"
    return f"{float(mean):.2f} ± {float(std):.2f}"


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.0f}"


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
