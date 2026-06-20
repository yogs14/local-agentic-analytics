"""GPU vs CPU benchmark for the local quantized text-to-SQL model.

This measures the same fixed question set under a GPU-offloaded run and a
CPU-only run. ``num_gpu`` is overridden at runtime only; ``configs/model.yaml``
(CPU default) is never modified.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from local_agentic_analytics.agents.reporter_agent import ReporterAgent
from local_agentic_analytics.agents.repair_agent import SQLRepairAgent
from local_agentic_analytics.agents.sql_agent import SQLAgent
from local_agentic_analytics.core.config import PROJECT_ROOT
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.resource_monitor import (
    ResourceSampler,
    ResourceUsage,
    aggregate_tokens_per_second,
    query_gpu_vram_mb,
    query_ollama_vram_mb,
)
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow
from local_agentic_analytics.tools.ollama_tool import OllamaTool


DEFAULT_CSV_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "gpu_cpu_benchmark.csv"
)
DEFAULT_SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "gpu_cpu_benchmark_summary.json"
)

# A high layer count tells Ollama to offload every layer that fits in VRAM.
DEFAULT_GPU_NUM_GPU = 999
DEFAULT_WARMUP_QUESTION = "Berapa rata-rata konsumsi daya aktif?"

GPU_REQUESTED_BUT_RAN_ON_CPU = "gpu_requested_but_ran_on_cpu"

GPU_CPU_BENCHMARK_COLUMNS = (
    "run_index",
    "mode",
    "label",
    "question_id",
    "question",
    "success",
    "latency_total",
    "tokens_per_second",
    "oom",
    "error_message",
)

_OOM_SIGNATURES = ("out of memory", "cudamalloc", "cuda error", "cuda_error")


@dataclass(frozen=True)
class BenchmarkMode:
    """One benchmark mode and the ``num_gpu`` it requests at runtime."""

    name: str
    num_gpu: int
    requires_gpu: bool


@dataclass(frozen=True)
class GpuEngagement:
    """Result of checking whether the model is resident on the GPU."""

    on_gpu: bool
    partial_offload: bool
    processor: str
    detail: str = ""


@dataclass(frozen=True)
class OllamaPsVram:
    """Model-attributable VRAM derived from ``ollama ps`` (WDDM-safe).

    ``vram_mb`` is the loaded model's SIZE scaled by the fraction of layers on
    the GPU (from the PROCESSOR column), so it works on Windows GeForce/WDDM
    where per-process ``nvidia-smi`` memory is unavailable.
    """

    found: bool
    size_mb: float | None
    processor: str
    gpu_fraction: float
    vram_mb: float
    detail: str = ""


_SIZE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s+(TB|GB|MB|KB)\b",
    flags=re.IGNORECASE,
)
_SIZE_UNIT_TO_MB = {
    "KB": 1.0 / 1024.0,
    "MB": 1.0,
    "GB": 1024.0,
    "TB": 1024.0 * 1024.0,
}


@dataclass
class ModeResult:
    """Per-mode metadata captured alongside the per-question rows."""

    label: str
    usage: ResourceUsage
    oom_observed: bool
    engagement: GpuEngagement | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    ps_vram: OllamaPsVram | None = None
    baseline_vram_mb: float | None = None
    compute_apps_vram_mb: float | None = None


WorkflowFactory = Callable[[int], Any]
GpuDetector = Callable[[str], GpuEngagement]
SamplerFactory = Callable[[], ResourceSampler]
Logger = Callable[[str], None]
Unloader = Callable[[str], None]
PsRunner = Callable[[], str]
GlobalVramReader = Callable[[], "float | None"]


def default_modes(gpu_num_gpu: int = DEFAULT_GPU_NUM_GPU) -> tuple[BenchmarkMode, ...]:
    """Build the standard GPU/CPU mode pair."""
    return (
        BenchmarkMode(name="gpu", num_gpu=gpu_num_gpu, requires_gpu=True),
        BenchmarkMode(name="cpu", num_gpu=0, requires_gpu=False),
    )


def build_overridden_workflow(num_gpu: int) -> SequentialAnalyticsWorkflow:
    """Build a workflow whose Ollama calls use an overridden ``num_gpu``.

    The override happens on a freshly built :class:`OllamaTool`; the committed
    ``configs/model.yaml`` (CPU default) is left untouched.
    """
    ollama_tool = OllamaTool.from_config()
    ollama_tool.num_gpu = num_gpu
    return SequentialAnalyticsWorkflow(
        sql_agent=SQLAgent(ollama_tool=ollama_tool),
        repair_agent=SQLRepairAgent(ollama_tool=ollama_tool),
        reporter_agent=ReporterAgent(ollama_tool=ollama_tool),
    )


def _run_ollama_ps() -> str:
    import subprocess

    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def detect_gpu_engagement(
    model: str,
    ps_runner: Callable[[], str] | None = None,
) -> GpuEngagement:
    """Determine whether ``model`` is running on the GPU via ``ollama ps``."""
    ps_runner = ps_runner or _run_ollama_ps
    try:
        output = ps_runner()
    except FileNotFoundError:
        return GpuEngagement(
            on_gpu=False,
            partial_offload=False,
            processor="",
            detail="ollama executable not found",
        )
    except Exception as exc:
        return GpuEngagement(
            on_gpu=False,
            partial_offload=False,
            processor="",
            detail=f"ollama ps failed: {exc}",
        )

    return parse_ollama_ps(output, model)


def parse_ollama_ps(output: str, model: str) -> GpuEngagement:
    """Parse ``ollama ps`` output to read the model's PROCESSOR placement."""
    target = model.strip().lower()
    data_lines = [line for line in output.splitlines() if line.strip()]
    for line in data_lines:
        lowered = line.lower()
        if lowered.startswith("name") and "processor" in lowered:
            continue  # header row
        if target and target not in lowered:
            continue

        processor = _extract_processor(line)
        processor_lower = processor.lower()
        on_gpu = "gpu" in processor_lower
        has_cpu = "cpu" in processor_lower
        return GpuEngagement(
            on_gpu=on_gpu,
            partial_offload=on_gpu and has_cpu,
            processor=processor,
            detail="" if on_gpu else "processor reported no GPU",
        )

    return GpuEngagement(
        on_gpu=False,
        partial_offload=False,
        processor="",
        detail="model not found in ollama ps output",
    )


def _extract_processor(line: str) -> str:
    match = re.search(
        r"\d+%(?:/\d+%)?\s+(?:CPU/GPU|GPU/CPU|GPU|CPU)",
        line,
        flags=re.IGNORECASE,
    )
    return match.group(0).strip() if match else ""


def read_ollama_ps_vram(
    model: str,
    ps_runner: Callable[[], str] | None = None,
) -> OllamaPsVram:
    """Read model-attributable VRAM from ``ollama ps`` (primary VRAM source)."""
    ps_runner = ps_runner or _run_ollama_ps
    try:
        output = ps_runner()
    except FileNotFoundError:
        return OllamaPsVram(
            found=False,
            size_mb=None,
            processor="",
            gpu_fraction=0.0,
            vram_mb=0.0,
            detail="ollama executable not found",
        )
    except Exception as exc:
        return OllamaPsVram(
            found=False,
            size_mb=None,
            processor="",
            gpu_fraction=0.0,
            vram_mb=0.0,
            detail=f"ollama ps failed: {exc}",
        )

    return parse_ollama_ps_vram(output, model)


def parse_ollama_ps_vram(output: str, model: str) -> OllamaPsVram:
    """Parse the loaded model's SIZE and PROCESSOR into attributable VRAM."""
    target = model.strip().lower()
    for line in output.splitlines():
        if not line.strip():
            continue
        lowered = line.lower()
        if lowered.startswith("name") and "processor" in lowered:
            continue  # header row
        if target and target not in lowered:
            continue

        processor = _extract_processor(line)
        size_mb = _extract_size_mb(line)
        gpu_fraction = _gpu_fraction_from_processor(processor)
        vram_mb = (size_mb or 0.0) * gpu_fraction
        return OllamaPsVram(
            found=True,
            size_mb=size_mb,
            processor=processor,
            gpu_fraction=gpu_fraction,
            vram_mb=vram_mb,
            detail="" if processor else "processor column not parsed",
        )

    return OllamaPsVram(
        found=False,
        size_mb=None,
        processor="",
        gpu_fraction=0.0,
        vram_mb=0.0,
        detail="model not found in ollama ps output",
    )


def _extract_size_mb(line: str) -> float | None:
    match = _SIZE_PATTERN.search(line)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    return value * _SIZE_UNIT_TO_MB[unit]


def _gpu_fraction_from_processor(processor: str) -> float:
    """Fraction of the model on the GPU, from the PROCESSOR column (0.0-1.0)."""
    if not processor:
        return 0.0
    percentages = [int(value) for value in re.findall(r"(\d+)%", processor)]
    labels = [label.upper() for label in re.findall(r"(CPU|GPU)", processor, re.IGNORECASE)]
    if not percentages or not labels:
        return 0.0

    gpu_percent = 0
    for percent, label in zip(percentages, labels):
        if label == "GPU":
            gpu_percent += percent

    return min(max(gpu_percent / 100.0, 0.0), 1.0)


def resolve_model_name(workflow: Any) -> str:
    """Best-effort read of the model name from a workflow's SQL agent."""
    ollama_tool = getattr(getattr(workflow, "sql_agent", None), "ollama_tool", None)
    return str(getattr(ollama_tool, "model", "") or "")


def is_oom_error(message: str | None) -> bool:
    """Return True when an error message looks like a CUDA out-of-memory error."""
    if not message:
        return False
    lowered = message.lower()
    return any(signature in lowered for signature in _OOM_SIGNATURES)


def run_single_benchmark_question(
    question: dict[str, Any],
    mode_name: str,
    label: str,
    workflow: Any,
    run_index: int = 1,
) -> dict[str, Any]:
    """Run one question under one mode and capture a benchmark row."""
    question_id = str(question.get("id", ""))
    question_text = str(question.get("question", ""))
    error_message = ""
    oom = False

    try:
        state = workflow.run(question_text)
    except Exception as exc:
        error_message = str(exc)
        oom = is_oom_error(error_message)
        state = AnalyticsState(
            user_query=question_text,
            success=False,
            error_message=error_message,
        )

    if not isinstance(state, AnalyticsState):
        error_message = "Workflow did not return AnalyticsState."
        state = AnalyticsState(
            user_query=question_text,
            success=False,
            error_message=error_message,
        )

    success = bool(state.success)
    if not success and state.error_message:
        if not error_message:
            error_message = state.error_message
        if is_oom_error(state.error_message):
            oom = True

    tps = aggregate_tokens_per_second(state.tool_calls)

    return {
        "run_index": run_index,
        "mode": mode_name,
        "label": label,
        "question_id": question_id,
        "question": question_text,
        "success": success,
        "latency_total": state.latency.get("total", ""),
        "tokens_per_second": tps if tps is not None else "",
        "oom": oom,
        "error_message": error_message,
    }


def _default_sampler_factory() -> ResourceSampler:
    # Samples global GPU memory.used; the baseline->peak delta is the secondary
    # cross-check. Primary VRAM comes from ``ollama ps`` after generations.
    return ResourceSampler(vram_reader=query_gpu_vram_mb)


def unload_ollama_model(model: str) -> None:
    """Best-effort unload of a model from VRAM (``keep_alive=0``).

    Never raises; failures are ignored so the benchmark continues. The committed
    ``keep_alive`` default in ``configs/model.yaml`` is not changed.
    """
    try:
        OllamaTool.from_config().unload(model or None)
    except Exception:
        return


def run_gpu_cpu_benchmark(
    questions: list[dict[str, Any]],
    modes: tuple[BenchmarkMode, ...] | None = None,
    workflow_factory: WorkflowFactory = build_overridden_workflow,
    gpu_detector: GpuDetector = detect_gpu_engagement,
    sampler_factory: SamplerFactory = _default_sampler_factory,
    unloader: Unloader = unload_ollama_model,
    ps_vram_runner: PsRunner = _run_ollama_ps,
    global_vram_reader: GlobalVramReader = query_gpu_vram_mb,
    compute_apps_runner: PsRunner | None = None,
    warmup_question: str | None = None,
    logger: Logger = print,
    run_index: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the question set under every mode and return rows plus a summary.

    ``run_index`` stamps each row (default 1 for a single run); the repeated
    runner passes the current iteration so per-run rows stay distinguishable.
    """
    modes = modes or default_modes()
    warmup_text = warmup_question or _select_warmup_question(questions)

    rows: list[dict[str, Any]] = []
    mode_results: dict[str, ModeResult] = {}

    for mode in modes:
        workflow = workflow_factory(mode.num_gpu)
        model_name = resolve_model_name(workflow)

        # Start each mode from a clean GPU: unload any model still resident from
        # a prior mode/run (keep_alive keeps it loaded) before this mode loads.
        unloader(model_name)

        # Secondary cross-check baseline: global VRAM AFTER unload, before load,
        # so it excludes this mode's model. Desktop/browser usage cancels out
        # against the peak measured during generation.
        baseline_vram_mb = global_vram_reader()

        # Warm-up generation absorbs model-load time and is excluded from metrics.
        _warm_up(workflow, warmup_text, logger)

        label = mode.name
        engagement: GpuEngagement | None = None
        if mode.requires_gpu:
            engagement = gpu_detector(model_name)
            if not engagement.on_gpu:
                label = GPU_REQUESTED_BUT_RAN_ON_CPU
                logger(
                    f"WARNING: '{mode.name}' mode requested GPU offload but the "
                    f"model is not on the GPU (processor="
                    f"'{engagement.processor or 'unknown'}'). "
                    f"Labeling this run as '{GPU_REQUESTED_BUT_RAN_ON_CPU}'."
                )
            elif engagement.partial_offload:
                logger(
                    f"WARNING: '{mode.name}' mode is partially offloaded "
                    f"(processor='{engagement.processor}'); it may not fully fit."
                )

        sampler = sampler_factory()
        sampler.start()
        mode_rows: list[dict[str, Any]] = []
        oom_observed = False
        try:
            for question in questions:
                row = run_single_benchmark_question(
                    question=question,
                    mode_name=mode.name,
                    label=label,
                    workflow=workflow,
                    run_index=run_index,
                )
                if row["oom"]:
                    oom_observed = True
                    logger(
                        f"CUDA OOM on question '{row['question_id']}' in "
                        f"'{mode.name}' mode; continuing. {row['error_message']}"
                    )
                mode_rows.append(row)
        finally:
            usage = sampler.stop()

        # Primary VRAM source: read ``ollama ps`` while the model is still
        # loaded (after generations, before the next mode's unload).
        ps_vram = read_ollama_ps_vram(model_name, ps_runner=ps_vram_runner)

        # Optional best-effort per-process figure; unreliable/N/A under WDDM, so
        # it is recorded but never used as the primary value.
        compute_apps_vram_mb = query_ollama_vram_mb(runner=compute_apps_runner)

        rows.extend(mode_rows)
        mode_results[mode.name] = ModeResult(
            label=label,
            usage=usage,
            oom_observed=oom_observed,
            engagement=engagement,
            rows=mode_rows,
            ps_vram=ps_vram,
            baseline_vram_mb=baseline_vram_mb,
            compute_apps_vram_mb=compute_apps_vram_mb,
        )

    return rows, summarize_gpu_cpu_benchmark(mode_results)


def run_gpu_cpu_benchmark_repeated(
    questions: list[dict[str, Any]],
    repeat: int = 1,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the full benchmark ``repeat`` times and aggregate across runs.

    Per-run rows are returned tagged with ``run_index`` (1..N); the summary adds
    mean/sd/min/max for latency and tokens/sec across runs. VRAM is kept as a
    single representative value (it is stable) plus min/max for transparency.
    """
    if repeat < 1:
        raise ValueError("repeat must be greater than 0")
    if "run_index" in kwargs:
        raise TypeError("run_index is set per repetition and must not be passed")

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index in range(1, repeat + 1):
        run_rows, run_summary = run_gpu_cpu_benchmark(
            questions,
            run_index=index,
            **kwargs,
        )
        rows.extend(run_rows)
        summaries.append(run_summary)

    return rows, aggregate_repeated_summaries(summaries)


def aggregate_repeated_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-run summaries into mean/sd/min/max per mode and metric."""
    if not summaries:
        return {"timestamp": _current_timestamp(), "repeat": 0, "modes": {}}

    representative = summaries[-1]
    mode_names = list(representative.get("modes", {}).keys())

    modes_summary: dict[str, Any] = {}
    for mode_name in mode_names:
        per_run = [
            summary["modes"][mode_name]
            for summary in summaries
            if mode_name in summary.get("modes", {})
        ]
        base = dict(representative["modes"][mode_name])

        base["runs"] = len(per_run)
        base["latency_stats"] = aggregate_stats(
            [_optional_float(item.get("avg_total_latency")) for item in per_run]
        )
        base["tokens_per_second_stats"] = aggregate_stats(
            [_optional_float(item.get("avg_tokens_per_second")) for item in per_run]
        )
        # VRAM is stable; keep the representative single value plus min/max.
        base["vram_mb_stats"] = aggregate_stats(
            [_optional_float(item.get("vram_mb")) for item in per_run]
        )
        modes_summary[mode_name] = base

    return {
        "timestamp": _current_timestamp(),
        "repeat": len(summaries),
        "modes": modes_summary,
    }


def aggregate_stats(values: list[float | None]) -> dict[str, Any]:
    """Mean/sd/min/max over the numeric values, ignoring ``None`` entries.

    ``sd`` is the sample standard deviation (``n >= 2``); for a single value it
    is ``0.0``, and for no values every statistic is ``None``.
    """
    numbers = [value for value in values if value is not None]
    if not numbers:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}

    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "sd": statistics.stdev(numbers) if len(numbers) >= 2 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
    }


def _select_warmup_question(questions: list[dict[str, Any]]) -> str:
    for question in questions:
        text = str(question.get("question", "")).strip()
        if text:
            return text
    return DEFAULT_WARMUP_QUESTION


def _warm_up(workflow: Any, warmup_text: str, logger: Logger) -> None:
    try:
        workflow.run(warmup_text)
    except Exception as exc:  # Warm-up failures must not abort the benchmark.
        logger(f"Warm-up run failed (ignored for metrics): {exc}")


def summarize_gpu_cpu_benchmark(
    mode_results: dict[str, ModeResult],
) -> dict[str, Any]:
    """Summarize per-mode success, latency, throughput, and resource usage."""
    modes_summary: dict[str, Any] = {}
    for mode_name, result in mode_results.items():
        rows = result.rows
        total = len(rows)
        success_count = sum(1 for row in rows if bool(row.get("success")))
        latencies = [
            value
            for value in (_optional_float(row.get("latency_total")) for row in rows)
            if value is not None
        ]
        throughputs = [
            value
            for value in (
                _optional_float(row.get("tokens_per_second")) for row in rows
            )
            if value is not None
        ]
        fit_in_4gb = _compute_fit_in_4gb(result)

        ps_vram = result.ps_vram
        vram_mb = ps_vram.vram_mb if ps_vram is not None else 0.0
        if ps_vram is not None and ps_vram.processor:
            processor_split = ps_vram.processor
        elif result.engagement is not None:
            processor_split = result.engagement.processor
        else:
            processor_split = ""

        peak_global_vram_mb = result.usage.peak_vram_mb
        baseline_global_vram_mb = result.baseline_vram_mb
        if peak_global_vram_mb is not None and baseline_global_vram_mb is not None:
            vram_delta_mb: float | None = max(
                peak_global_vram_mb - baseline_global_vram_mb, 0.0
            )
        else:
            vram_delta_mb = None

        modes_summary[mode_name] = {
            "label": result.label,
            "total": total,
            "success_count": success_count,
            "success_rate": success_count / total if total else 0.0,
            "avg_total_latency": sum(latencies) / len(latencies) if latencies else 0.0,
            "avg_tokens_per_second": (
                sum(throughputs) / len(throughputs) if throughputs else 0.0
            ),
            # Primary VRAM: model-attributable, from ``ollama ps`` (WDDM-safe).
            "vram_mb": vram_mb,
            "processor_split": processor_split,
            # Secondary cross-check: baseline->peak delta of global memory.used.
            "vram_delta_mb": vram_delta_mb,
            "peak_global_vram_mb": peak_global_vram_mb,
            "baseline_global_vram_mb": baseline_global_vram_mb,
            # Best-effort per-process figure; may be 0/N/A under WDDM.
            "compute_apps_vram_mb": result.compute_apps_vram_mb,
            "peak_rss_mb": result.usage.peak_rss_mb,
            "vram_available": result.usage.vram_available,
            "oom_count": sum(1 for row in rows if bool(row.get("oom"))),
            "fit_in_4gb": fit_in_4gb,
            "gpu_engagement": _engagement_to_dict(result.engagement),
        }

    return {
        "timestamp": _current_timestamp(),
        "modes": modes_summary,
    }


def _compute_fit_in_4gb(result: ModeResult) -> bool:
    if result.oom_observed:
        return False
    engagement = result.engagement
    if engagement is None:
        # CPU-only modes do not offload, so they trivially "fit".
        return True
    return engagement.on_gpu and not engagement.partial_offload


def _engagement_to_dict(engagement: GpuEngagement | None) -> dict[str, Any] | None:
    if engagement is None:
        return None
    return {
        "on_gpu": engagement.on_gpu,
        "partial_offload": engagement.partial_offload,
        "processor": engagement.processor,
        "detail": engagement.detail,
    }


def write_gpu_cpu_benchmark_rows(
    rows: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_CSV_OUTPUT_PATH,
) -> Path:
    """Write per-question per-mode benchmark rows to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=GPU_CPU_BENCHMARK_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in GPU_CPU_BENCHMARK_COLUMNS}
            )
    return path


def write_gpu_cpu_benchmark_summary(
    summary: dict[str, Any],
    output_path: str | Path = DEFAULT_SUMMARY_OUTPUT_PATH,
) -> Path:
    """Write the benchmark summary JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
