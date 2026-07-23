"""Repeated-run benchmark for the fine-tuned energy insight narrator.

A single report run is one stochastic sample of the model (temperature 0.4). To
report representative narrative-quality numbers -- and to quantify the narrator's
inference efficiency -- this benchmark regenerates the six energy insights over
several repeats, scores each narrative with the deterministic
:mod:`insight_quality_eval` checks, and records Ollama latency / throughput per
generation.

The model-calling runner takes an injected agent so the aggregation logic stays
unit-testable offline; only the script wires in a live
``InsightAgent.for_energy_finetune()``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from local_agentic_analytics.core.config import PROJECT_ROOT
from local_agentic_analytics.evaluation.insight_quality_eval import (
    evaluate_insight_record,
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "insight_model_benchmark.json"
)
DEFAULT_CSV_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "insight_model_benchmark.csv"
)


def tokens_per_second(metrics: dict[str, Any]) -> float | None:
    """Tokens/sec from Ollama eval metrics, or ``None`` when unavailable."""
    count = metrics.get("eval_count")
    duration = metrics.get("eval_duration")
    if not isinstance(count, (int, float)) or not isinstance(duration, (int, float)):
        return None
    if duration <= 0:
        return None
    return count / duration


def _stats(values: list[float]) -> dict[str, float]:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    if not clean:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    mean = sum(clean) / len(clean)
    if len(clean) > 1:
        variance = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
        sd = variance ** 0.5
    else:
        sd = 0.0
    return {
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
        "n": len(clean),
    }


def _rate(flags: list[bool]) -> float:
    return round(sum(1 for f in flags if f) / len(flags), 6) if flags else 0.0


def aggregate_benchmark_samples(
    samples: list[dict[str, Any]],
    *,
    repeats: int,
) -> dict[str, Any]:
    """Aggregate per-generation samples into quality + efficiency summaries."""
    qualities = [s["quality"] for s in samples]
    unit_scoped = [q for q in qualities if q.get("require_unit")]
    total_numbers = sum(q.get("total_numbers", 0) for q in qualities)
    total_ungrounded = sum(q.get("ungrounded_numbers", 0) for q in qualities)

    quality_summary = {
        "samples": len(samples),
        "acceptance_rate": _rate([q.get("ok", False) for q in qualities]),
        "number_groundedness": round(
            (total_numbers - total_ungrounded) / total_numbers if total_numbers else 1.0,
            6,
        ),
        "fully_grounded_rate": _rate(
            [q.get("ungrounded_numbers", 0) == 0 for q in qualities]
        ),
        "total_numbers": total_numbers,
        "total_ungrounded_numbers": total_ungrounded,
        "unit_presence_rate": _rate([q.get("has_unit", False) for q in unit_scoped]),
        "concept_coverage_rate": _rate(
            [q.get("enough_concept_terms", False) for q in qualities]
        ),
        "mean_concept_terms": round(
            sum(q.get("concept_terms", 0) for q in qualities) / len(qualities), 4
        )
        if qualities
        else 0.0,
        "markdown_clean_rate": _rate([q.get("markdown_clean", False) for q in qualities]),
        "mean_sentence_count": round(
            sum(q.get("sentence_count", 0) for q in qualities) / len(qualities), 4
        )
        if qualities
        else 0.0,
    }

    efficiency_summary = {
        "latency_seconds": _stats([s.get("latency_seconds") for s in samples]),
        "tokens_per_second": _stats([s.get("tokens_per_second") for s in samples]),
        "eval_count": _stats([s.get("eval_count") for s in samples]),
    }

    return {
        "repeats": repeats,
        "quality": quality_summary,
        "efficiency": efficiency_summary,
        "samples": samples,
    }


def run_insight_model_benchmark(
    agent: Any,
    chart_contexts: Iterable[dict[str, Any]],
    *,
    repeats: int = 3,
    min_concept_terms: int = 2,
) -> dict[str, Any]:
    """Generate and score the energy insights over ``repeats`` passes.

    ``agent`` must expose ``generate_insight(context)`` and an ``ollama_tool`` with
    ``get_last_metrics()`` (the :class:`InsightAgent` contract). Injected in tests.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    contexts = list(chart_contexts)
    samples: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        for context in contexts:
            narrative = agent.generate_insight(context)
            metrics = agent.ollama_tool.get_last_metrics()
            quality = evaluate_insight_record(
                {
                    "chart_id": context["chart_id"],
                    "stats": context["stats"],
                    "insight": narrative,
                    "success": True,
                },
                min_concept_terms=min_concept_terms,
            )
            samples.append(
                {
                    "chart_id": str(context["chart_id"]),
                    "repeat_index": repeat_index,
                    "latency_seconds": metrics.get("total_duration"),
                    "tokens_per_second": tokens_per_second(metrics),
                    "eval_count": metrics.get("eval_count"),
                    "quality": quality,
                }
            )

    summary = aggregate_benchmark_samples(samples, repeats=repeats)
    summary["model"] = getattr(getattr(agent, "ollama_tool", None), "model", "")
    return summary


def write_benchmark_result(
    result: dict[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> dict[str, Path]:
    json_out = Path(output_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_out = Path(csv_path)
    columns = [
        "chart_id",
        "repeat_index",
        "latency_seconds",
        "tokens_per_second",
        "eval_count",
        "ok",
        "total_numbers",
        "ungrounded_numbers",
        "has_unit",
        "concept_terms",
        "markdown_clean",
        "sentence_count",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for sample in result.get("samples", []):
            quality = sample.get("quality", {})
            writer.writerow(
                [
                    sample.get("chart_id", ""),
                    sample.get("repeat_index", ""),
                    sample.get("latency_seconds", ""),
                    sample.get("tokens_per_second", ""),
                    sample.get("eval_count", ""),
                    quality.get("ok", ""),
                    quality.get("total_numbers", ""),
                    quality.get("ungrounded_numbers", ""),
                    quality.get("has_unit", ""),
                    quality.get("concept_terms", ""),
                    quality.get("markdown_clean", ""),
                    quality.get("sentence_count", ""),
                ]
            )

    return {"json": json_out, "csv": csv_out}
