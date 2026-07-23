"""Deterministic quality evaluation for generated report insight narratives.

The report pipeline already scores *structural* report quality (sections, charts,
PDF, unit/numeric coverage in the LaTeX) via :mod:`report_eval`. It does not score
the *semantic quality of the insight narratives themselves* -- a gap the thesis
flags explicitly as future work.

This module fills that gap without an LLM judge, so results stay reproducible. It
reuses the very same deterministic validator that filtered the fine-tune training
corpus (:func:`validate_narrative`), applied here to the *production* narratives
written by the energy InsightAgent. Each narrative is scored on:

* numeric groundedness -- every number traceable to the chart's input statistics
  (or a whitelisted derived value: load factor, coefficient of variation, range);
* unit presence -- at least one number carries a standard electrical unit
  (skipped for the dimensionless correlation chart);
* concept coverage -- domain vocabulary terms used;
* latency -- end-to-end and per-step timing from the report orchestration log;
* safety -- no forbidden tokens (external standards, untested statistics, numeric
  future predictions) and no truncation;
* formatting cleanliness -- no leaked Markdown (headings, bold, tables).

The stats block fed to the validator is rebuilt with the same flat formatter the
fine-tuned model was prompted with, so groundedness is checked against exactly the
numbers the model saw.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_agentic_analytics.core.config import PROJECT_ROOT
from local_agentic_analytics.finetune.validators import (
    extract_numbers,
    validate_narrative,
)
from local_agentic_analytics.finetune.value_sampler import UNIT_REQUIRED
from local_agentic_analytics.prompts.insight_prompt import (
    build_energy_finetune_prompt,
)


DEFAULT_REPORT_LOG_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "report_generation_log.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "insight_quality_eval.json"
)
DEFAULT_CSV_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "insight_quality_eval.csv"
)

# Markdown artefacts a clean academic narrative should not contain: ATX headings,
# bold/italic markers, or pipe tables. The model's baked system prompt forbids
# Markdown, so any hit is a formatting-quality regression worth surfacing.
_MARKDOWN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("heading", re.compile(r"(?m)^\s{0,3}#{1,6}\s")),
    ("bold_or_italic", re.compile(r"\*\*|\*[^*\n]+\*|__[^_\n]+__")),
    ("table", re.compile(r"(?m)^\s*\|.*\|\s*$")),
)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")


def _markdown_artifacts(text: str) -> list[str]:
    return [label for label, pattern in _MARKDOWN_PATTERNS if pattern.search(text)]


def _sentence_count(text: str) -> int:
    return len([part for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()])


def _count_reason(reasons: list[str], prefix: str) -> int:
    return sum(1 for reason in reasons if reason.startswith(prefix))


def evaluate_insight_record(
    record: dict[str, Any],
    *,
    min_concept_terms: int = 2,
) -> dict[str, Any]:
    """Score one ``{chart_id, stats, insight, ...}`` record from the report log."""
    chart_id = str(record.get("chart_id", "")).strip()
    narrative = str(record.get("insight", "")).strip()
    stats = record.get("stats")
    generation_success = bool(record.get("success", True))

    require_unit = bool(UNIT_REQUIRED.get(chart_id, True))

    if not generation_success or not narrative or not isinstance(stats, dict) or not stats:
        # A failed generation (e.g. Ollama unreachable fallback text) cannot be
        # graded for groundedness; record it explicitly so aggregates stay honest.
        return {
            "chart_id": chart_id,
            "evaluated": False,
            "generation_success": generation_success,
            "require_unit": require_unit,
            "ok": False,
            "reasons": ["not_evaluated"],
            "total_numbers": 0,
            "ungrounded_numbers": 0,
            "number_groundedness": 0.0,
            "has_unit": False,
            "concept_terms": 0,
            "enough_concept_terms": False,
            "truncated": True,
            "forbidden_tokens": 0,
            "markdown_artifacts": [],
            "markdown_clean": False,
            "word_count": len(narrative.split()),
            "sentence_count": _sentence_count(narrative),
        }

    stats_block = build_energy_finetune_prompt(
        {"chart_id": chart_id, "stats": stats}
    )
    result = validate_narrative(
        narrative,
        stats_block,
        chart_id,
        require_unit=require_unit,
        min_concept_terms=min_concept_terms,
    )

    total_numbers = len(extract_numbers(narrative))
    ungrounded = _count_reason(result.reasons, "ungrounded_number:")
    grounded = max(total_numbers - ungrounded, 0)
    number_groundedness = grounded / total_numbers if total_numbers else 1.0

    markdown_artifacts = _markdown_artifacts(narrative)

    return {
        "chart_id": chart_id,
        "evaluated": True,
        "generation_success": True,
        "require_unit": require_unit,
        "ok": result.ok,
        "reasons": list(result.reasons),
        "total_numbers": total_numbers,
        "ungrounded_numbers": ungrounded,
        "number_groundedness": round(number_groundedness, 6),
        "has_unit": "missing_unit" not in result.reasons,
        "concept_terms": result.concept_terms,
        "enough_concept_terms": result.concept_terms >= min_concept_terms,
        "truncated": "truncated" in result.reasons,
        "forbidden_tokens": _count_reason(result.reasons, "forbidden:"),
        "markdown_artifacts": markdown_artifacts,
        "markdown_clean": not markdown_artifacts,
        "word_count": len(narrative.split()),
        "sentence_count": _sentence_count(narrative),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _parse_iso_timestamp(ts: str) -> datetime | None:
    if not ts:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    ts_clean = re.sub(r"\.\d+([+-])", r"\1", ts)
    try:
        return datetime.strptime(ts_clean, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None


def _extract_latency_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract end-to-end and per-step latency from the report generation metadata."""
    latency_info: dict[str, Any] = {
        "total_elapsed_seconds": 0.0,
        "total_elapsed_label": "",
        "step_latencies": {},
        "tool_call_latency_total": 0.0,
        "tool_call_count": 0,
    }

    start_ts = str(metadata.get("timestamp_start", ""))
    end_ts = str(metadata.get("timestamp_end", ""))
    start_dt = _parse_iso_timestamp(start_ts)
    end_dt = _parse_iso_timestamp(end_ts)

    if start_dt and end_dt:
        elapsed = (end_dt - start_dt).total_seconds()
        latency_info["total_elapsed_seconds"] = round(elapsed, 3)
        latency_info["total_elapsed_label"] = f"{elapsed:.3f}s"

    step_latencies = metadata.get("latency")
    if isinstance(step_latencies, dict) and step_latencies:
        normalized: dict[str, float] = {}
        for k, v in step_latencies.items():
            try:
                normalized[str(k)] = round(float(v), 4)
            except (TypeError, ValueError):
                normalized[str(k)] = 0.0
        latency_info["step_latencies"] = normalized

    tool_calls = metadata.get("tool_calls")
    if isinstance(tool_calls, list):
        total = 0.0
        for tc in tool_calls:
            if isinstance(tc, dict):
                try:
                    total += float(tc.get("latency_seconds", 0.0))
                except (TypeError, ValueError):
                    pass
        latency_info["tool_call_latency_total"] = round(total, 4)
        latency_info["tool_call_count"] = len(tool_calls)

    return latency_info


def evaluate_insight_quality(
    metadata: dict[str, Any],
    *,
    min_concept_terms: int = 2,
) -> dict[str, Any]:
    """Aggregate per-insight quality scores over a report metadata object."""
    records = metadata.get("insights", [])
    per_insight = [
        evaluate_insight_record(record, min_concept_terms=min_concept_terms)
        for record in records
        if isinstance(record, dict)
    ]
    evaluated = [item for item in per_insight if item["evaluated"]]
    unit_scoped = [item for item in evaluated if item["require_unit"]]

    total_numbers = sum(item["total_numbers"] for item in evaluated)
    total_ungrounded = sum(item["ungrounded_numbers"] for item in evaluated)

    latency_info = _extract_latency_metrics(metadata)

    summary = {
        "model": str(metadata.get("model", "")),
        "engine": str(metadata.get("engine", "")),
        "insights_total": len(per_insight),
        "insights_evaluated": len(evaluated),
        "acceptance_rate": round(
            _mean([1.0 if item["ok"] else 0.0 for item in evaluated]), 6
        ),
        "number_groundedness": round(
            (total_numbers - total_ungrounded) / total_numbers if total_numbers else 1.0,
            6,
        ),
        "fully_grounded_rate": round(
            _mean([1.0 if item["ungrounded_numbers"] == 0 else 0.0 for item in evaluated]),
            6,
        ),
        "total_numbers": total_numbers,
        "total_ungrounded_numbers": total_ungrounded,
        "unit_presence_rate": round(
            _mean([1.0 if item["has_unit"] else 0.0 for item in unit_scoped]), 6
        ),
        "concept_coverage_rate": round(
            _mean([1.0 if item["enough_concept_terms"] else 0.0 for item in evaluated]),
            6,
        ),
        "mean_concept_terms": round(
            _mean([float(item["concept_terms"]) for item in evaluated]), 4
        ),
        "truncation_rate": round(
            _mean([1.0 if item["truncated"] else 0.0 for item in evaluated]), 6
        ),
        "forbidden_token_rate": round(
            _mean([1.0 if item["forbidden_tokens"] else 0.0 for item in evaluated]), 6
        ),
        "markdown_clean_rate": round(
            _mean([1.0 if item["markdown_clean"] else 0.0 for item in evaluated]), 6
        ),
        "mean_word_count": round(
            _mean([float(item["word_count"]) for item in evaluated]), 2
        ),
        "mean_sentence_count": round(
            _mean([float(item["sentence_count"]) for item in evaluated]), 2
        ),
        "total_latency_seconds": latency_info["total_elapsed_seconds"],
        "total_latency_label": latency_info["total_elapsed_label"],
        "step_latencies": latency_info["step_latencies"],
        "tool_call_latency_total": latency_info["tool_call_latency_total"],
        "tool_call_count": latency_info["tool_call_count"],
        "composite_insight_score": round(
            _mean([
                (total_numbers - total_ungrounded) / total_numbers if total_numbers else 1.0,
                _mean([1.0 if item["has_unit"] else 0.0 for item in unit_scoped]),
                _mean([1.0 if item["enough_concept_terms"] else 0.0 for item in evaluated]),
                _mean([1.0 if item["ok"] else 0.0 for item in evaluated]),
            ]), 6
        ),
    }
    summary["per_insight"] = per_insight
    return summary


def load_report_metadata(
    path: str | Path = DEFAULT_REPORT_LOG_PATH,
) -> dict[str, Any]:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Report metadata file not found: {metadata_path}")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Report metadata must be a JSON object: {metadata_path}")
    return data


def write_insight_quality_result(
    result: dict[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> dict[str, Path]:
    """Write the aggregate JSON and a per-insight CSV; return both paths."""
    json_out = Path(output_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_out = Path(csv_path)
    columns = [
        "chart_id",
        "evaluated",
        "ok",
        "total_numbers",
        "ungrounded_numbers",
        "number_groundedness",
        "has_unit",
        "require_unit",
        "concept_terms",
        "enough_concept_terms",
        "truncated",
        "forbidden_tokens",
        "markdown_clean",
        "word_count",
        "sentence_count",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for item in result.get("per_insight", []):
            writer.writerow([item.get(column, "") for column in columns])

    return {"json": json_out, "csv": csv_out}
