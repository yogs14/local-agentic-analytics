"""Compare custom and LangGraph report workflow engines."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol

from local_agentic_analytics.core.config import PROJECT_ROOT
from local_agentic_analytics.graph.report_workflow import EnergyReportWorkflow


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "report_engine_comparison.json"
)


class ReportWorkflowRunner(Protocol):
    """Minimal report workflow interface used by the comparison evaluator."""

    def run(self) -> dict[str, Any]:
        """Run report generation and return metadata."""


def run_report_engine_comparison(
    custom_workflow: ReportWorkflowRunner | None = None,
    langgraph_workflow: ReportWorkflowRunner | None = None,
) -> dict[str, Any]:
    """Run both report engines and return a JSON-ready comparison."""
    if custom_workflow is None:
        custom_workflow = EnergyReportWorkflow()
    if langgraph_workflow is None:
        langgraph_workflow = _build_default_langgraph_report_workflow()

    custom_metadata = _run_report_workflow_safely(custom_workflow, engine="custom")
    langgraph_metadata = _run_report_workflow_safely(
        langgraph_workflow,
        engine="langgraph",
    )

    custom_chart_count = _int_value(custom_metadata.get("chart_count"))
    langgraph_chart_count = _int_value(langgraph_metadata.get("chart_count"))
    custom_tex_success = bool(custom_metadata.get("tex_success"))
    langgraph_tex_success = bool(langgraph_metadata.get("tex_success"))
    custom_pdf_success = bool(custom_metadata.get("pdf_success"))
    langgraph_pdf_success = bool(langgraph_metadata.get("pdf_success"))

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "custom_success": bool(custom_metadata.get("success")),
        "langgraph_success": bool(langgraph_metadata.get("success")),
        "custom_tex_success": custom_tex_success,
        "langgraph_tex_success": langgraph_tex_success,
        "custom_pdf_success": custom_pdf_success,
        "langgraph_pdf_success": langgraph_pdf_success,
        "custom_chart_count": custom_chart_count,
        "langgraph_chart_count": langgraph_chart_count,
        "custom_insight_success_count": _int_value(
            custom_metadata.get("insight_success_count")
        ),
        "langgraph_insight_success_count": _int_value(
            langgraph_metadata.get("insight_success_count")
        ),
        "custom_latency_total": _latency_total(custom_metadata),
        "langgraph_latency_total": _latency_total(langgraph_metadata),
        "custom_tool_call_count": _tool_call_count(custom_metadata),
        "langgraph_tool_call_count": _tool_call_count(langgraph_metadata),
        "same_chart_count": custom_chart_count == langgraph_chart_count,
        "same_tex_success": custom_tex_success == langgraph_tex_success,
        "same_pdf_success": custom_pdf_success == langgraph_pdf_success,
        "custom_error": _error_message(custom_metadata),
        "langgraph_error": _error_message(langgraph_metadata),
    }


def write_report_engine_comparison_result(
    comparison: dict[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write report engine comparison metadata to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _build_default_langgraph_report_workflow() -> ReportWorkflowRunner:
    from local_agentic_analytics.graph.langgraph_report_workflow import (
        LangGraphReportWorkflow,
    )

    return LangGraphReportWorkflow()


def _run_report_workflow_safely(
    workflow: ReportWorkflowRunner,
    engine: str,
) -> dict[str, Any]:
    try:
        metadata = workflow.run()
    except Exception as exc:
        return {
            "engine": engine,
            "success": False,
            "tex_success": False,
            "pdf_success": False,
            "chart_count": 0,
            "insight_success_count": 0,
            "latency": {},
            "tool_calls": [],
            "error_message": str(exc),
            "pdf_error": "",
        }

    if not isinstance(metadata, dict):
        return {
            "engine": engine,
            "success": False,
            "tex_success": False,
            "pdf_success": False,
            "chart_count": 0,
            "insight_success_count": 0,
            "latency": {},
            "tool_calls": [],
            "error_message": "Workflow did not return metadata dict.",
            "pdf_error": "",
        }

    normalized = dict(metadata)
    normalized.setdefault("engine", engine)
    normalized.setdefault("latency", {})
    normalized.setdefault("tool_calls", [])
    normalized.setdefault("error_message", "")
    normalized.setdefault("pdf_error", "")
    return normalized


def _latency_total(metadata: dict[str, Any]) -> float | None:
    latency = metadata.get("latency")
    if not isinstance(latency, dict) or not latency:
        return None

    total = _float_value(latency.get("total"))
    if total is not None:
        return total

    numeric_values = [
        value for value in (_float_value(item) for item in latency.values())
        if value is not None
    ]
    if not numeric_values:
        return None
    return sum(numeric_values)


def _tool_call_count(metadata: dict[str, Any]) -> int:
    tool_calls = metadata.get("tool_calls")
    if not isinstance(tool_calls, list):
        return 0
    return len(tool_calls)


def _error_message(metadata: dict[str, Any]) -> str:
    messages = [
        str(metadata.get(key, "")).strip()
        for key in ("error_message", "pdf_error")
        if str(metadata.get(key, "")).strip()
    ]
    return " | ".join(messages)


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
