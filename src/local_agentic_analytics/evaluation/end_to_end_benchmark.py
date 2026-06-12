"""End-to-end benchmark for custom and LangGraph workflows."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from local_agentic_analytics.core.config import PROJECT_ROOT, load_config
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.batch_eval import (
    DEFAULT_QUESTIONS_PATH,
    load_questions,
)
from local_agentic_analytics.evaluation.report_eval import evaluate_report_artifacts
from local_agentic_analytics.evaluation.sql_gold_eval import (
    DEFAULT_GOLD_QUESTIONS_PATH,
    SqlExecutionResult,
    compare_numeric_results,
    execute_sql,
    load_gold_questions,
    load_gold_sql,
)
from local_agentic_analytics.graph.report_workflow import EnergyReportWorkflow
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool


DEFAULT_BENCHMARK_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "end_to_end_benchmark.csv"
)
DEFAULT_BENCHMARK_SUMMARY_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "end_to_end_benchmark_summary.json"
)

END_TO_END_BENCHMARK_COLUMNS = (
    "workflow_type",
    "engine",
    "question_id",
    "question",
    "success",
    "generated_sql",
    "repaired_sql",
    "final_answer",
    "latency_total",
    "tool_call_count",
    "repair_used",
    "error_message",
    "gold_sql",
    "gold_numeric_match",
    "gold_absolute_error",
    "gold_relative_error",
    "gold_error_message",
    "report_tex_success",
    "report_pdf_success",
    "report_chart_count",
    "report_eval_score",
)


class QAWorkflowRunner(Protocol):
    """Minimal Q&A workflow interface used by the benchmark."""

    def run(self, user_query: str) -> AnalyticsState:
        """Run one question and return analytics state."""


class ReportWorkflowRunner(Protocol):
    """Minimal report workflow interface used by the benchmark."""

    def run(self) -> dict[str, Any]:
        """Generate report artifacts and return metadata."""


ReportEvaluator = Callable[[], dict[str, Any]]


def run_end_to_end_benchmark(
    questions: list[dict[str, Any]] | None = None,
    custom_qa_workflow: QAWorkflowRunner | None = None,
    langgraph_qa_workflow: QAWorkflowRunner | None = None,
    custom_report_workflow: ReportWorkflowRunner | None = None,
    langgraph_report_workflow: ReportWorkflowRunner | None = None,
    duckdb_tool: DuckDBTool | None = None,
    gold_questions: list[dict[str, Any]] | None = None,
    report_evaluator: ReportEvaluator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run Q&A and report workflows for both engines."""
    questions = questions if questions is not None else load_questions()
    custom_qa_workflow = custom_qa_workflow or SequentialAnalyticsWorkflow()
    langgraph_qa_workflow = langgraph_qa_workflow or _build_default_langgraph_workflow()
    custom_report_workflow = custom_report_workflow or EnergyReportWorkflow()
    langgraph_report_workflow = (
        langgraph_report_workflow or _build_default_langgraph_report_workflow()
    )
    duckdb_tool = duckdb_tool or _build_default_duckdb_tool()
    gold_question_map = _build_gold_question_map(gold_questions)
    report_evaluator = report_evaluator or evaluate_report_artifacts

    rows: list[dict[str, Any]] = []
    for question in questions:
        rows.append(
            run_single_qa_benchmark(
                question=question,
                engine="custom",
                workflow=custom_qa_workflow,
                duckdb_tool=duckdb_tool,
                gold_question=gold_question_map.get(str(question.get("id", ""))),
            )
        )
        rows.append(
            run_single_qa_benchmark(
                question=question,
                engine="langgraph",
                workflow=langgraph_qa_workflow,
                duckdb_tool=duckdb_tool,
                gold_question=gold_question_map.get(str(question.get("id", ""))),
            )
        )

    rows.append(
        run_single_report_benchmark(
            engine="custom",
            workflow=custom_report_workflow,
            report_evaluator=report_evaluator,
        )
    )
    rows.append(
        run_single_report_benchmark(
            engine="langgraph",
            workflow=langgraph_report_workflow,
            report_evaluator=report_evaluator,
        )
    )

    return rows, summarize_end_to_end_benchmark(rows)


def run_single_qa_benchmark(
    question: dict[str, Any],
    engine: str,
    workflow: QAWorkflowRunner,
    duckdb_tool: DuckDBTool,
    gold_question: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one Q&A benchmark row for one engine."""
    question_id = str(question.get("id", ""))
    question_text = str(question.get("question", ""))
    try:
        state = workflow.run(question_text)
    except Exception as exc:
        state = AnalyticsState(
            user_query=question_text,
            success=False,
            error_message=str(exc),
        )

    if not isinstance(state, AnalyticsState):
        state = AnalyticsState(
            user_query=question_text,
            success=False,
            error_message="Workflow did not return AnalyticsState.",
        )

    row = _qa_state_to_row(
        state=state,
        engine=engine,
        question_id=question_id,
        question=question_text,
    )
    row.update(_gold_sql_comparison_fields(state, gold_question, duckdb_tool))
    return row


def run_single_report_benchmark(
    engine: str,
    workflow: ReportWorkflowRunner,
    report_evaluator: ReportEvaluator,
) -> dict[str, Any]:
    """Run one report benchmark row for one engine."""
    try:
        metadata = workflow.run()
        if not isinstance(metadata, dict):
            metadata = {
                "success": False,
                "tex_success": False,
                "pdf_success": False,
                "error_message": "Workflow did not return metadata dict.",
            }
    except Exception as exc:
        metadata = {
            "success": False,
            "tex_success": False,
            "pdf_success": False,
            "error_message": str(exc),
            "pdf_error": "",
            "latency": {},
            "tool_calls": [],
        }

    report_eval_score = _safe_report_eval_score(report_evaluator)
    return _report_metadata_to_row(
        metadata=metadata,
        engine=engine,
        report_eval_score=report_eval_score,
    )


def write_end_to_end_benchmark_rows(
    rows: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_BENCHMARK_OUTPUT_PATH,
) -> Path:
    """Write end-to-end benchmark rows to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=END_TO_END_BENCHMARK_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in END_TO_END_BENCHMARK_COLUMNS}
            )
    return path


def write_end_to_end_benchmark_summary(
    summary: dict[str, Any],
    output_path: str | Path = DEFAULT_BENCHMARK_SUMMARY_PATH,
) -> Path:
    """Write end-to-end benchmark summary to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def summarize_end_to_end_benchmark(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create benchmark summary JSON from CSV rows."""
    custom_rows = [row for row in rows if row.get("engine") == "custom"]
    langgraph_rows = [row for row in rows if row.get("engine") == "langgraph"]
    qa_rows = [row for row in rows if row.get("workflow_type") == "qa"]
    tool_counts = [_int_value(row.get("tool_call_count")) for row in rows]
    report_rows = {
        str(row.get("engine")): row
        for row in rows
        if row.get("workflow_type") == "report"
    }
    numeric_rows = [row for row in qa_rows if row.get("gold_numeric_match") != ""]

    return {
        "timestamp": _current_timestamp(),
        "total_row_count": len(rows),
        "qa_question_count": len(qa_rows) // 2,
        "custom_success_rate": _success_rate(custom_rows),
        "langgraph_success_rate": _success_rate(langgraph_rows),
        "custom_avg_latency": _avg_latency(custom_rows),
        "langgraph_avg_latency": _avg_latency(langgraph_rows),
        "report_pdf_success": {
            "custom": _bool_value(report_rows.get("custom", {}).get("report_pdf_success")),
            "langgraph": _bool_value(
                report_rows.get("langgraph", {}).get("report_pdf_success")
            ),
        },
        "report_eval_score": {
            "custom": _optional_float(report_rows.get("custom", {}).get("report_eval_score")),
            "langgraph": _optional_float(
                report_rows.get("langgraph", {}).get("report_eval_score")
            ),
        },
        "avg_tool_call_count": (
            sum(tool_counts) / len(tool_counts) if tool_counts else 0.0
        ),
        "gold_numeric_compared_count": len(numeric_rows),
        "gold_numeric_match_count": sum(
            1 for row in numeric_rows if _bool_value(row.get("gold_numeric_match"))
        ),
        "gold_numeric_match_rate": _gold_numeric_match_rate(numeric_rows),
    }


def load_gold_questions_if_available(
    path: str | Path = DEFAULT_GOLD_QUESTIONS_PATH,
) -> list[dict[str, Any]]:
    """Load gold SQL questions when present; otherwise return an empty list."""
    try:
        return load_gold_questions(path)
    except (FileNotFoundError, ValueError):
        return []


def _qa_state_to_row(
    state: AnalyticsState,
    engine: str,
    question_id: str,
    question: str,
) -> dict[str, Any]:
    return {
        "workflow_type": "qa",
        "engine": engine,
        "question_id": question_id,
        "question": question,
        "success": bool(state.success),
        "generated_sql": state.generated_sql or "",
        "repaired_sql": state.repaired_sql or "",
        "final_answer": state.final_answer or "",
        "latency_total": state.latency.get("total", ""),
        "tool_call_count": len(state.tool_calls),
        "repair_used": bool(state.repaired_sql),
        "error_message": state.error_message or "",
        "report_tex_success": "",
        "report_pdf_success": "",
        "report_chart_count": "",
        "report_eval_score": "",
    }


def _report_metadata_to_row(
    metadata: dict[str, Any],
    engine: str,
    report_eval_score: float | None,
) -> dict[str, Any]:
    return {
        "workflow_type": "report",
        "engine": engine,
        "question_id": "",
        "question": "",
        "success": bool(metadata.get("success")),
        "generated_sql": "",
        "repaired_sql": "",
        "final_answer": "",
        "latency_total": _latency_total(metadata),
        "tool_call_count": _tool_call_count(metadata),
        "repair_used": False,
        "error_message": _report_error_message(metadata),
        "gold_sql": "",
        "gold_numeric_match": "",
        "gold_absolute_error": "",
        "gold_relative_error": "",
        "gold_error_message": "",
        "report_tex_success": bool(metadata.get("tex_success")),
        "report_pdf_success": bool(metadata.get("pdf_success")),
        "report_chart_count": _int_value(metadata.get("chart_count")),
        "report_eval_score": report_eval_score if report_eval_score is not None else "",
    }


def _gold_sql_comparison_fields(
    state: AnalyticsState,
    gold_question: dict[str, Any] | None,
    duckdb_tool: DuckDBTool,
) -> dict[str, Any]:
    defaults = {
        "gold_sql": "",
        "gold_numeric_match": "",
        "gold_absolute_error": "",
        "gold_relative_error": "",
        "gold_error_message": "",
    }
    if gold_question is None:
        return defaults

    errors: list[str] = []
    try:
        gold_sql = load_gold_sql(gold_question["gold_sql_file"])
    except Exception as exc:
        defaults["gold_error_message"] = f"gold_sql_load: {exc}"
        return defaults

    agent_sql = state.repaired_sql or state.generated_sql or ""
    if agent_sql:
        agent_result = execute_sql(duckdb_tool, agent_sql)
    else:
        agent_result = SqlExecutionResult(
            success=False,
            result_text="",
            error_message="Agent did not produce SQL",
        )

    gold_result = execute_sql(duckdb_tool, gold_sql)
    if agent_result.error_message:
        errors.append(f"agent_sql: {agent_result.error_message}")
    if gold_result.error_message:
        errors.append(f"gold_sql: {gold_result.error_message}")

    comparison = compare_numeric_results(
        agent_result.numeric_value,
        gold_result.numeric_value,
    )
    return {
        "gold_sql": gold_sql,
        "gold_numeric_match": comparison["numeric_match"],
        "gold_absolute_error": comparison["absolute_error"],
        "gold_relative_error": comparison["relative_error"],
        "gold_error_message": " | ".join(errors),
    }


def _build_gold_question_map(
    gold_questions: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if gold_questions is None:
        gold_questions = load_gold_questions_if_available()
    return {str(question.get("id", "")): question for question in gold_questions}


def _build_default_langgraph_workflow() -> QAWorkflowRunner:
    from local_agentic_analytics.graph.langgraph_workflow import (
        LangGraphAnalyticsWorkflow,
    )

    return LangGraphAnalyticsWorkflow()


def _build_default_langgraph_report_workflow() -> ReportWorkflowRunner:
    from local_agentic_analytics.graph.langgraph_report_workflow import (
        LangGraphReportWorkflow,
    )

    return LangGraphReportWorkflow()


def _build_default_duckdb_tool() -> DuckDBTool:
    config = load_config("duckdb.yaml")
    duckdb_config = config.get("duckdb", {})
    db_path = ""
    if isinstance(duckdb_config, dict):
        db_path = str(duckdb_config.get("database_path", ""))
    if not db_path:
        db_path = "databases/duckdb/analytics.duckdb"

    resolved = Path(db_path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return DuckDBTool(str(resolved))


def _safe_report_eval_score(report_evaluator: ReportEvaluator) -> float | None:
    try:
        result = report_evaluator()
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    return _optional_float(result.get("final_report_score", result.get("final_score")))


def _latency_total(metadata: dict[str, Any]) -> float | str:
    latency = metadata.get("latency")
    if not isinstance(latency, dict) or not latency:
        return ""
    total = _optional_float(latency.get("total"))
    if total is not None:
        return total
    values = [
        value for value in (_optional_float(item) for item in latency.values())
        if value is not None
    ]
    return sum(values) if values else ""


def _tool_call_count(metadata: dict[str, Any]) -> int:
    tool_calls = metadata.get("tool_calls")
    if not isinstance(tool_calls, list):
        return 0
    return len(tool_calls)


def _report_error_message(metadata: dict[str, Any]) -> str:
    messages = []
    for key in ("error_message", "pdf_error"):
        value = str(metadata.get(key, "")).strip()
        if value:
            messages.append(value)
    return " | ".join(messages)


def _success_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if _bool_value(row.get("success"))) / len(rows)


def _avg_latency(rows: list[dict[str, Any]]) -> float:
    latencies = [
        value for value in (_optional_float(row.get("latency_total")) for row in rows)
        if value is not None
    ]
    return sum(latencies) / len(latencies) if latencies else 0.0


def _gold_numeric_match_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if _bool_value(row.get("gold_numeric_match"))) / len(rows)


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.casefold() == "true"
    return bool(value)


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
