"""Compare custom and LangGraph analytics workflow engines."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol

from local_agentic_analytics.core.config import PROJECT_ROOT
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.batch_eval import (
    DEFAULT_QUESTIONS_PATH,
    load_questions,
)
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "engine_comparison.csv"
)

ENGINE_COMPARISON_COLUMNS = (
    "question_id",
    "question",
    "custom_success",
    "langgraph_success",
    "custom_sql",
    "langgraph_sql",
    "custom_answer",
    "langgraph_answer",
    "custom_latency_total",
    "langgraph_latency_total",
    "custom_tool_call_count",
    "langgraph_tool_call_count",
    "same_sql",
    "same_success_status",
    "both_success",
    "error_message",
)


class WorkflowRunner(Protocol):
    """Minimal workflow interface used by the comparison evaluator."""

    def run(self, user_query: str) -> AnalyticsState:
        """Run one user query and return analytics state."""


def run_engine_comparison(
    questions: list[dict[str, Any]],
    custom_workflow: WorkflowRunner | None = None,
    langgraph_workflow: WorkflowRunner | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Run both workflow engines for each question and return rows plus summary."""
    if custom_workflow is None:
        custom_workflow = SequentialAnalyticsWorkflow()
    if langgraph_workflow is None:
        langgraph_workflow = _build_default_langgraph_workflow()

    rows = [
        compare_single_question(question, custom_workflow, langgraph_workflow)
        for question in questions
    ]
    return rows, summarize_comparison(rows)


def compare_single_question(
    question: dict[str, Any],
    custom_workflow: WorkflowRunner,
    langgraph_workflow: WorkflowRunner,
) -> dict[str, Any]:
    """Run one question through both engines and return a CSV-ready row."""
    question_text = str(question["question"])
    custom_state = _run_workflow_safely(custom_workflow, question_text)
    langgraph_state = _run_workflow_safely(langgraph_workflow, question_text)

    custom_sql = _executed_sql(custom_state)
    langgraph_sql = _executed_sql(langgraph_state)
    custom_success = bool(custom_state.success)
    langgraph_success = bool(langgraph_state.success)

    return {
        "question_id": str(question["id"]),
        "question": question_text,
        "custom_success": custom_success,
        "langgraph_success": langgraph_success,
        "custom_sql": custom_sql,
        "langgraph_sql": langgraph_sql,
        "custom_answer": custom_state.final_answer or "",
        "langgraph_answer": langgraph_state.final_answer or "",
        "custom_latency_total": custom_state.latency.get("total", ""),
        "langgraph_latency_total": langgraph_state.latency.get("total", ""),
        "custom_tool_call_count": len(custom_state.tool_calls),
        "langgraph_tool_call_count": len(langgraph_state.tool_calls),
        "same_sql": _same_sql(custom_sql, langgraph_sql),
        "same_success_status": custom_success == langgraph_success,
        "both_success": custom_success and langgraph_success,
        "error_message": _combine_error_messages(
            custom_state=custom_state,
            langgraph_state=langgraph_state,
        ),
    }


def write_engine_comparison_results(
    rows: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Write engine comparison rows to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ENGINE_COMPARISON_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in ENGINE_COMPARISON_COLUMNS}
            )


def summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Create a compact summary for engine comparison results."""
    total_questions = len(rows)
    custom_latencies = _collect_latencies(rows, "custom_latency_total")
    langgraph_latencies = _collect_latencies(rows, "langgraph_latency_total")

    return {
        "total_questions": total_questions,
        "custom_success_count": sum(
            1 for row in rows if bool(row.get("custom_success"))
        ),
        "langgraph_success_count": sum(
            1 for row in rows if bool(row.get("langgraph_success"))
        ),
        "both_success_count": sum(1 for row in rows if bool(row.get("both_success"))),
        "same_sql_count": sum(1 for row in rows if bool(row.get("same_sql"))),
        "avg_custom_latency": (
            sum(custom_latencies) / len(custom_latencies) if custom_latencies else 0.0
        ),
        "avg_langgraph_latency": (
            sum(langgraph_latencies) / len(langgraph_latencies)
            if langgraph_latencies
            else 0.0
        ),
    }


def normalize_sql(sql: str) -> str:
    """Normalize SQL with simple whitespace and case folding."""
    return " ".join(sql.split()).casefold()


def _build_default_langgraph_workflow() -> WorkflowRunner:
    from local_agentic_analytics.graph.langgraph_workflow import (
        LangGraphAnalyticsWorkflow,
    )

    return LangGraphAnalyticsWorkflow()


def _run_workflow_safely(
    workflow: WorkflowRunner,
    question_text: str,
) -> AnalyticsState:
    try:
        state = workflow.run(question_text)
    except Exception as exc:
        return AnalyticsState(
            user_query=question_text,
            success=False,
            error_message=str(exc),
        )

    if not isinstance(state, AnalyticsState):
        return AnalyticsState(
            user_query=question_text,
            success=False,
            error_message="Workflow did not return AnalyticsState.",
        )

    return state


def _executed_sql(state: AnalyticsState) -> str:
    return state.repaired_sql or state.generated_sql or ""


def _same_sql(first_sql: str, second_sql: str) -> bool:
    first_normalized = normalize_sql(first_sql)
    second_normalized = normalize_sql(second_sql)
    if not first_normalized or not second_normalized:
        return False
    return first_normalized == second_normalized


def _combine_error_messages(
    custom_state: AnalyticsState,
    langgraph_state: AnalyticsState,
) -> str:
    messages = []
    if not custom_state.success and custom_state.error_message:
        messages.append(f"custom: {custom_state.error_message}")
    if not langgraph_state.success and langgraph_state.error_message:
        messages.append(f"langgraph: {langgraph_state.error_message}")
    return " | ".join(messages)


def _collect_latencies(rows: list[dict[str, Any]], column: str) -> list[float]:
    latencies = []
    for row in rows:
        value = row.get(column)
        if value in ("", None):
            continue
        try:
            latencies.append(float(value))
        except (TypeError, ValueError):
            continue
    return latencies
