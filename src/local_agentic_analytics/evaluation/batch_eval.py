"""Batch evaluation utilities for the energy text-to-SQL workflow."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_agentic_analytics.core.config import PROJECT_ROOT
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow


DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "energy_questions.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "experiments" / "batch_eval_energy.csv"

BATCH_EVAL_COLUMNS = (
    "question_id",
    "question",
    "generated_sql",
    "repaired_sql",
    "success",
    "error_message",
    "final_answer",
    "expected_unit",
    "latency_total",
    "sql_execution_success",
    "timestamp",
)


def load_questions(path: str | Path = DEFAULT_QUESTIONS_PATH) -> list[dict[str, Any]]:
    """Load batch evaluation questions from JSON."""
    questions_path = Path(path)
    if not questions_path.is_file():
        raise FileNotFoundError(f"Evaluation questions file not found: {questions_path}")

    try:
        data = json.loads(questions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON questions file: {questions_path}") from exc

    if not isinstance(data, list):
        raise ValueError("Evaluation questions JSON must contain a list")

    for index, item in enumerate(data, start=1):
        _validate_question(item, index)

    return data


def run_batch_evaluation(
    questions: list[dict[str, Any]],
    workflow: SequentialAnalyticsWorkflow | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Run all questions sequentially and return rows plus summary."""
    if workflow is None:
        workflow = SequentialAnalyticsWorkflow()

    rows = []
    for question in questions:
        rows.append(run_single_question(question, workflow))

    return rows, summarize_results(rows)


def run_single_question(
    question: dict[str, Any],
    workflow: SequentialAnalyticsWorkflow,
) -> dict[str, Any]:
    """Run one question and return a CSV-ready evaluation row."""
    timestamp = _current_timestamp()
    question_text = str(question["question"])

    try:
        state = workflow.run(question_text)
    except Exception as exc:
        state = AnalyticsState(
            user_query=question_text,
            error_message=str(exc),
            success=False,
        )

    return state_to_batch_row(
        state=state,
        question_id=str(question["id"]),
        expected_unit=str(question.get("expected_unit", "")),
        timestamp=timestamp,
    )


def state_to_batch_row(
    state: AnalyticsState,
    question_id: str,
    expected_unit: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Convert workflow state to a batch evaluation row."""
    return {
        "question_id": question_id,
        "question": state.user_query,
        "generated_sql": state.generated_sql or "",
        "repaired_sql": state.repaired_sql or "",
        "success": state.success,
        "error_message": state.error_message or "",
        "final_answer": state.final_answer or "",
        "expected_unit": expected_unit,
        "latency_total": state.latency.get("total", ""),
        "sql_execution_success": state.sql_result is not None,
        "timestamp": timestamp or _current_timestamp(),
    }


def write_batch_results(
    rows: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Write batch evaluation rows to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BATCH_EVAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in BATCH_EVAL_COLUMNS})


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Create a compact summary for batch results."""
    total_questions = len(rows)
    success_count = sum(1 for row in rows if bool(row.get("success")))
    failed_count = total_questions - success_count
    latencies = [
        float(row["latency_total"])
        for row in rows
        if row.get("latency_total") not in ("", None)
    ]

    return {
        "total_questions": total_questions,
        "success_count": success_count,
        "failed_count": failed_count,
        "success_rate": success_count / total_questions if total_questions else 0.0,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        "max_latency": max(latencies) if latencies else 0.0,
    }


def _validate_question(item: Any, index: int) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"Question item #{index} must be a dictionary")

    for field_name in ("id", "question", "expected_unit", "category"):
        value = item.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Question item #{index} must contain non-empty '{field_name}'"
            )


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
