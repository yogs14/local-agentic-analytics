"""Gold SQL evaluation for energy text-to-SQL accuracy."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd

from local_agentic_analytics.core.config import PROJECT_ROOT, load_config
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.result_comparison import compare_result_sets
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool


DEFAULT_GOLD_QUESTIONS_PATH = (
    PROJECT_ROOT / "references" / "sql_gold" / "energy_gold_questions.json"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "experiments" / "sql_gold_eval.csv"
NUMERIC_TOLERANCE = 1e-6

# "numeric_match" is the legacy scalar-only metric (numeric_match_legacy);
# "result_match_full" is the row-set comparison added in Fase 2. Both are
# reported side by side; the legacy metric is never altered.
SQL_GOLD_EVAL_COLUMNS = (
    "question_id",
    "question",
    "agent_sql",
    "gold_sql",
    "agent_success",
    "gold_success",
    "numeric_match",
    "absolute_error",
    "relative_error",
    "agent_result",
    "gold_result",
    "error_message",
    "result_match_full",
    "result_match_reason",
)


@dataclass
class SqlExecutionResult:
    success: bool
    result_text: str
    numeric_value: float | None = None
    error_message: str = ""


def load_gold_questions(
    path: str | Path = DEFAULT_GOLD_QUESTIONS_PATH,
) -> list[dict[str, Any]]:
    """Load gold SQL benchmark questions."""
    questions_path = Path(path)
    if not questions_path.is_file():
        raise FileNotFoundError(f"Gold questions file not found: {questions_path}")

    try:
        data = json.loads(questions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid gold questions JSON: {questions_path}") from exc

    if not isinstance(data, list):
        raise ValueError("Gold questions JSON must contain a list")

    for index, item in enumerate(data, start=1):
        _validate_gold_question(item, index)

    return data


def run_sql_gold_evaluation(
    questions: list[dict[str, Any]],
    workflow: SequentialAnalyticsWorkflow | None = None,
    duckdb_tool: DuckDBTool | None = None,
) -> list[dict[str, Any]]:
    """Run gold SQL evaluation sequentially."""
    workflow = workflow or SequentialAnalyticsWorkflow()
    duckdb_tool = duckdb_tool or _load_default_duckdb_tool()

    rows = []
    for question in questions:
        try:
            rows.append(run_single_gold_question(question, workflow, duckdb_tool))
        except Exception as exc:
            rows.append(_failed_row(question, str(exc)))

    return rows


def run_single_gold_question(
    question: dict[str, Any],
    workflow: SequentialAnalyticsWorkflow,
    duckdb_tool: DuckDBTool,
) -> dict[str, Any]:
    """Run one gold SQL benchmark item."""
    question_id = str(question["id"])
    question_text = str(question["question"])
    errors: list[str] = []

    try:
        gold_sql = load_gold_sql(question["gold_sql_file"])
    except Exception as exc:
        gold_sql = ""
        errors.append(f"gold_sql_load: {exc}")

    try:
        state = workflow.run(question_text)
    except Exception as exc:
        state = AnalyticsState(
            user_query=question_text,
            error_message=str(exc),
            success=False,
        )

    if state.error_message and not state.success:
        errors.append(f"workflow: {state.error_message}")

    agent_sql = state.repaired_sql or state.generated_sql or ""
    if agent_sql:
        agent_result = execute_sql(duckdb_tool, agent_sql)
    else:
        agent_result = SqlExecutionResult(
            success=False,
            result_text="",
            error_message="Agent did not produce SQL",
        )

    if gold_sql:
        gold_result = execute_sql(duckdb_tool, gold_sql)
    else:
        gold_result = SqlExecutionResult(
            success=False,
            result_text="",
            error_message="Gold SQL was not loaded",
        )

    if agent_result.error_message:
        errors.append(f"agent_sql: {agent_result.error_message}")
    if gold_result.error_message:
        errors.append(f"gold_sql: {gold_result.error_message}")

    comparison = compare_numeric_results(
        agent_result.numeric_value,
        gold_result.numeric_value,
    )
    result_comparison = compare_result_sets(
        agent_result.result_text,
        gold_result.result_text,
    )

    return {
        "question_id": question_id,
        "question": question_text,
        "agent_sql": agent_sql,
        "gold_sql": gold_sql,
        "agent_success": agent_result.success,
        "gold_success": gold_result.success,
        "numeric_match": comparison["numeric_match"],
        "absolute_error": comparison["absolute_error"],
        "relative_error": comparison["relative_error"],
        "agent_result": agent_result.result_text,
        "gold_result": gold_result.result_text,
        "error_message": " | ".join(errors),
        "result_match_full": result_comparison["result_match_full"],
        "result_match_reason": result_comparison["result_match_reason"],
    }


def _failed_row(question: dict[str, Any], error_message: str) -> dict[str, Any]:
    return {
        "question_id": str(question.get("id", "")),
        "question": str(question.get("question", "")),
        "agent_sql": "",
        "gold_sql": "",
        "agent_success": False,
        "gold_success": False,
        "numeric_match": "",
        "absolute_error": "",
        "relative_error": "",
        "agent_result": "",
        "gold_result": "",
        "error_message": error_message,
        "result_match_full": "",
        "result_match_reason": "not_evaluated",
    }


def load_gold_sql(path: str | Path) -> str:
    """Load one gold SQL file."""
    sql_path = Path(path)
    if not sql_path.is_absolute():
        sql_path = PROJECT_ROOT / sql_path

    if not sql_path.is_file():
        raise FileNotFoundError(f"Gold SQL file not found: {sql_path}")

    return sql_path.read_text(encoding="utf-8").strip()


def execute_sql(duckdb_tool: DuckDBTool, sql: str) -> SqlExecutionResult:
    """Execute SQL and capture comparable result text."""
    try:
        dataframe = duckdb_tool.execute_query(sql)
    except Exception as exc:
        return SqlExecutionResult(
            success=False,
            result_text="",
            error_message=str(exc),
        )

    return SqlExecutionResult(
        success=True,
        result_text=dataframe_to_result_text(dataframe),
        numeric_value=extract_numeric_scalar(dataframe),
    )


def compare_numeric_results(
    agent_value: float | None,
    gold_value: float | None,
) -> dict[str, bool | float | str]:
    """Compare numeric scalar results if both sides are numeric."""
    if agent_value is None or gold_value is None:
        return {
            "numeric_match": "",
            "absolute_error": "",
            "relative_error": "",
        }

    absolute_error = abs(agent_value - gold_value)
    if gold_value == 0:
        relative_error = 0.0 if absolute_error <= NUMERIC_TOLERANCE else float("inf")
    else:
        relative_error = absolute_error / abs(gold_value)

    return {
        "numeric_match": absolute_error <= NUMERIC_TOLERANCE,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }


def write_sql_gold_results(
    rows: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Write gold SQL evaluation rows to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SQL_GOLD_EVAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in SQL_GOLD_EVAL_COLUMNS}
            )


def summarize_sql_gold_results(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize gold SQL evaluation rows."""
    total_questions = len(rows)
    agent_success_count = sum(1 for row in rows if bool(row.get("agent_success")))
    gold_success_count = sum(1 for row in rows if bool(row.get("gold_success")))
    numeric_rows = [row for row in rows if row.get("numeric_match") != ""]
    numeric_match_count = sum(1 for row in numeric_rows if bool(row.get("numeric_match")))
    result_full_rows = [row for row in rows if row.get("result_match_full") != ""]
    result_full_match_count = sum(
        1 for row in result_full_rows if bool(row.get("result_match_full"))
    )

    return {
        "total_questions": total_questions,
        "agent_success_count": agent_success_count,
        "gold_success_count": gold_success_count,
        "numeric_compared_count": len(numeric_rows),
        "numeric_match_count": numeric_match_count,
        "numeric_match_rate": (
            numeric_match_count / len(numeric_rows) if numeric_rows else 0.0
        ),
        "result_full_compared_count": len(result_full_rows),
        "result_full_match_count": result_full_match_count,
        "result_match_full_rate": (
            result_full_match_count / total_questions if total_questions else 0.0
        ),
    }


def dataframe_to_result_text(dataframe: pd.DataFrame) -> str:
    """Convert a query result to stable JSON text."""
    return dataframe.to_json(orient="records", date_format="iso")


def extract_numeric_scalar(dataframe: pd.DataFrame) -> float | None:
    """Extract one numeric scalar from a 1x1 DataFrame when possible."""
    if dataframe.shape != (1, 1):
        return None

    value = dataframe.iat[0, 0]
    if pd.isna(value) or isinstance(value, bool) or not isinstance(value, Number):
        return None

    return float(value)


def _validate_gold_question(item: Any, index: int) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"Gold question item #{index} must be a dictionary")

    for field_name in ("id", "question", "gold_sql_file", "expected_unit"):
        value = item.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Gold question item #{index} must contain non-empty '{field_name}'"
            )


def _load_default_duckdb_tool() -> DuckDBTool:
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
