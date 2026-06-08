"""CSV logging utilities for experiment runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from local_agentic_analytics.core.config import PROJECT_ROOT


RUN_LOG_COLUMNS = (
    "timestamp",
    "engine",
    "user_query",
    "generated_sql",
    "repaired_sql",
    "success",
    "error_message",
    "latency_total",
    "latency_sql_generation",
    "latency_sql_execution",
    "latency_reporting",
    "selected_tools",
    "tool_calls",
    "route",
)
RUNS_CSV_PATH = PROJECT_ROOT / "reports" / "experiments" / "runs.csv"


def append_run_log(log: dict[str, Any]) -> None:
    """Append one workflow run log to ``reports/experiments/runs.csv``."""
    if not isinstance(log, dict):
        raise ValueError("log must be a dictionary")

    RUNS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ensure_current_run_log_columns()
    file_exists = RUNS_CSV_PATH.is_file()
    row = _normalize_log_row(log)

    with RUNS_CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _ensure_current_run_log_columns() -> None:
    if not RUNS_CSV_PATH.is_file():
        return

    with RUNS_CSV_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        existing_columns = tuple(reader.fieldnames or ())
        if existing_columns == RUN_LOG_COLUMNS:
            return
        rows = list(reader)

    with RUNS_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_LOG_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RUN_LOG_COLUMNS})


def _normalize_log_row(log: dict[str, Any]) -> dict[str, Any]:
    latency = log.get("latency")
    if not isinstance(latency, dict):
        latency = {}

    return {
        "timestamp": log.get("timestamp") or _current_timestamp(),
        "engine": log.get("engine", ""),
        "user_query": log.get("user_query", ""),
        "generated_sql": log.get("generated_sql", ""),
        "repaired_sql": log.get("repaired_sql", ""),
        "success": log.get("success", ""),
        "error_message": log.get("error_message", ""),
        "latency_total": log.get("latency_total", latency.get("total", "")),
        "latency_sql_generation": log.get(
            "latency_sql_generation", latency.get("sql_generation", "")
        ),
        "latency_sql_execution": log.get(
            "latency_sql_execution", latency.get("sql_execution", "")
        ),
        "latency_reporting": log.get(
            "latency_reporting", latency.get("reporting", "")
        ),
        "selected_tools": _json_dumps(log.get("selected_tools", [])),
        "tool_calls": _json_dumps(log.get("tool_calls", [])),
        "route": log.get("route", ""),
    }


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)
