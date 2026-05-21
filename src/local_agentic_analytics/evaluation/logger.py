"""CSV logging utilities for experiment runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_agentic_analytics.core.config import PROJECT_ROOT


RUN_LOG_COLUMNS = (
    "timestamp",
    "user_query",
    "generated_sql",
    "repaired_sql",
    "success",
    "error_message",
    "latency_total",
    "latency_sql_generation",
    "latency_sql_execution",
    "latency_reporting",
)
RUNS_CSV_PATH = PROJECT_ROOT / "reports" / "experiments" / "runs.csv"


def append_run_log(log: dict[str, Any]) -> None:
    """Append one workflow run log to ``reports/experiments/runs.csv``."""
    if not isinstance(log, dict):
        raise ValueError("log must be a dictionary")

    RUNS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RUNS_CSV_PATH.is_file()
    row = _normalize_log_row(log)

    with RUNS_CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _normalize_log_row(log: dict[str, Any]) -> dict[str, Any]:
    latency = log.get("latency")
    if not isinstance(latency, dict):
        latency = {}

    return {
        "timestamp": log.get("timestamp") or _current_timestamp(),
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
    }


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
