"""Sequential text-to-SQL analytics workflow."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar

import pandas as pd

from local_agentic_analytics.agents.repair_agent import SQLRepairAgent
from local_agentic_analytics.agents.reporter_agent import ReporterAgent
from local_agentic_analytics.agents.sql_agent import SQLAgent
from local_agentic_analytics.core.config import PROJECT_ROOT, load_config
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool


T = TypeVar("T")
DEFAULT_TABLE_NAME = "electric_power"
DEFAULT_MAX_REPORT_ROWS = 20


class SequentialAnalyticsWorkflow:
    """Run the local analytics pipeline step by step on one thread."""

    def __init__(
        self,
        duckdb_tool: DuckDBTool | None = None,
        sql_agent: SQLAgent | None = None,
        repair_agent: SQLRepairAgent | None = None,
        reporter_agent: ReporterAgent | None = None,
        table_name: str = DEFAULT_TABLE_NAME,
        max_report_rows: int = DEFAULT_MAX_REPORT_ROWS,
    ):
        if not table_name or not table_name.strip():
            raise ValueError("table_name must not be empty")
        if max_report_rows < 1:
            raise ValueError("max_report_rows must be greater than 0")

        self.duckdb_tool = duckdb_tool or DuckDBTool(str(_load_default_db_path()))
        self.sql_agent = sql_agent or SQLAgent()
        self.repair_agent = repair_agent or SQLRepairAgent()
        self.reporter_agent = reporter_agent or ReporterAgent()
        self.table_name = table_name.strip()
        self.max_report_rows = max_report_rows

    def run(self, user_query: str) -> AnalyticsState:
        """Execute the full sequential workflow and return the final state."""
        state = AnalyticsState(user_query=user_query)
        total_start = perf_counter()

        try:
            state.schema = self._timed(
                state,
                "schema",
                lambda: self.duckdb_tool.get_schema(self.table_name),
            )
            state.generated_sql = self._timed(
                state,
                "sql_generation",
                lambda: self.sql_agent.generate_sql(
                    question=state.user_query,
                    schema=state.schema or "",
                ),
            )

            try:
                result_df = self._timed(
                    state,
                    "sql_execution",
                    lambda: self.duckdb_tool.execute_query(state.generated_sql or ""),
                )
            except Exception as exc:
                state.error_message = str(exc)
                state.repaired_sql = self._timed(
                    state,
                    "sql_repair",
                    lambda: self.repair_agent.repair_sql(
                        failed_sql=state.generated_sql or "",
                        error_message=state.error_message or "",
                        schema=state.schema or "",
                        repair_attempted=False,
                    ),
                )
                result_df = self._timed(
                    state,
                    "repair_execution",
                    lambda: self.duckdb_tool.execute_query(state.repaired_sql or ""),
                )

            state.sql_result = dataframe_to_compact_result(
                result_df,
                max_rows=self.max_report_rows,
            )
            executed_sql = state.repaired_sql or state.generated_sql or ""
            state.final_answer = self._timed(
                state,
                "reporting",
                lambda: self.reporter_agent.generate_answer(
                    question=state.user_query,
                    sql=executed_sql,
                    query_result=state.sql_result,
                ),
            )
            state.success = True
            return state
        except Exception as exc:
            state.success = False
            state.error_message = str(exc)
            return state
        finally:
            state.latency["total"] = perf_counter() - total_start

    @staticmethod
    def _timed(state: AnalyticsState, name: str, func: Callable[[], T]) -> T:
        start = perf_counter()
        try:
            return func()
        finally:
            state.latency[name] = perf_counter() - start


def dataframe_to_compact_result(
    dataframe: pd.DataFrame,
    max_rows: int = DEFAULT_MAX_REPORT_ROWS,
) -> dict[str, Any]:
    """Convert a query DataFrame into a compact JSON-serializable result."""
    if max_rows < 1:
        raise ValueError("max_rows must be greater than 0")

    preview = dataframe.head(max_rows)
    records = json.loads(preview.to_json(orient="records", date_format="iso"))

    return {
        "row_count": int(len(dataframe)),
        "columns": [str(column) for column in dataframe.columns],
        "rows": records,
        "truncated": len(dataframe) > max_rows,
    }


def _load_default_db_path() -> Path:
    config = load_config("duckdb.yaml")
    db_path = config.get("duckdb", {}).get("database_path")

    if not isinstance(db_path, str) or not db_path.strip():
        return PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"

    path = Path(db_path)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def run_workflow(user_query: str) -> AnalyticsState:
    """Convenience function for running the default sequential workflow."""
    workflow = SequentialAnalyticsWorkflow()
    return workflow.run(user_query)
