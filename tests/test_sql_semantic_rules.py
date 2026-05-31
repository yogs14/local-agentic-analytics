from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.dataset_profile import (
    load_dataset_profile,
    profile_to_compact_sql_context,
)
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow
from local_agentic_analytics.prompts.sql_prompt import build_sql_prompt
from local_agentic_analytics.tools.sql_semantic_guard import (
    validate_energy_sql_semantics,
)


class FakeDuckDBTool:
    def __init__(self, result_sql: str):
        self.result_sql = result_sql
        self.executed_sql = []

    def get_schema(self, table_name: str) -> str:
        return f"{table_name}(Global_active_power DOUBLE)"

    def execute_query(self, sql: str) -> pd.DataFrame:
        self.executed_sql.append(sql)
        if sql != self.result_sql:
            raise AssertionError("Invalid semantic SQL should not be executed")
        return pd.DataFrame({"total_energy_kwh": [1.23]})


class FixedSQLAgent:
    def __init__(self, sql: str):
        self.sql = sql

    def generate_sql(
        self,
        question: str,
        schema: str,
        dataset_profile_context: str | None = None,
    ) -> str:
        return self.sql


class CapturingRepairAgent:
    def __init__(self, repaired_sql: str):
        self.repaired_sql = repaired_sql
        self.calls = []

    def repair_sql(
        self,
        failed_sql: str,
        error_message: str,
        schema: str,
        repair_attempted: bool = False,
        user_question: str = "",
    ) -> str:
        self.calls.append(
            {
                "failed_sql": failed_sql,
                "error_message": error_message,
                "schema": schema,
                "repair_attempted": repair_attempted,
                "user_question": user_question,
            }
        )
        return self.repaired_sql


class FakeReporterAgent:
    def generate_answer(self, question: str, sql: str, query_result):
        return "Total energi adalah 1,23 kWh."


class NoopAuditLogger:
    def log(self, **kwargs):
        return None


def test_sql_prompt_contains_energy_semantic_regression_rules():
    prompt = build_sql_prompt(
        question="Berapa total energi kWh?",
        schema="electric_power(Global_active_power DOUBLE)",
    )
    compact_context = profile_to_compact_sql_context(load_dataset_profile())

    assert "SUM(Global_active_power) / 60.0" in prompt
    assert "COUNT(*) FILTER (WHERE <column> IS NULL)" in prompt
    assert "COUNT(DISTINCT CASE WHEN" in prompt
    assert "SUM(Global_active_power) / 60.0" in compact_context
    assert "COUNT(*) FILTER (WHERE <column> IS NULL)" in compact_context


def test_validate_energy_sql_semantics_accepts_correct_energy_sql():
    is_valid, message = validate_energy_sql_semantics(
        question="Berapa total energi kWh?",
        sql="SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh FROM electric_power;",
    )

    assert is_valid is True
    assert message == ""


def test_validate_energy_sql_semantics_rejects_total_energy_without_dividing_by_60():
    is_valid, message = validate_energy_sql_semantics(
        question="Berapa total energi kWh?",
        sql="SELECT SUM(Global_active_power) AS total_energy_kwh FROM electric_power;",
    )

    assert is_valid is False
    assert "divide Global_active_power by 60.0" in message


def test_validate_energy_sql_semantics_rejects_forbidden_missing_value_pattern():
    is_valid, message = validate_energy_sql_semantics(
        question="Berapa jumlah missing value pada kolom Global_active_power?",
        sql=(
            "SELECT COUNT(DISTINCT CASE WHEN Global_active_power IS NULL THEN 1 END) "
            "FROM electric_power;"
        ),
    )

    assert is_valid is False
    assert "COUNT(DISTINCT CASE WHEN" in message


def test_validate_energy_sql_semantics_accepts_count_filter_for_missing_values():
    is_valid, message = validate_energy_sql_semantics(
        question="Berapa jumlah nilai hilang pada kolom Global_active_power?",
        sql=(
            "SELECT COUNT(*) FILTER (WHERE Global_active_power IS NULL) "
            "AS missing_global_active_power_count FROM electric_power;"
        ),
    )

    assert is_valid is True
    assert message == ""


def test_workflow_repairs_semantically_invalid_total_energy_before_duckdb_execution():
    invalid_sql = "SELECT SUM(Global_active_power) AS total_energy_kwh FROM electric_power;"
    repaired_sql = (
        "SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh "
        "FROM electric_power;"
    )
    duckdb_tool = FakeDuckDBTool(result_sql=repaired_sql)
    repair_agent = CapturingRepairAgent(repaired_sql=repaired_sql)
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=FixedSQLAgent(invalid_sql),
        repair_agent=repair_agent,
        reporter_agent=FakeReporterAgent(),
        audit_logger=NoopAuditLogger(),
    )

    state = workflow.run("Berapa total energi kWh?")

    assert state.success is True
    assert state.generated_sql == invalid_sql
    assert state.repaired_sql == repaired_sql
    assert duckdb_tool.executed_sql == [repaired_sql]
    assert repair_agent.calls[0]["failed_sql"] == invalid_sql
    assert "Semantic SQL validation failed" in repair_agent.calls[0]["error_message"]
    assert "divide Global_active_power by 60.0" in repair_agent.calls[0]["error_message"]
    tool_calls = {event["tool"]: event for event in state.tool_calls}
    assert tool_calls["sql_semantic_guard.validate"]["status"] == "error"
    assert tool_calls["ollama.sql_repair"]["status"] == "success"
    assert tool_calls["sql_semantic_guard.validate_repaired"]["status"] == "success"
    assert tool_calls["duckdb.query_repaired"]["status"] == "success"
    assert "duckdb.query" not in tool_calls
