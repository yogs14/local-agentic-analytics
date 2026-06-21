from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.dataset_profile import DatasetProfile
from local_agentic_analytics.core.pipeline_toggles import PipelineToggles
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow


class FakeDuckDBTool:
    def __init__(self):
        self.schema_table_names = []
        self.executed_sql = []

    def get_schema(self, table_name: str) -> str:
        self.schema_table_names.append(table_name)
        return f"Table: {table_name}\nColumns:\nid: INTEGER"

    def execute_query(self, sql: str) -> pd.DataFrame:
        self.executed_sql.append(sql)
        return pd.DataFrame({"value": [1]})


class CapturingSQLAgent:
    def __init__(self):
        self.calls = []

    def generate_sql(
        self,
        question: str,
        schema: str,
        dataset_profile_context: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "question": question,
                "schema": schema,
                "dataset_profile_context": dataset_profile_context,
            }
        )
        return "SELECT 1 AS value"


class FakeReporterAgent:
    def generate_answer(self, question: str, sql: str, query_result):
        return "Jawaban ringkas."


class NoopAuditLogger:
    def log(self, *args, **kwargs):
        return None


def test_workflow_default_domain_energy_uses_profile_table_name():
    duckdb_tool = FakeDuckDBTool()
    sql_agent = CapturingSQLAgent()
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=sql_agent,
        reporter_agent=FakeReporterAgent(),
        audit_logger=NoopAuditLogger(),
    )

    state = workflow.run("Berapa jumlah record?")

    assert state.success is True
    assert workflow.table_name == "electric_power"
    assert duckdb_tool.schema_table_names == ["electric_power"]
    assert workflow.dataset_profile.name == "household_power_consumption"


def test_workflow_passes_dataset_profile_context_to_sql_agent():
    duckdb_tool = FakeDuckDBTool()
    sql_agent = CapturingSQLAgent()
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=sql_agent,
        reporter_agent=FakeReporterAgent(),
        audit_logger=NoopAuditLogger(),
    )

    workflow.run("Berapa total energi kWh?")

    context = sql_agent.calls[0]["dataset_profile_context"]
    assert context is not None
    assert "table_name: electric_power" in context
    assert "available_columns:" in context
    assert "SUM(Global_active_power) / 60.0" in context
    assert "CAST(datetime AS DATE)" in context
    assert "Dataset konsumsi listrik rumah tangga per menit" not in context


def test_workflow_finance_domain_uses_finance_profile_and_rule_sql():
    duckdb_tool = FakeDuckDBTool()
    # Disable the LLM planner so this rule-based SQL assertion stays deterministic
    # and offline; the deterministic resolver still routes it to STRUCTURED_SQL.
    workflow = SequentialAnalyticsWorkflow(
        domain="finance",
        duckdb_tool=duckdb_tool,
        sql_agent=CapturingSQLAgent(),
        reporter_agent=FakeReporterAgent(),
        audit_logger=NoopAuditLogger(),
        toggles=PipelineToggles(use_planner=False),
    )

    state = workflow.run("Berapa rata-rata harga penutupan?")

    assert state.success is True
    assert workflow.table_name == "stock_prices"
    assert workflow.dataset_profile.domain == "finance"
    assert workflow.dataset_profile.name == "sp500_analyst_ratings"
    assert duckdb_tool.schema_table_names == ["stock_prices"]
    assert state.generated_sql == (
        "SELECT AVG(close) AS avg_close_usd\n"
        "FROM stock_prices;"
    )
    assert state.route == "rule_based_sql"


def test_workflow_accepts_manual_table_name_override():
    duckdb_tool = FakeDuckDBTool()
    custom_profile = DatasetProfile(
        name="custom",
        domain="energy",
        table_name="profile_table",
        datetime_column="datetime",
    )
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=CapturingSQLAgent(),
        reporter_agent=FakeReporterAgent(),
        dataset_profile=custom_profile,
        table_name="manual_table",
        audit_logger=NoopAuditLogger(),
    )

    state = workflow.run("Berapa jumlah record?")

    assert state.success is True
    assert workflow.table_name == "manual_table"
    assert duckdb_tool.schema_table_names == ["manual_table"]
    assert "table_name: profile_table" in workflow.dataset_profile_context
