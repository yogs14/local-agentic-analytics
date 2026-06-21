from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.config import load_config
from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.graph import langgraph_workflow
from local_agentic_analytics.graph.langgraph_workflow import (
    LangGraphAnalyticsWorkflow,
    run_langgraph_workflow,
)


class FakeDuckDBTool:
    def __init__(self, results_by_sql=None, errors_by_sql=None):
        self.results_by_sql = results_by_sql or {}
        self.errors_by_sql = errors_by_sql or {}
        self.schema_table_names = []
        self.executed_sql = []

    def get_schema(self, table_name: str) -> str:
        self.schema_table_names.append(table_name)
        return f"{table_name}(datetime TIMESTAMP, Global_active_power DOUBLE)"

    def execute_query(self, sql: str) -> pd.DataFrame:
        self.executed_sql.append(sql)
        if sql in self.errors_by_sql:
            raise ValueError(self.errors_by_sql[sql])
        return self.results_by_sql[sql]


class FakeSQLAgent:
    def __init__(self, sql: str):
        self.sql = sql
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
        return self.sql


class FailingSQLAgent:
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
        raise AssertionError("SQLAgent should not be called")


class FakeRepairAgent:
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
    def __init__(self, answer: str = "Jawaban ringkas."):
        self.answer = answer
        self.calls = []

    def generate_answer(self, question: str, sql: str, query_result):
        self.calls.append(
            {
                "question": question,
                "sql": sql,
                "query_result": query_result,
            }
        )
        return self.answer


class NoopAuditLogger:
    def log(self, **kwargs):
        return None


class StrictDuckDBTool:
    def get_schema(self, table_name: str) -> str:
        raise AssertionError("SQL schema must not be loaded on the RAG route")

    def execute_query(self, sql: str) -> pd.DataFrame:
        raise AssertionError("SQL must not be executed on the RAG route")


class FakeChromaDBTool:
    def __init__(self, matches):
        self.matches = matches
        self.queries = []

    def query(self, text, top_k=3, where=None):
        self.queries.append({"text": text, "top_k": top_k, "where": where})
        return list(self.matches)


def _news_matches():
    return [
        {
            "document": "TSLA melonjak setelah laporan pengiriman kuat.",
            "metadata": {"ticker": "TSLA", "date": "2020-01-10", "publisher": "Reuters"},
            "distance": 0.1,
        }
    ]


def test_langgraph_routes_finance_news_question_to_rag():
    chroma_tool = FakeChromaDBTool(_news_matches())
    reporter = FakeReporterAgent(answer="Sentimen TSLA cenderung positif.")
    workflow = LangGraphAnalyticsWorkflow(
        domain="finance",
        duckdb_tool=StrictDuckDBTool(),
        sql_agent=FailingSQLAgent(),
        reporter_agent=reporter,
        chroma_tool=chroma_tool,
        audit_logger=NoopAuditLogger(),
    )

    state = workflow.run("Bagaimana sentimen berita terbaru tentang TSLA?")

    assert state.success is True
    assert state.planned_route == "RAG_NEWS"
    assert state.route == "rag_news"
    assert state.final_answer == "Sentimen TSLA cenderung positif."
    assert state.retrieved_context[0]["ticker"] == "TSLA"
    assert len(chroma_tool.queries) == 1
    tools = {event["tool"] for event in state.tool_calls}
    assert {"planner.route", "chromadb.query", "ollama.reporting"} <= tools
    assert "duckdb.schema" not in tools


def test_langgraph_routes_hybrid_question_to_fusion():
    chroma_tool = FakeChromaDBTool(_news_matches())
    duckdb_tool = FakeDuckDBTool(
        results_by_sql={
            (
                "SELECT\n"
                "    MIN(close) AS min_close_usd,\n"
                "    AVG(close) AS avg_close_usd,\n"
                "    MAX(close) AS max_close_usd,\n"
                "    COUNT(*) AS trading_days\n"
                "FROM stock_prices\n"
                "WHERE ticker = 'NVDA'\n"
                "  AND CAST(date AS DATE) BETWEEN DATE '2019-06-01' "
                "AND DATE '2019-06-30';"
            ): pd.DataFrame(
                {
                    "min_close_usd": [30.0],
                    "avg_close_usd": [33.0],
                    "max_close_usd": [36.0],
                    "trading_days": [20],
                }
            )
        }
    )
    workflow = LangGraphAnalyticsWorkflow(
        domain="finance",
        duckdb_tool=duckdb_tool,
        sql_agent=FailingSQLAgent(),
        reporter_agent=FakeReporterAgent(),
        chroma_tool=chroma_tool,
        audit_logger=NoopAuditLogger(),
    )

    state = workflow.run(
        "Ringkas pergerakan harga NVDA pada Juni 2019 dan kaitkan dengan beritanya."
    )

    assert state.success is True
    assert state.planned_route == "HYBRID"
    assert state.route == "hybrid"
    assert state.sql_result["row_count"] == 1
    assert state.retrieved_context[0]["ticker"] == "TSLA"
    assert chroma_tool.queries[0]["where"] == {"ticker": "NVDA"}


def test_run_langgraph_workflow_returns_analytics_state(monkeypatch):
    class FakeLangGraphAnalyticsWorkflow:
        def __init__(self, domain: str = "energy"):
            self.domain = domain

        def run(self, user_query: str) -> AnalyticsState:
            return AnalyticsState(user_query=user_query, success=True)

    monkeypatch.setattr(
        langgraph_workflow,
        "LangGraphAnalyticsWorkflow",
        FakeLangGraphAnalyticsWorkflow,
    )

    state = run_langgraph_workflow("Berapa nilainya?", domain="energy")

    assert isinstance(state, AnalyticsState)
    assert state.user_query == "Berapa nilainya?"
    assert state.success is True


def test_langgraph_workflow_uses_rule_based_sql_before_sql_agent():
    generated_sql = (
        "SELECT AVG(Global_active_power) AS avg_global_active_power_kw\n"
        "FROM electric_power\n"
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-16';"
    )
    result_df = pd.DataFrame({"avg_global_active_power_kw": [1.23]})
    duckdb_tool = FakeDuckDBTool(results_by_sql={generated_sql: result_df})
    sql_agent = FailingSQLAgent()
    reporter_agent = FakeReporterAgent(answer="Rata-rata adalah 1,23 kW.")
    workflow = LangGraphAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=sql_agent,
        repair_agent=FakeRepairAgent(repaired_sql="SELECT 1"),
        reporter_agent=reporter_agent,
        audit_logger=NoopAuditLogger(),
    )

    state = workflow.run(
        "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
    )

    assert state.success is True
    assert state.generated_sql == generated_sql
    assert state.route == "rule_based_sql"
    assert sql_agent.calls == []
    assert duckdb_tool.schema_table_names == ["electric_power"]
    assert duckdb_tool.executed_sql == [generated_sql]
    assert reporter_agent.calls[0]["sql"] == generated_sql
    tool_calls = {event["tool"]: event for event in state.tool_calls}
    assert tool_calls["duckdb.schema"]["status"] == "success"
    assert tool_calls["rule_based_sql_resolver.resolve"]["status"] == "success"
    assert tool_calls["sql_semantic_guard.validate"]["status"] == "success"
    assert tool_calls["duckdb.query"]["status"] == "success"
    assert tool_calls["ollama.reporting"]["status"] == "success"
    assert "ollama.sql_generation" not in tool_calls


def test_langgraph_workflow_runs_average_query_when_database_available():
    db_path = _default_duckdb_path()
    if not db_path.exists():
        pytest.skip("Default DuckDB database is not available.")

    reporter_agent = FakeReporterAgent(answer="Rata-rata adalah 3,05 kW.")
    workflow = LangGraphAnalyticsWorkflow(
        sql_agent=FailingSQLAgent(),
        repair_agent=FakeRepairAgent(repaired_sql="SELECT 1"),
        reporter_agent=reporter_agent,
        audit_logger=NoopAuditLogger(),
    )

    state = workflow.run(
        "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
    )

    assert state.success is True
    assert state.generated_sql is not None
    assert "AVG(Global_active_power)" in state.generated_sql
    assert state.sql_result is not None
    assert state.sql_result["row_count"] == 1
    assert state.final_answer == "Rata-rata adalah 3,05 kW."


def test_langgraph_workflow_repairs_semantically_invalid_sql_before_execution():
    invalid_sql = "SELECT SUM(Global_active_power) AS total_energy_kwh FROM electric_power;"
    repaired_sql = (
        "SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh "
        "FROM electric_power;"
    )
    duckdb_tool = FakeDuckDBTool(
        results_by_sql={
            repaired_sql: pd.DataFrame({"total_energy_kwh": [1.23]}),
        }
    )
    repair_agent = FakeRepairAgent(repaired_sql=repaired_sql)
    workflow = LangGraphAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=FakeSQLAgent(sql=invalid_sql),
        repair_agent=repair_agent,
        reporter_agent=FakeReporterAgent(answer="Total energi adalah 1,23 kWh."),
        audit_logger=NoopAuditLogger(),
    )

    state = workflow.run("Berapa total energi kWh?")

    assert state.success is True
    assert state.generated_sql == invalid_sql
    assert state.repaired_sql == repaired_sql
    assert state.route == "llm_sql_with_repair"
    assert duckdb_tool.executed_sql == [repaired_sql]
    assert repair_agent.calls[0]["failed_sql"] == invalid_sql
    assert "Semantic SQL validation failed" in repair_agent.calls[0]["error_message"]
    assert state.final_answer == "Total energi adalah 1,23 kWh."
    tool_calls = {event["tool"]: event for event in state.tool_calls}
    assert tool_calls["sql_semantic_guard.validate"]["status"] == "error"
    assert tool_calls["ollama.sql_repair"]["status"] == "success"
    assert tool_calls["sql_semantic_guard.validate_repaired"]["status"] == "success"
    assert tool_calls["duckdb.query_repaired"]["status"] == "success"
    assert "duckdb.query" not in tool_calls


def _default_duckdb_path() -> Path:
    config = load_config("duckdb.yaml")
    db_path = config.get("duckdb", {}).get("database_path")
    if not isinstance(db_path, str) or not db_path.strip():
        return PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"

    path = Path(db_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
