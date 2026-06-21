from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.pipeline_toggles import PipelineToggles
from local_agentic_analytics.graph.workflow import (
    NO_SQL_MARKER,
    SequentialAnalyticsWorkflow,
)


class FakeDuckDBTool:
    def __init__(self, default_df=None):
        self.default_df = (
            default_df if default_df is not None else pd.DataFrame({"value": [1]})
        )
        self.executed_sql = []

    def get_schema(self, table_name: str) -> str:
        return f"{table_name}(date DATE, ticker VARCHAR, close DOUBLE)"

    def execute_query(self, sql: str) -> pd.DataFrame:
        self.executed_sql.append(sql)
        return self.default_df


class StrictDuckDBTool:
    """Fails loudly if any SQL path touches it (used for RAG-only tests)."""

    def get_schema(self, table_name: str) -> str:
        raise AssertionError("SQL schema must not be loaded on the RAG route")

    def execute_query(self, sql: str) -> pd.DataFrame:
        raise AssertionError("SQL must not be executed on the RAG route")


class FailingSQLAgent:
    def __init__(self):
        self.calls = []

    def generate_sql(self, question, schema, dataset_profile_context=None):
        self.calls.append(question)
        raise AssertionError("SQLAgent should not be called")


class FakeSQLAgent:
    def __init__(self, sql):
        self.sql = sql
        self.calls = []

    def generate_sql(self, question, schema, dataset_profile_context=None):
        self.calls.append(question)
        return self.sql


class FakeReporterAgent:
    def __init__(self, answer="Jawaban ringkas."):
        self.answer = answer
        self.calls = []

    def generate_answer(self, question, sql, query_result):
        self.calls.append(
            {"question": question, "sql": sql, "query_result": query_result}
        )
        return self.answer


class FakeMetricsOllamaTool:
    def get_last_metrics(self):
        return {}


class FakePlannerAgent:
    def __init__(self, route=None, raise_error=False):
        self.route = route
        self.raise_error = raise_error
        self.calls = []
        self.ollama_tool = FakeMetricsOllamaTool()

    def plan_route(self, question):
        self.calls.append(question)
        if self.raise_error:
            raise RuntimeError("planner boom")
        return self.route


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
            "metadata": {
                "ticker": "TSLA",
                "date": "2020-01-10",
                "publisher": "Reuters",
            },
            "distance": 0.1,
        }
    ]


def test_energy_question_short_circuits_to_sql_without_planner():
    expected_sql = (
        "SELECT AVG(Global_active_power) AS avg_global_active_power_kw\n"
        "FROM electric_power\n"
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-16';"
    )
    duckdb_tool = FakeDuckDBTool()
    planner = FakePlannerAgent(raise_error=True)
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=FailingSQLAgent(),
        reporter_agent=FakeReporterAgent(),
        planner_agent=planner,
    )

    state = workflow.run(
        "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
    )

    assert state.success is True
    assert state.generated_sql == expected_sql
    assert state.route == "rule_based_sql"
    # Energy is forced to SQL without ever consulting the planner.
    assert state.planned_route == "STRUCTURED_SQL"
    assert state.route_source == "forced_energy"
    assert planner.calls == []
    tools = {event["tool"] for event in state.tool_calls}
    assert "planner.route" not in tools
    assert "ollama.planning" not in tools


def test_finance_news_question_routes_to_rag_and_executes():
    chroma_tool = FakeChromaDBTool(_news_matches())
    reporter = FakeReporterAgent(answer="Sentimen TSLA cenderung positif.")
    duckdb_tool = StrictDuckDBTool()
    workflow = SequentialAnalyticsWorkflow(
        domain="finance",
        duckdb_tool=duckdb_tool,
        sql_agent=FailingSQLAgent(),
        reporter_agent=reporter,
        chroma_tool=chroma_tool,
    )

    state = workflow.run("Bagaimana sentimen berita terbaru tentang TSLA?")

    assert state.success is True
    assert state.planned_route == "RAG_NEWS"
    assert state.route_source == "rule_based"
    assert state.route == "rag_news"
    assert state.final_answer == "Sentimen TSLA cenderung positif."
    assert len(chroma_tool.queries) == 1
    assert state.retrieved_context[0]["ticker"] == "TSLA"
    # Reporter is grounded on headlines, with no SQL.
    assert reporter.calls[0]["sql"] == NO_SQL_MARKER
    assert "retrieved_headlines" in reporter.calls[0]["query_result"]
    tools = {event["tool"] for event in state.tool_calls}
    assert {"planner.route", "chromadb.query", "ollama.reporting"} <= tools
    assert "duckdb.schema" not in tools
    assert "ollama.sql_generation" not in tools


def test_planner_llm_failure_falls_back_to_structured_sql():
    expected_sql = (
        "SELECT AVG(close) AS avg_close_usd\n"
        "FROM stock_prices\n"
        "WHERE ticker = 'NVDA';"
    )
    duckdb_tool = FakeDuckDBTool()
    planner = FakePlannerAgent(raise_error=True)
    workflow = SequentialAnalyticsWorkflow(
        domain="finance",
        duckdb_tool=duckdb_tool,
        sql_agent=FailingSQLAgent(),
        reporter_agent=FakeReporterAgent(),
        planner_agent=planner,
    )

    state = workflow.run("Berapa rata-rata harga penutupan NVDA?")

    assert state.success is True
    assert state.generated_sql == expected_sql
    assert state.route == "rule_based_sql"
    assert state.planned_route == "STRUCTURED_SQL"
    assert state.route_source == "default"
    assert planner.calls == ["Berapa rata-rata harga penutupan NVDA?"]
    events_by_tool = {event["tool"]: event for event in state.tool_calls}
    assert events_by_tool["planner.route"]["status"] == "success"
    # The failed planning call is still audited, with an error status.
    assert events_by_tool["ollama.planning"]["status"] == "error"


def test_hybrid_question_runs_full_fusion():
    chroma_tool = FakeChromaDBTool(_news_matches())
    duckdb_tool = FakeDuckDBTool(
        default_df=pd.DataFrame(
            {
                "min_close_usd": [30.0],
                "avg_close_usd": [33.0],
                "max_close_usd": [36.0],
                "trading_days": [20],
            }
        )
    )
    reporter = FakeReporterAgent()
    workflow = SequentialAnalyticsWorkflow(
        domain="finance",
        duckdb_tool=duckdb_tool,
        sql_agent=FailingSQLAgent(),
        reporter_agent=reporter,
        chroma_tool=chroma_tool,
    )

    state = workflow.run(
        "Ringkas pergerakan harga NVDA pada Juni 2019 dan kaitkan dengan beritanya."
    )

    assert state.success is True
    assert state.planned_route == "HYBRID"
    assert state.route == "hybrid"
    assert state.generated_sql == (
        "SELECT\n"
        "    MIN(close) AS min_close_usd,\n"
        "    AVG(close) AS avg_close_usd,\n"
        "    MAX(close) AS max_close_usd,\n"
        "    COUNT(*) AS trading_days\n"
        "FROM stock_prices\n"
        "WHERE ticker = 'NVDA'\n"
        "  AND CAST(date AS DATE) BETWEEN DATE '2019-06-01' AND DATE '2019-06-30';"
    )
    assert state.sql_result["row_count"] == 1
    assert state.retrieved_context[0]["ticker"] == "TSLA"
    assert chroma_tool.queries[0]["where"] == {"ticker": "NVDA"}
    assert "price_summary" in reporter.calls[0]["query_result"]
    tools = {event["tool"] for event in state.tool_calls}
    assert {"planner.route", "duckdb.query", "chromadb.query", "ollama.reporting"} <= tools


def test_hybrid_degrades_to_rag_when_date_cannot_be_extracted():
    chroma_tool = FakeChromaDBTool(_news_matches())
    workflow = SequentialAnalyticsWorkflow(
        domain="finance",
        duckdb_tool=StrictDuckDBTool(),
        sql_agent=FailingSQLAgent(),
        reporter_agent=FakeReporterAgent(),
        chroma_tool=chroma_tool,
    )

    state = workflow.run(
        "Ringkas pergerakan harga TSLA dan kaitkan dengan beritanya."
    )

    assert state.success is True
    # Started as HYBRID but degraded safely to RAG_NEWS (no date range).
    assert state.planned_route == "RAG_NEWS"
    assert state.route == "rag_news"
    assert "Degradasi HYBRID" in state.route_reasoning
    assert len(chroma_tool.queries) == 1


def test_planner_disabled_uses_rule_based_route_only():
    chroma_tool = FakeChromaDBTool(_news_matches())
    planner = FakePlannerAgent(raise_error=True)
    workflow = SequentialAnalyticsWorkflow(
        domain="finance",
        duckdb_tool=StrictDuckDBTool(),
        sql_agent=FailingSQLAgent(),
        reporter_agent=FakeReporterAgent(),
        chroma_tool=chroma_tool,
        planner_agent=planner,
        toggles=PipelineToggles(use_planner=False),
    )

    state = workflow.run("Bagaimana sentimen berita terbaru tentang TSLA?")

    # Rule-based resolver is confident here, so the LLM planner is never used.
    assert state.planned_route == "RAG_NEWS"
    assert state.route_source == "rule_based"
    assert planner.calls == []
