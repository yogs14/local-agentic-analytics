from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.graph.workflow import (
    SequentialAnalyticsWorkflow,
    dataframe_to_compact_result,
)


class FakeDuckDBTool:
    def __init__(self, results_by_sql=None, errors_by_sql=None):
        self.results_by_sql = results_by_sql or {}
        self.errors_by_sql = errors_by_sql or {}
        self.executed_sql = []

    def get_schema(self, table_name: str) -> str:
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


class FakeMetricsOllamaTool:
    def __init__(self, metrics):
        self.metrics = metrics

    def get_last_metrics(self):
        return dict(self.metrics)


class MetricsSQLAgent(FakeSQLAgent):
    def __init__(self, sql: str, metrics):
        super().__init__(sql)
        self.ollama_tool = FakeMetricsOllamaTool(metrics)


class MetricsRepairAgent(FakeRepairAgent):
    def __init__(self, repaired_sql: str, metrics):
        super().__init__(repaired_sql)
        self.ollama_tool = FakeMetricsOllamaTool(metrics)


class MetricsReporterAgent(FakeReporterAgent):
    def __init__(self, answer: str, metrics):
        super().__init__(answer=answer)
        self.ollama_tool = FakeMetricsOllamaTool(metrics)


class CapturingAuditLogger:
    def __init__(self):
        self.events = []

    def log(self, **kwargs):
        self.events.append(kwargs)


def test_workflow_success_without_repair():
    sql = "SELECT AVG(Global_active_power) AS avg_power FROM electric_power"
    result_df = pd.DataFrame({"avg_power": [1.23]})
    duckdb_tool = FakeDuckDBTool(results_by_sql={sql: result_df})
    sql_agent = FakeSQLAgent(sql=sql)
    repair_agent = FakeRepairAgent(repaired_sql="SELECT 1")
    reporter_agent = FakeReporterAgent(answer="Rata-rata konsumsi adalah 1,23.")
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=sql_agent,
        repair_agent=repair_agent,
        reporter_agent=reporter_agent,
    )

    state = workflow.run("Berapa rata-rata konsumsi listrik?")

    assert state.success is True
    assert state.generated_sql == sql
    assert state.repaired_sql is None
    assert state.sql_result == {
        "row_count": 1,
        "columns": ["avg_power"],
        "rows": [{"avg_power": 1.23}],
        "truncated": False,
    }
    assert state.final_answer == "Rata-rata konsumsi adalah 1,23."
    assert duckdb_tool.executed_sql == [sql]
    assert repair_agent.calls == []
    assert reporter_agent.calls[0]["sql"] == sql
    assert {"schema", "sql_generation", "sql_execution", "reporting", "total"} <= set(
        state.latency
    )


def test_workflow_uses_rule_based_sql_before_sql_agent():
    generated_sql = (
        "SELECT AVG(Global_active_power) AS avg_global_active_power_kw\n"
        "FROM electric_power\n"
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-16';"
    )
    result_df = pd.DataFrame({"avg_global_active_power_kw": [1.23]})
    duckdb_tool = FakeDuckDBTool(results_by_sql={generated_sql: result_df})
    sql_agent = FailingSQLAgent()
    audit_logger = CapturingAuditLogger()
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=sql_agent,
        repair_agent=FakeRepairAgent(repaired_sql="SELECT 1"),
        reporter_agent=FakeReporterAgent(answer="Rata-rata adalah 1,23 kW."),
        audit_logger=audit_logger,
    )

    state = workflow.run(
        "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
    )

    assert state.success is True
    assert state.generated_sql == generated_sql
    assert sql_agent.calls == []
    assert duckdb_tool.executed_sql == [generated_sql]
    tool_calls_by_tool = {event["tool"]: event for event in state.tool_calls}
    assert tool_calls_by_tool["duckdb.schema"]["status"] == "success"
    assert tool_calls_by_tool["rule_based_sql_resolver.resolve"]["status"] == "success"
    assert tool_calls_by_tool["duckdb.query"]["status"] == "success"
    assert tool_calls_by_tool["ollama.reporting"]["status"] == "success"
    assert "ollama.sql_generation" not in tool_calls_by_tool
    assert "duckdb.schema" in state.selected_tools
    assert "rule_based_sql_resolver.resolve" in state.selected_tools
    assert state.route == "rule_based_sql"
    assert "rule_based_sql_resolution" in state.latency
    assert "sql_generation" not in state.latency
    assert any(
        event["component"] == "rule_based_sql_resolver"
        and event["tool"] == "rule_based_sql_resolver.resolve"
        and event["output_summary"] == generated_sql
        for event in audit_logger.events
    )
    assert not any(
        event["tool"] == "ollama.sql_generation" for event in audit_logger.events
    )


def test_workflow_logs_ollama_metrics_for_generation_repair_and_reporting():
    failed_sql = "SELECT missing_column FROM electric_power"
    repaired_sql = "SELECT 1 AS value"
    result_df = pd.DataFrame({"value": [1]})
    audit_logger = CapturingAuditLogger()
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=FakeDuckDBTool(
            results_by_sql={repaired_sql: result_df},
            errors_by_sql={failed_sql: "Invalid SQL query: column not found"},
        ),
        sql_agent=MetricsSQLAgent(
            sql=failed_sql,
            metrics={"total_duration": 2.0, "prompt_eval_count": 12},
        ),
        repair_agent=MetricsRepairAgent(
            repaired_sql=repaired_sql,
            metrics={"total_duration": 3.0, "eval_count": 5},
        ),
        reporter_agent=MetricsReporterAgent(
            answer="Nilai adalah 1.",
            metrics={"eval_duration": 1.5, "eval_count": 9},
        ),
        audit_logger=audit_logger,
    )

    state = workflow.run("Berapa nilainya?")

    assert state.success is True
    ollama_events = {
        event["action"]: event
        for event in audit_logger.events
        if event["component"] == "ollama"
    }
    assert {"sql_generation", "sql_repair", "reporting"} <= set(ollama_events)
    assert ollama_events["sql_generation"]["metadata"]["total_duration"] == 2.0
    assert ollama_events["sql_generation"]["metadata"]["prompt_eval_count"] == 12
    assert ollama_events["sql_repair"]["metadata"]["total_duration"] == 3.0
    assert ollama_events["reporting"]["metadata"]["eval_duration"] == 1.5
    assert ollama_events["sql_generation"]["tool"] == "ollama.sql_generation"


def test_workflow_repairs_sql_once_after_execution_error():
    failed_sql = "SELECT total_power FROM electric_power"
    repaired_sql = "SELECT SUM(Global_active_power) AS total_power FROM electric_power"
    result_df = pd.DataFrame({"total_power": [100.0]})
    duckdb_tool = FakeDuckDBTool(
        results_by_sql={repaired_sql: result_df},
        errors_by_sql={failed_sql: "Invalid SQL query: column not found"},
    )
    repair_agent = FakeRepairAgent(repaired_sql=repaired_sql)
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=FakeSQLAgent(sql=failed_sql),
        repair_agent=repair_agent,
        reporter_agent=FakeReporterAgent(answer="Total konsumsi adalah 100."),
    )

    state = workflow.run("Berapa total konsumsi?")

    assert state.success is True
    assert state.generated_sql == failed_sql
    assert state.repaired_sql == repaired_sql
    assert state.error_message == "Invalid SQL query: column not found"
    assert duckdb_tool.executed_sql == [failed_sql, repaired_sql]
    assert len(repair_agent.calls) == 1
    assert repair_agent.calls[0]["repair_attempted"] is False
    assert repair_agent.calls[0]["user_question"] == "Berapa total konsumsi?"
    assert {"sql_repair", "repair_execution"} <= set(state.latency)


def test_workflow_fails_when_repaired_sql_also_fails():
    failed_sql = "SELECT total_power FROM electric_power"
    repaired_sql = "SELECT still_missing FROM electric_power"
    duckdb_tool = FakeDuckDBTool(
        errors_by_sql={
            failed_sql: "Invalid SQL query: column not found",
            repaired_sql: "Invalid SQL query: repaired column not found",
        }
    )
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=duckdb_tool,
        sql_agent=FakeSQLAgent(sql=failed_sql),
        repair_agent=FakeRepairAgent(repaired_sql=repaired_sql),
        reporter_agent=FakeReporterAgent(),
    )

    state = workflow.run("Berapa total konsumsi?")

    assert state.success is False
    assert state.generated_sql == failed_sql
    assert state.repaired_sql == repaired_sql
    assert state.final_answer is None
    assert "repaired column not found" in state.error_message
    assert duckdb_tool.executed_sql == [failed_sql, repaired_sql]


def test_dataframe_to_compact_result_truncates_preview_rows():
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    result = dataframe_to_compact_result(dataframe, max_rows=2)

    assert result == {
        "row_count": 3,
        "columns": ["value"],
        "rows": [{"value": 1}, {"value": 2}],
        "truncated": True,
    }
