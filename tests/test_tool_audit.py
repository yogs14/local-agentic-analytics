from pathlib import Path
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.audit_logger import ToolAuditLogger
from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow


class FakeDuckDBTool:
    def get_schema(self, table_name: str) -> str:
        return f"{table_name}(value INTEGER)"

    def execute_query(self, sql: str) -> pd.DataFrame:
        return pd.DataFrame({"value": [1]})


class FakeMetricsOllamaTool:
    def __init__(self, metrics):
        self.metrics = metrics

    def get_last_metrics(self):
        return dict(self.metrics)


class FakeSQLAgent:
    def __init__(self):
        self.ollama_tool = FakeMetricsOllamaTool(
            {
                "total_duration": 2.0,
                "load_duration": 0.1,
                "prompt_eval_count": 32,
                "prompt_eval_duration": 0.4,
                "eval_count": 8,
                "eval_duration": 1.5,
            }
        )

    def generate_sql(self, question: str, schema: str, dataset_profile_context=None) -> str:
        return "SELECT 1 AS value"


class FakeReporterAgent:
    def __init__(self):
        self.ollama_tool = FakeMetricsOllamaTool({"eval_duration": 0.25})

    def generate_answer(self, question: str, sql: str, query_result):
        return "Nilainya adalah 1."


def test_workflow_records_tool_calls_to_state_and_jsonl(tmp_path):
    audit_path = tmp_path / "tool_call_audit.jsonl"
    workflow = SequentialAnalyticsWorkflow(
        duckdb_tool=FakeDuckDBTool(),
        sql_agent=FakeSQLAgent(),
        reporter_agent=FakeReporterAgent(),
        audit_logger=ToolAuditLogger(log_path=audit_path),
    )

    state = workflow.run("Berapa nilai agregat?")

    assert state.success is True
    assert state.route == "llm_sql"
    assert state.retrieved_context == []
    assert "duckdb.schema" in state.selected_tools
    assert "ollama.sql_generation" in state.selected_tools
    assert "duckdb.query" in state.selected_tools
    assert "ollama.reporting" in state.selected_tools

    state_events_by_tool = {event["tool"]: event for event in state.tool_calls}
    assert state_events_by_tool["duckdb.schema"]["status"] == "success"
    assert state_events_by_tool["rule_based_sql_resolver.resolve"]["status"] == "no_match"
    assert state_events_by_tool["ollama.sql_generation"]["metadata"]["eval_count"] == 8
    assert state_events_by_tool["duckdb.query"]["status"] == "success"
    assert state_events_by_tool["ollama.reporting"]["metadata"]["eval_duration"] == 0.25

    audit_events = [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    required_keys = {
        "timestamp",
        "component",
        "action",
        "tool",
        "status",
        "latency_seconds",
        "input_summary",
        "output_summary",
        "error_message",
        "metadata",
    }
    assert all(required_keys <= set(event) for event in audit_events)

    audit_events_by_tool = {event["tool"]: event for event in audit_events}
    assert audit_events_by_tool["ollama.sql_generation"]["metadata"] == {
        "total_duration": 2.0,
        "load_duration": 0.1,
        "prompt_eval_count": 32,
        "prompt_eval_duration": 0.4,
        "eval_count": 8,
        "eval_duration": 1.5,
    }
