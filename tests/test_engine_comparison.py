from pathlib import Path
import csv
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.engine_comparison import (
    ENGINE_COMPARISON_COLUMNS,
    normalize_sql,
    run_engine_comparison,
    write_engine_comparison_results,
)


class FakeWorkflow:
    def __init__(
        self,
        states_by_query: dict[str, AnalyticsState] | None = None,
        errors_by_query: dict[str, str] | None = None,
    ):
        self.states_by_query = states_by_query or {}
        self.errors_by_query = errors_by_query or {}
        self.calls: list[str] = []

    def run(self, user_query: str) -> AnalyticsState:
        self.calls.append(user_query)
        if user_query in self.errors_by_query:
            raise RuntimeError(self.errors_by_query[user_query])
        return self.states_by_query[user_query]


def make_state(
    user_query: str,
    sql: str = "SELECT 1",
    answer: str = "ok",
    success: bool = True,
    repaired_sql: str | None = None,
    error_message: str | None = None,
    latency: float = 1.0,
    tool_call_count: int = 1,
) -> AnalyticsState:
    return AnalyticsState(
        user_query=user_query,
        generated_sql=sql,
        repaired_sql=repaired_sql,
        final_answer=answer,
        sql_result={"rows": [{"value": 1}]} if success else None,
        success=success,
        error_message=error_message,
        latency={"total": latency},
        tool_calls=[
            {
                "tool": f"tool.{index}",
                "status": "success",
            }
            for index in range(tool_call_count)
        ],
    )


def test_run_engine_comparison_compares_sql_status_latency_and_tool_counts():
    questions = [
        {
            "id": "E001",
            "question": "Berapa nilainya?",
            "expected_unit": "count",
            "category": "aggregation",
        }
    ]
    custom_workflow = FakeWorkflow(
        {
            "Berapa nilainya?": make_state(
                "Berapa nilainya?",
                sql="SELECT  1 AS VALUE\nFROM electric_power",
                answer="custom answer",
                latency=2.0,
                tool_call_count=2,
            )
        }
    )
    langgraph_workflow = FakeWorkflow(
        {
            "Berapa nilainya?": make_state(
                "Berapa nilainya?",
                sql="select 1 as value from electric_power",
                answer="langgraph answer",
                latency=4.0,
                tool_call_count=3,
            )
        }
    )

    rows, summary = run_engine_comparison(
        questions,
        custom_workflow=custom_workflow,
        langgraph_workflow=langgraph_workflow,
    )

    assert rows[0]["custom_success"] is True
    assert rows[0]["langgraph_success"] is True
    assert rows[0]["same_sql"] is True
    assert rows[0]["same_success_status"] is True
    assert rows[0]["both_success"] is True
    assert rows[0]["custom_tool_call_count"] == 2
    assert rows[0]["langgraph_tool_call_count"] == 3
    assert rows[0]["custom_answer"] == "custom answer"
    assert rows[0]["langgraph_answer"] == "langgraph answer"
    assert summary["total_questions"] == 1
    assert summary["custom_success_count"] == 1
    assert summary["langgraph_success_count"] == 1
    assert summary["both_success_count"] == 1
    assert summary["same_sql_count"] == 1
    assert summary["avg_custom_latency"] == 2.0
    assert summary["avg_langgraph_latency"] == 4.0


def test_run_engine_comparison_continues_when_one_engine_fails():
    questions = [
        {
            "id": "E001",
            "question": "pertanyaan sukses",
            "expected_unit": "count",
            "category": "aggregation",
        },
        {
            "id": "E002",
            "question": "pertanyaan custom gagal",
            "expected_unit": "count",
            "category": "aggregation",
        },
    ]
    custom_workflow = FakeWorkflow(
        states_by_query={
            "pertanyaan sukses": make_state("pertanyaan sukses"),
        },
        errors_by_query={"pertanyaan custom gagal": "custom workflow error"},
    )
    langgraph_workflow = FakeWorkflow(
        states_by_query={
            "pertanyaan sukses": make_state("pertanyaan sukses"),
            "pertanyaan custom gagal": make_state(
                "pertanyaan custom gagal",
                sql="SELECT 2",
                latency=3.0,
            ),
        }
    )

    rows, summary = run_engine_comparison(
        questions,
        custom_workflow=custom_workflow,
        langgraph_workflow=langgraph_workflow,
    )

    assert custom_workflow.calls == ["pertanyaan sukses", "pertanyaan custom gagal"]
    assert langgraph_workflow.calls == ["pertanyaan sukses", "pertanyaan custom gagal"]
    assert rows[1]["custom_success"] is False
    assert rows[1]["langgraph_success"] is True
    assert rows[1]["custom_sql"] == ""
    assert rows[1]["langgraph_sql"] == "SELECT 2"
    assert rows[1]["same_sql"] is False
    assert rows[1]["same_success_status"] is False
    assert rows[1]["both_success"] is False
    assert rows[1]["custom_tool_call_count"] == 0
    assert rows[1]["error_message"] == "custom: custom workflow error"
    assert summary["total_questions"] == 2
    assert summary["custom_success_count"] == 1
    assert summary["langgraph_success_count"] == 2
    assert summary["both_success_count"] == 1


def test_write_engine_comparison_results_writes_expected_columns(tmp_path):
    output_path = tmp_path / "engine_comparison.csv"
    rows = [
        {
            "question_id": "E001",
            "question": "test",
            "custom_success": True,
            "langgraph_success": False,
            "custom_sql": "SELECT 1",
            "langgraph_sql": "",
            "custom_answer": "ok",
            "langgraph_answer": "",
            "custom_latency_total": 1.0,
            "langgraph_latency_total": "",
            "custom_tool_call_count": 2,
            "langgraph_tool_call_count": 0,
            "same_sql": False,
            "same_success_status": False,
            "both_success": False,
            "error_message": "langgraph: error",
        }
    ]

    write_engine_comparison_results(rows, output_path)

    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        loaded_rows = list(reader)

    assert reader.fieldnames == list(ENGINE_COMPARISON_COLUMNS)
    assert loaded_rows[0]["question_id"] == "E001"
    assert loaded_rows[0]["custom_success"] == "True"
    assert loaded_rows[0]["error_message"] == "langgraph: error"


def test_normalize_sql_uses_simple_whitespace_and_case_normalization():
    assert normalize_sql("SELECT  1\nFROM electric_power") == (
        "select 1 from electric_power"
    )
