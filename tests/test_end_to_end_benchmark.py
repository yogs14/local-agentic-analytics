from pathlib import Path
import csv
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.end_to_end_benchmark import (
    END_TO_END_BENCHMARK_COLUMNS,
    run_end_to_end_benchmark,
    write_end_to_end_benchmark_rows,
    write_end_to_end_benchmark_summary,
)


class FakeQAWorkflow:
    def __init__(self, states_by_question=None, errors_by_question=None):
        self.states_by_question = states_by_question or {}
        self.errors_by_question = errors_by_question or {}
        self.calls = []

    def run(self, user_query: str) -> AnalyticsState:
        self.calls.append(user_query)
        if user_query in self.errors_by_question:
            raise RuntimeError(self.errors_by_question[user_query])
        return self.states_by_question[user_query]


class FakeReportWorkflow:
    def __init__(self, metadata=None, error=None):
        self.metadata = metadata or {}
        self.error = error
        self.calls = 0

    def run(self) -> dict:
        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        return dict(self.metadata)


class FakeDuckDBTool:
    def execute_query(self, sql: str) -> pd.DataFrame:
        if "bad" in sql:
            raise RuntimeError("bad sql")
        return pd.DataFrame({"value": [1.0]})


def make_state(
    question: str,
    success: bool = True,
    sql: str = "SELECT 1.0 AS value",
    repaired_sql: str | None = None,
    latency: float = 1.0,
    tool_call_count: int = 1,
    answer: str = "Nilainya 1.",
    error_message: str = "",
) -> AnalyticsState:
    return AnalyticsState(
        user_query=question,
        generated_sql=sql,
        repaired_sql=repaired_sql,
        final_answer=answer if success else None,
        sql_result={"rows": [{"value": 1.0}]} if success else None,
        success=success,
        error_message=error_message,
        latency={"total": latency},
        tool_calls=[{"tool": f"tool.{index}"} for index in range(tool_call_count)],
    )


def test_end_to_end_benchmark_runs_all_engines_and_uses_gold_sql(tmp_path):
    question = {
        "id": "E001",
        "question": "Berapa nilainya?",
        "expected_unit": "count",
        "category": "aggregation",
    }
    gold_sql_path = tmp_path / "E001.sql"
    gold_sql_path.write_text("SELECT 1.0 AS value", encoding="utf-8")
    gold_questions = [
        {
            "id": "E001",
            "question": "Berapa nilainya?",
            "gold_sql_file": str(gold_sql_path),
            "expected_unit": "count",
        }
    ]
    custom_qa = FakeQAWorkflow(
        {"Berapa nilainya?": make_state("Berapa nilainya?", latency=2.0)}
    )
    langgraph_qa = FakeQAWorkflow(
        {
            "Berapa nilainya?": make_state(
                "Berapa nilainya?",
                repaired_sql="SELECT 1.0 AS value",
                latency=4.0,
                tool_call_count=3,
            )
        }
    )
    custom_report = FakeReportWorkflow(
        {
            "success": True,
            "tex_success": True,
            "pdf_success": False,
            "chart_count": 6,
            "latency": {},
            "tool_calls": [],
        }
    )
    langgraph_report = FakeReportWorkflow(
        {
            "success": True,
            "tex_success": True,
            "pdf_success": True,
            "chart_count": 6,
            "latency": {"generate_charts": 1.0, "compile_pdf": 2.0},
            "tool_calls": [{"tool": "report.generate_charts"}],
        }
    )
    eval_scores = iter(
        [
            {"final_report_score": 0.7},
            {"final_report_score": 0.9},
        ]
    )

    rows, summary = run_end_to_end_benchmark(
        questions=[question],
        custom_qa_workflow=custom_qa,
        langgraph_qa_workflow=langgraph_qa,
        custom_report_workflow=custom_report,
        langgraph_report_workflow=langgraph_report,
        duckdb_tool=FakeDuckDBTool(),
        gold_questions=gold_questions,
        report_evaluator=lambda: next(eval_scores),
    )

    assert custom_qa.calls == ["Berapa nilainya?"]
    assert langgraph_qa.calls == ["Berapa nilainya?"]
    assert custom_report.calls == 1
    assert langgraph_report.calls == 1
    assert len(rows) == 4
    assert rows[0]["workflow_type"] == "qa"
    assert rows[0]["engine"] == "custom"
    assert rows[0]["gold_numeric_match"] is True
    assert rows[1]["repair_used"] is True
    assert rows[2]["workflow_type"] == "report"
    assert rows[2]["report_eval_score"] == 0.7
    assert rows[3]["report_pdf_success"] is True
    assert rows[3]["latency_total"] == 3.0
    assert summary["custom_success_rate"] == 1.0
    assert summary["langgraph_success_rate"] == 1.0
    assert summary["custom_avg_latency"] == 2.0
    assert summary["langgraph_avg_latency"] == 3.5
    assert summary["report_pdf_success"] == {
        "custom": False,
        "langgraph": True,
    }
    assert summary["report_eval_score"] == {
        "custom": 0.7,
        "langgraph": 0.9,
    }
    assert summary["gold_numeric_compared_count"] == 2
    assert summary["gold_numeric_match_rate"] == 1.0


def test_end_to_end_benchmark_continues_when_question_fails():
    question = {
        "id": "E001",
        "question": "Pertanyaan gagal",
        "expected_unit": "count",
        "category": "aggregation",
    }
    custom_qa = FakeQAWorkflow(errors_by_question={"Pertanyaan gagal": "custom failed"})
    langgraph_qa = FakeQAWorkflow(
        {"Pertanyaan gagal": make_state("Pertanyaan gagal")}
    )

    rows, summary = run_end_to_end_benchmark(
        questions=[question],
        custom_qa_workflow=custom_qa,
        langgraph_qa_workflow=langgraph_qa,
        custom_report_workflow=FakeReportWorkflow({"success": False}),
        langgraph_report_workflow=FakeReportWorkflow({"success": False}),
        duckdb_tool=FakeDuckDBTool(),
        gold_questions=[],
        report_evaluator=lambda: {"final_report_score": 0.0},
    )

    assert rows[0]["success"] is False
    assert rows[0]["error_message"] == "custom failed"
    assert rows[1]["success"] is True
    assert summary["custom_success_rate"] == 0.0
    assert summary["langgraph_success_rate"] == 0.5


def test_write_end_to_end_benchmark_outputs_csv_and_summary(tmp_path):
    rows = [
        {
            "workflow_type": "qa",
            "engine": "custom",
            "question_id": "E001",
            "question": "test",
            "success": True,
            "generated_sql": "SELECT 1",
            "repaired_sql": "",
            "final_answer": "ok",
            "latency_total": 1.0,
            "tool_call_count": 2,
            "repair_used": False,
            "error_message": "",
        }
    ]
    summary = {
        "custom_success_rate": 1.0,
        "langgraph_success_rate": 0.0,
        "custom_avg_latency": 1.0,
        "langgraph_avg_latency": 0.0,
        "report_pdf_success": {"custom": False, "langgraph": False},
        "report_eval_score": {"custom": None, "langgraph": None},
        "avg_tool_call_count": 2.0,
    }

    csv_path = write_end_to_end_benchmark_rows(
        rows,
        output_path=tmp_path / "benchmark.csv",
    )
    summary_path = write_end_to_end_benchmark_summary(
        summary,
        output_path=tmp_path / "summary.json",
    )

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        loaded_rows = list(reader)

    assert reader.fieldnames == list(END_TO_END_BENCHMARK_COLUMNS)
    assert loaded_rows[0]["workflow_type"] == "qa"
    assert loaded_rows[0]["generated_sql"] == "SELECT 1"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
