from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.report_engine_comparison import (
    run_report_engine_comparison,
    write_report_engine_comparison_result,
)


class FakeReportWorkflow:
    def __init__(self, metadata: dict | None = None, error: str | None = None):
        self.metadata = metadata or {}
        self.error = error
        self.calls = 0

    def run(self) -> dict:
        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        return dict(self.metadata)


def test_report_engine_comparison_compares_metadata_counts_and_statuses():
    custom_workflow = FakeReportWorkflow(
        {
            "success": True,
            "tex_success": True,
            "pdf_success": False,
            "chart_count": 6,
            "insight_success_count": 4,
            "latency": {},
            "tool_calls": [],
            "pdf_error": "pdflatex not found",
        }
    )
    langgraph_workflow = FakeReportWorkflow(
        {
            "success": False,
            "tex_success": True,
            "pdf_success": False,
            "chart_count": 6,
            "insight_success_count": 5,
            "latency": {
                "generate_charts": 1.0,
                "generate_insights": 2.5,
            },
            "tool_calls": [
                {"tool": "report.generate_charts"},
                {"tool": "ollama.generate_insights"},
            ],
            "error_message": "",
            "pdf_error": "pdflatex not found",
        }
    )

    comparison = run_report_engine_comparison(
        custom_workflow=custom_workflow,
        langgraph_workflow=langgraph_workflow,
    )

    assert custom_workflow.calls == 1
    assert langgraph_workflow.calls == 1
    assert comparison["custom_success"] is True
    assert comparison["langgraph_success"] is False
    assert comparison["custom_tex_success"] is True
    assert comparison["langgraph_tex_success"] is True
    assert comparison["custom_pdf_success"] is False
    assert comparison["langgraph_pdf_success"] is False
    assert comparison["custom_chart_count"] == 6
    assert comparison["langgraph_chart_count"] == 6
    assert comparison["custom_insight_success_count"] == 4
    assert comparison["langgraph_insight_success_count"] == 5
    assert comparison["custom_latency_total"] is None
    assert comparison["langgraph_latency_total"] == 3.5
    assert comparison["custom_tool_call_count"] == 0
    assert comparison["langgraph_tool_call_count"] == 2
    assert comparison["same_chart_count"] is True
    assert comparison["same_tex_success"] is True
    assert comparison["same_pdf_success"] is True
    assert comparison["custom_error"] == "pdflatex not found"
    assert comparison["langgraph_error"] == "pdflatex not found"


def test_report_engine_comparison_runs_langgraph_when_custom_fails():
    custom_workflow = FakeReportWorkflow(error="custom crashed")
    langgraph_workflow = FakeReportWorkflow(
        {
            "success": True,
            "tex_success": True,
            "pdf_success": True,
            "chart_count": 6,
            "insight_success_count": 6,
            "latency": {"total": 7.0},
            "tool_calls": [{"tool": "report.generate_charts"}],
        }
    )

    comparison = run_report_engine_comparison(
        custom_workflow=custom_workflow,
        langgraph_workflow=langgraph_workflow,
    )

    assert custom_workflow.calls == 1
    assert langgraph_workflow.calls == 1
    assert comparison["custom_success"] is False
    assert comparison["langgraph_success"] is True
    assert comparison["custom_tex_success"] is False
    assert comparison["langgraph_tex_success"] is True
    assert comparison["custom_error"] == "custom crashed"
    assert comparison["langgraph_latency_total"] == 7.0
    assert comparison["langgraph_tool_call_count"] == 1


def test_write_report_engine_comparison_result_writes_json(tmp_path):
    output_path = tmp_path / "report_engine_comparison.json"
    comparison = {
        "timestamp": "2026-06-12T00:00:00+00:00",
        "custom_success": True,
        "langgraph_success": True,
        "custom_tex_success": True,
        "langgraph_tex_success": True,
        "custom_pdf_success": False,
        "langgraph_pdf_success": False,
        "custom_chart_count": 6,
        "langgraph_chart_count": 6,
        "custom_insight_success_count": 6,
        "langgraph_insight_success_count": 6,
        "custom_latency_total": None,
        "langgraph_latency_total": 3.0,
        "custom_tool_call_count": 0,
        "langgraph_tool_call_count": 7,
        "same_chart_count": True,
        "same_tex_success": True,
        "same_pdf_success": True,
        "custom_error": "",
        "langgraph_error": "",
    }

    written_path = write_report_engine_comparison_result(comparison, output_path)

    assert written_path == output_path
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == comparison
