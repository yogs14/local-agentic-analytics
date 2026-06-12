from pathlib import Path
import json
import sys

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.graph import langgraph_report_workflow
from local_agentic_analytics.evaluation.audit_logger import ToolAuditLogger
from local_agentic_analytics.graph.langgraph_report_workflow import (
    LangGraphReportWorkflow,
    run_langgraph_report_workflow,
)
from local_agentic_analytics.graph.report_workflow import (
    EnergyReportWorkflow,
)


class FakeMetricsOllamaTool:
    def __init__(self, metrics):
        self.metrics = dict(metrics)

    def get_last_metrics(self):
        return dict(self.metrics)


class FakeInsightAgent:
    def __init__(self):
        self.calls = []
        self.ollama_tool = FakeMetricsOllamaTool(
            {
                "total_duration": 2.0,
                "prompt_eval_count": 24,
                "eval_count": 12,
                "eval_duration": 1.25,
            }
        )

    def generate_insight(self, chart_context):
        self.calls.append(chart_context)
        return f"Insight untuk {chart_context['chart_title']}."


class FailingInsightAgent:
    def __init__(self):
        self.ollama_tool = FakeMetricsOllamaTool({"eval_duration": 0.1})

    def generate_insight(self, chart_context):
        raise RuntimeError("ollama unavailable")


def _create_energy_database(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE electric_power AS
            SELECT * FROM (
                VALUES
                    (TIMESTAMP '2006-12-16 00:00:00', 1.0, 0.1, 240.0, 4.0, 0.0, 1.0, 17.0),
                    (TIMESTAMP '2006-12-16 01:00:00', 2.0, 0.2, 241.0, 8.0, 1.0, 2.0, 18.0),
                    (TIMESTAMP '2006-12-17 00:00:00', 3.0, 0.3, 242.0, 12.0, 2.0, 3.0, 19.0),
                    (TIMESTAMP '2006-12-17 01:00:00', 4.0, 0.4, 243.0, 16.0, 3.0, 4.0, 20.0)
            ) AS t(
                datetime,
                Global_active_power,
                Global_reactive_power,
                Voltage,
                Global_intensity,
                Sub_metering_1,
                Sub_metering_2,
                Sub_metering_3
            )
            """
        )
    finally:
        con.close()


def test_langgraph_report_workflow_imports_and_can_be_constructed(tmp_path):
    workflow = LangGraphReportWorkflow(
        db_path=tmp_path / "missing.duckdb",
        figures_dir=tmp_path / "figures",
        latex_output_path=tmp_path / "latex" / "energy_analysis_report.tex",
        pdf_dir=tmp_path / "pdf",
        log_path=tmp_path / "experiments" / "report_generation_log.json",
        audit_logger=ToolAuditLogger(log_path=tmp_path / "tool_call_audit.jsonl"),
        insight_agent=FakeInsightAgent(),
    )

    assert callable(run_langgraph_report_workflow)
    assert isinstance(workflow, LangGraphReportWorkflow)
    assert EnergyReportWorkflow is not None


def test_langgraph_report_workflow_generates_tex_pdf_log_and_metadata(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "analytics.duckdb"
    figures_dir = tmp_path / "figures"
    latex_path = tmp_path / "latex" / "energy_analysis_report.tex"
    pdf_dir = tmp_path / "pdf"
    log_path = tmp_path / "experiments" / "report_generation_log.json"
    audit_path = tmp_path / "experiments" / "tool_call_audit.jsonl"
    fake_agent = FakeInsightAgent()
    _create_energy_database(db_path)

    def fake_compile_pdf(tex_path, output_dir):
        pdf_path = Path(output_dir) / "energy_analysis_report.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4")
        return pdf_path

    monkeypatch.setattr(langgraph_report_workflow, "compile_pdf", fake_compile_pdf)

    workflow = LangGraphReportWorkflow(
        db_path=db_path,
        figures_dir=figures_dir,
        latex_output_path=latex_path,
        pdf_dir=pdf_dir,
        log_path=log_path,
        audit_logger=ToolAuditLogger(log_path=audit_path),
        insight_agent=fake_agent,
    )

    metadata = workflow.run()

    assert metadata["engine"] == "langgraph"
    assert metadata["success"] is True
    assert metadata["tex_success"] is True
    assert metadata["pdf_success"] is True
    assert metadata["error_message"] == ""
    assert metadata["pdf_error"] == ""
    assert metadata["tex_path"] == str(latex_path)
    assert metadata["pdf_path"] == str(pdf_dir / "energy_analysis_report.pdf")
    assert metadata["log_path"] == str(log_path)
    assert metadata["chart_count"] == 6
    assert metadata["insight_success_count"] == 6
    assert metadata["insight_failed_count"] == 0
    assert len(metadata["charts"]) == 6
    assert len(metadata["insights"]) == 6
    assert len(fake_agent.calls) == 6
    assert "generate_charts" in metadata["latency"]
    required_tools = {
        "report.generate_charts",
        "report.build_chart_contexts",
        "ollama.generate_insights",
        "report.build_report",
        "report.render_latex",
        "report.compile_pdf",
        "report.write_log",
    }
    state_tools = {call["tool"] for call in metadata["tool_calls"]}
    assert required_tools <= state_tools
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
    assert all(required_keys <= set(call) for call in metadata["tool_calls"])
    assert latex_path.exists()
    assert (pdf_dir / "energy_analysis_report.pdf").exists()
    assert log_path.exists()
    assert audit_path.exists()

    audit_events = [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(audit_events) == len(metadata["tool_calls"])
    assert required_tools <= {event["tool"] for event in audit_events}
    insight_event = next(
        event for event in audit_events if event["tool"] == "ollama.generate_insights"
    )
    assert insight_event["metadata"]["eval_count"] == 12


def test_langgraph_report_workflow_keeps_tex_when_pdf_compile_fails(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "analytics.duckdb"
    latex_path = tmp_path / "latex" / "energy_analysis_report.tex"
    log_path = tmp_path / "experiments" / "report_generation_log.json"
    _create_energy_database(db_path)

    def fake_compile_pdf(tex_path, output_dir):
        raise RuntimeError("compiler missing")

    monkeypatch.setattr(langgraph_report_workflow, "compile_pdf", fake_compile_pdf)

    workflow = LangGraphReportWorkflow(
        db_path=db_path,
        figures_dir=tmp_path / "figures",
        latex_output_path=latex_path,
        pdf_dir=tmp_path / "pdf",
        log_path=log_path,
        audit_logger=ToolAuditLogger(log_path=tmp_path / "tool_call_audit.jsonl"),
        insight_agent=FakeInsightAgent(),
    )

    metadata = workflow.run()

    assert metadata["success"] is False
    assert metadata["tex_success"] is True
    assert metadata["pdf_success"] is False
    assert metadata["pdf_error"] == "compiler missing"
    assert metadata["error_message"] == ""
    assert latex_path.exists()
    assert log_path.exists()


def test_langgraph_report_workflow_returns_failure_when_database_is_missing(tmp_path):
    workflow = LangGraphReportWorkflow(
        db_path=tmp_path / "missing.duckdb",
        figures_dir=tmp_path / "figures",
        latex_output_path=tmp_path / "latex" / "energy_analysis_report.tex",
        pdf_dir=tmp_path / "pdf",
        log_path=tmp_path / "experiments" / "report_generation_log.json",
        audit_logger=ToolAuditLogger(log_path=tmp_path / "tool_call_audit.jsonl"),
        insight_agent=FakeInsightAgent(),
    )

    metadata = workflow.run()

    assert metadata["success"] is False
    assert metadata["tex_success"] is False
    assert metadata["chart_count"] == 0
    assert "DuckDB database not found" in metadata["error_message"]


def test_langgraph_report_workflow_continues_when_insight_fails(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "analytics.duckdb"
    _create_energy_database(db_path)
    monkeypatch.setattr(
        langgraph_report_workflow,
        "compile_pdf",
        lambda tex_path, output_dir: Path(output_dir) / "energy_analysis_report.pdf",
    )

    workflow = LangGraphReportWorkflow(
        db_path=db_path,
        figures_dir=tmp_path / "figures",
        latex_output_path=tmp_path / "latex" / "energy_analysis_report.tex",
        pdf_dir=tmp_path / "pdf",
        log_path=tmp_path / "experiments" / "report_generation_log.json",
        audit_logger=ToolAuditLogger(log_path=tmp_path / "tool_call_audit.jsonl"),
        insight_agent=FailingInsightAgent(),
    )

    metadata = workflow.run()

    assert metadata["insight_success_count"] == 0
    assert metadata["insight_failed_count"] == 6
    assert all(record["success"] is False for record in metadata["insights"])
    assert all(
        "Insight otomatis tidak dapat dibuat" in record["insight"]
        for record in metadata["insights"]
    )
