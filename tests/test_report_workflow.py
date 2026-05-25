from pathlib import Path
import sys

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.graph import report_workflow
from local_agentic_analytics.graph.report_workflow import EnergyReportWorkflow


class FakeInsightAgent:
    def __init__(self):
        self.calls = []

    def generate_insight(self, chart_context):
        self.calls.append(chart_context)
        return (
            f"Grafik {chart_context['chart_title']} menunjukkan pola ringkas "
            "berdasarkan statistik yang tersedia. Untuk menyimpulkan anomali, "
            "diperlukan pembanding historis."
        )


class FailingInsightAgent:
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


def test_energy_report_workflow_generates_tex_pdf_and_log(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    figures_dir = tmp_path / "figures"
    latex_path = tmp_path / "latex" / "energy_analysis_report.tex"
    pdf_dir = tmp_path / "pdf"
    log_path = tmp_path / "experiments" / "report_generation_log.json"
    fake_agent = FakeInsightAgent()
    _create_energy_database(db_path)

    def fake_compile_pdf(tex_path, output_dir):
        pdf_path = Path(output_dir) / "energy_analysis_report.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4")
        return pdf_path

    monkeypatch.setattr(report_workflow, "compile_pdf", fake_compile_pdf)

    workflow = EnergyReportWorkflow(
        db_path=db_path,
        figures_dir=figures_dir,
        latex_output_path=latex_path,
        pdf_dir=pdf_dir,
        log_path=log_path,
        insight_agent=fake_agent,
    )

    metadata = workflow.run()

    assert metadata["success"] is True
    assert metadata["tex_success"] is True
    assert metadata["pdf_success"] is True
    assert metadata["chart_count"] == 6
    assert metadata["insight_success_count"] == 6
    assert len(fake_agent.calls) == 6
    assert latex_path.exists()
    assert (pdf_dir / "energy_analysis_report.pdf").exists()
    assert log_path.exists()

    tex_content = latex_path.read_text(encoding="utf-8")
    assert "Laporan Analisis Konsumsi Daya Listrik Rumah Tangga" in tex_content
    assert "Detailed Analysis" in tex_content


def test_energy_report_workflow_keeps_tex_when_pdf_compile_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    latex_path = tmp_path / "latex" / "energy_analysis_report.tex"
    log_path = tmp_path / "experiments" / "report_generation_log.json"
    _create_energy_database(db_path)

    def fake_compile_pdf(tex_path, output_dir):
        raise RuntimeError("compiler missing")

    monkeypatch.setattr(report_workflow, "compile_pdf", fake_compile_pdf)

    workflow = EnergyReportWorkflow(
        db_path=db_path,
        figures_dir=tmp_path / "figures",
        latex_output_path=latex_path,
        pdf_dir=tmp_path / "pdf",
        log_path=log_path,
        insight_agent=FakeInsightAgent(),
    )

    metadata = workflow.run()

    assert metadata["success"] is False
    assert metadata["tex_success"] is True
    assert metadata["pdf_success"] is False
    assert metadata["pdf_error"] == "compiler missing"
    assert latex_path.exists()
    assert log_path.exists()


def test_energy_report_workflow_continues_when_insight_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    _create_energy_database(db_path)

    monkeypatch.setattr(
        report_workflow,
        "compile_pdf",
        lambda tex_path, output_dir: Path(output_dir) / "energy_analysis_report.pdf",
    )

    workflow = EnergyReportWorkflow(
        db_path=db_path,
        figures_dir=tmp_path / "figures",
        latex_output_path=tmp_path / "latex" / "energy_analysis_report.tex",
        pdf_dir=tmp_path / "pdf",
        log_path=tmp_path / "experiments" / "report_generation_log.json",
        insight_agent=FailingInsightAgent(),
    )

    metadata = workflow.run()

    assert metadata["insight_success_count"] == 0
    assert metadata["insight_failed_count"] == 6
    assert all(record["error_message"] == "ollama unavailable" for record in metadata["insights"])


def test_energy_report_workflow_logs_failure_when_database_missing(tmp_path):
    log_path = tmp_path / "experiments" / "report_generation_log.json"
    workflow = EnergyReportWorkflow(
        db_path=tmp_path / "missing.duckdb",
        figures_dir=tmp_path / "figures",
        latex_output_path=tmp_path / "latex" / "energy_analysis_report.tex",
        pdf_dir=tmp_path / "pdf",
        log_path=log_path,
        insight_agent=FakeInsightAgent(),
    )

    metadata = workflow.run()

    assert metadata["success"] is False
    assert metadata["tex_success"] is False
    assert "DuckDB database not found" in metadata["error_message"]
    assert log_path.exists()
