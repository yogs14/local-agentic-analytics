from pathlib import Path
import json
import sys

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.graph import finance_report_workflow
from local_agentic_analytics.graph.finance_report_workflow import FinanceReportWorkflow


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


def _create_finance_database(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE stock_prices AS
            SELECT
                CAST(d AS DATE) AS date,
                ticker,
                close - 1 AS open,
                close + 2 AS high,
                close - 2 AS low,
                close,
                volume
            FROM (
                VALUES
                    ('2019-01-02', 'NVDA', 100.0, 1000000),
                    ('2019-01-03', 'NVDA', 102.0, 1100000),
                    ('2019-01-04', 'NVDA', 101.0, 1050000),
                    ('2019-01-02', 'TSLA', 60.0, 2000000),
                    ('2019-01-03', 'TSLA', 62.0, 2100000),
                    ('2019-01-04', 'TSLA', 64.0, 2050000),
                    ('2019-01-02', 'NFLX', 250.0, 500000),
                    ('2019-01-03', 'NFLX', 248.0, 520000),
                    ('2019-01-04', 'NFLX', 252.0, 510000),
                    ('2019-01-02', 'GOOGL', 1050.0, 1500000),
                    ('2019-01-03', 'GOOGL', 1060.0, 1520000),
                    ('2019-01-04', 'GOOGL', 1045.0, 1510000)
            ) AS t(d, ticker, close, volume)
            """
        )
    finally:
        con.close()


def test_finance_report_workflow_generates_tex_pdf_and_log(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    figures_dir = tmp_path / "figures"
    latex_path = tmp_path / "latex" / "finance_analysis_report.tex"
    pdf_dir = tmp_path / "pdf"
    log_path = tmp_path / "experiments" / "finance_report_generation_log.json"
    fake_agent = FakeInsightAgent()
    _create_finance_database(db_path)

    def fake_compile_pdf(tex_path, output_dir):
        pdf_path = Path(output_dir) / "finance_analysis_report.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4")
        return pdf_path

    monkeypatch.setattr(finance_report_workflow, "compile_pdf", fake_compile_pdf)

    workflow = FinanceReportWorkflow(
        db_path=db_path,
        figures_dir=figures_dir,
        latex_output_path=latex_path,
        pdf_dir=pdf_dir,
        log_path=log_path,
        insight_agent=fake_agent,
    )

    metadata = workflow.run()

    assert metadata["engine"] == "custom"
    assert metadata["domain"] == "finance"
    assert metadata["success"] is True
    assert metadata["tex_success"] is True
    assert metadata["pdf_success"] is True
    assert metadata["chart_count"] == 5
    assert metadata["insight_success_count"] == 5
    assert len(fake_agent.calls) == 5
    # Insight contexts must carry the finance domain so the prompt is finance-aware.
    assert all(call["domain"] == "finance" for call in fake_agent.calls)
    assert latex_path.exists()
    assert (pdf_dir / "finance_analysis_report.pdf").exists()
    assert log_path.exists()

    tex_content = latex_path.read_text(encoding="utf-8")
    assert "Laporan Analisis Harga Saham Harian" in tex_content
    assert "Detailed Analysis" in tex_content

    log_metadata = json.loads(log_path.read_text(encoding="utf-8"))
    assert log_metadata["domain"] == "finance"


def test_finance_report_workflow_continues_when_insight_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    _create_finance_database(db_path)

    monkeypatch.setattr(
        finance_report_workflow,
        "compile_pdf",
        lambda tex_path, output_dir: Path(output_dir) / "finance_analysis_report.pdf",
    )

    workflow = FinanceReportWorkflow(
        db_path=db_path,
        figures_dir=tmp_path / "figures",
        latex_output_path=tmp_path / "latex" / "finance_analysis_report.tex",
        pdf_dir=tmp_path / "pdf",
        log_path=tmp_path / "experiments" / "finance_report_generation_log.json",
        insight_agent=FailingInsightAgent(),
    )

    metadata = workflow.run()

    assert metadata["insight_success_count"] == 0
    assert metadata["insight_failed_count"] == 5


def test_finance_report_workflow_logs_failure_when_database_missing(tmp_path):
    log_path = tmp_path / "experiments" / "finance_report_generation_log.json"
    workflow = FinanceReportWorkflow(
        db_path=tmp_path / "missing.duckdb",
        figures_dir=tmp_path / "figures",
        latex_output_path=tmp_path / "latex" / "finance_analysis_report.tex",
        pdf_dir=tmp_path / "pdf",
        log_path=log_path,
        insight_agent=FakeInsightAgent(),
    )

    metadata = workflow.run()

    assert metadata["success"] is False
    assert metadata["tex_success"] is False
    assert "DuckDB database not found" in metadata["error_message"]
    assert log_path.exists()
