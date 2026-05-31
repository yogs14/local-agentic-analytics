from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics import cli
from local_agentic_analytics.core.state import AnalyticsState


class FakeWorkflow:
    def run(self, user_query: str) -> AnalyticsState:
        return AnalyticsState(
            user_query=user_query,
            generated_sql="SELECT 1 AS value",
            repaired_sql=None,
            sql_result={
                "row_count": 1,
                "columns": ["value"],
                "rows": [{"value": 1}],
                "truncated": False,
            },
            final_answer="Nilainya adalah 1.",
            latency={"total": 0.1},
            tool_calls=[
                {
                    "tool": "duckdb.schema",
                    "status": "success",
                    "latency_seconds": 0.01,
                }
            ],
            success=True,
        )


class FakeReportWorkflow:
    def __init__(self, db_path):
        self.db_path = db_path

    def run(self) -> dict:
        return {
            "tex_path": "reports/latex/energy_analysis_report.tex",
            "pdf_path": "reports/pdf/energy_analysis_report.pdf",
            "chart_count": 6,
            "pdf_success": True,
            "tex_success": True,
            "pdf_error": "",
            "error_message": "",
        }


def test_cli_ask_prints_expected_sections(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "analytics.duckdb"
    db_path.write_bytes(b"duckdb placeholder")
    monkeypatch.setattr(cli, "get_default_duckdb_path", lambda: db_path)
    monkeypatch.setattr(cli, "SequentialAnalyticsWorkflow", lambda: FakeWorkflow())
    monkeypatch.setattr(cli, "append_run_log", lambda log: None)

    exit_code = cli.main(["ask", "Berapa", "nilainya?"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Generated SQL:" in output
    assert "Result:" in output
    assert "Final answer:" in output
    assert "Latency:" in output
    assert "Tool calls:" in output
    assert "- duckdb.schema: success, 0.010s" in output
    assert "Status: sukses" in output


def test_cli_ask_prints_clear_message_when_database_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_default_duckdb_path", lambda: tmp_path / "missing.duckdb")

    exit_code = cli.main(["ask", "Berapa rata-ratanya?"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert cli.MISSING_DATABASE_MESSAGE in output


def test_cli_report_energy_prints_report_metadata(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "analytics.duckdb"
    db_path.write_bytes(b"duckdb placeholder")
    monkeypatch.setattr(cli, "DEFAULT_REPORT_DB_PATH", db_path)
    monkeypatch.setattr(cli, "EnergyReportWorkflow", FakeReportWorkflow)

    exit_code = cli.main(["report", "energy"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "LaTeX path:" in output
    assert "PDF path:" in output
    assert "Jumlah chart: 6" in output
    assert "Status compile: sukses" in output


def test_cli_report_energy_prints_clear_message_when_database_missing(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(cli, "DEFAULT_REPORT_DB_PATH", tmp_path / "missing.duckdb")

    exit_code = cli.main(["report", "energy"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert cli.MISSING_DATABASE_MESSAGE in output
