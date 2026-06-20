from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics import cli
from local_agentic_analytics.core.state import AnalyticsState


class FakeWorkflow:
    def __init__(self, domain: str = "energy"):
        self.domain = domain

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
            "engine": "custom",
            "success": True,
            "tex_success": True,
            "pdf_success": True,
            "tex_path": "reports/latex/energy_analysis_report.tex",
            "pdf_path": "reports/pdf/energy_analysis_report.pdf",
            "log_path": "reports/experiments/report_generation_log.json",
            "chart_count": 6,
            "latency": {"total": 0.25},
            "tool_calls": [
                {
                    "tool": "report.generate_charts",
                    "status": "success",
                    "latency_seconds": 0.02,
                }
            ],
            "pdf_error": "",
            "error_message": "",
        }


def fake_langgraph_report_runner(domain: str) -> dict:
    return {
        "engine": "langgraph",
        "success": False,
        "tex_success": True,
        "pdf_success": False,
        "tex_path": "reports/latex/energy_analysis_report.tex",
        "pdf_path": "",
        "log_path": "reports/experiments/report_generation_log.json",
        "chart_count": 6,
        "latency": {"generate_charts": 0.12},
        "tool_calls": [
            {
                "tool": "report.compile_pdf",
                "status": "error",
                "latency_seconds": 0.03,
            }
        ],
        "pdf_error": "pdflatex not found",
        "error_message": "",
    }


def test_cli_ask_prints_expected_sections(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "analytics.duckdb"
    db_path.write_bytes(b"duckdb placeholder")
    monkeypatch.setattr(cli, "get_default_duckdb_path", lambda: db_path)
    monkeypatch.setattr(cli, "SequentialAnalyticsWorkflow", FakeWorkflow)
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


def test_cli_ask_can_select_langgraph_engine(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "analytics.duckdb"
    db_path.write_bytes(b"duckdb placeholder")
    created = []
    monkeypatch.setattr(cli, "get_default_duckdb_path", lambda: db_path)
    monkeypatch.setattr(
        cli,
        "_LANGGRAPH_WORKFLOW_RUNNER",
        lambda user_query, domain="energy": created.append((user_query, domain))
        or FakeWorkflow().run(user_query),
    )
    monkeypatch.setattr(cli, "append_run_log", lambda log: None)

    exit_code = cli.main(["ask", "--engine", "langgraph", "Berapa", "nilainya?"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert created == [("Berapa nilainya?", "energy")]
    assert "Final answer:" in output
    assert "Status: sukses" in output


def test_cli_ask_run_log_includes_engine(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    db_path.write_bytes(b"duckdb placeholder")
    logs = []
    monkeypatch.setattr(cli, "get_default_duckdb_path", lambda: db_path)
    monkeypatch.setattr(
        cli,
        "_LANGGRAPH_WORKFLOW_RUNNER",
        lambda user_query, domain="energy": FakeWorkflow(domain=domain).run(user_query),
    )
    monkeypatch.setattr(cli, "append_run_log", logs.append)

    exit_code = cli.main(["ask", "--engine", "langgraph", "Berapa", "nilainya?"])

    assert exit_code == 0
    assert logs[0]["engine"] == "langgraph"
    assert logs[0]["domain"] == "energy"


def test_cli_ask_passes_domain_to_custom_workflow(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    db_path.write_bytes(b"duckdb placeholder")
    created_domains = []

    def workflow_factory(domain="energy"):
        created_domains.append(domain)
        return FakeWorkflow(domain=domain)

    monkeypatch.setattr(cli, "get_default_duckdb_path", lambda: db_path)
    monkeypatch.setattr(cli, "SequentialAnalyticsWorkflow", workflow_factory)
    monkeypatch.setattr(cli, "append_run_log", lambda log: None)

    exit_code = cli.main(
        ["ask", "--domain", "finance", "Berapa", "rata-rata", "harga", "penutupan?"]
    )

    assert exit_code == 0
    assert created_domains == ["finance"]


def test_cli_ask_passes_domain_to_langgraph_workflow(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    db_path.write_bytes(b"duckdb placeholder")
    calls = []

    def fake_runner(user_query, domain="energy"):
        calls.append((user_query, domain))
        return FakeWorkflow(domain=domain).run(user_query)

    monkeypatch.setattr(cli, "get_default_duckdb_path", lambda: db_path)
    monkeypatch.setattr(cli, "_LANGGRAPH_WORKFLOW_RUNNER", fake_runner)
    monkeypatch.setattr(cli, "append_run_log", lambda log: None)

    exit_code = cli.main(
        [
            "ask",
            "--domain",
            "finance",
            "--engine",
            "langgraph",
            "Berapa rata-rata harga penutupan?",
        ]
    )

    assert exit_code == 0
    assert calls == [("Berapa rata-rata harga penutupan?", "finance")]


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
    assert "- engine: custom" in output
    assert "- success: True" in output
    assert "- tex_success: True" in output
    assert "- pdf_success: True" in output
    assert "- tex_path: reports/latex/energy_analysis_report.tex" in output
    assert "- pdf_path: reports/pdf/energy_analysis_report.pdf" in output
    assert "- log_path: reports/experiments/report_generation_log.json" in output
    assert "- chart_count: 6" in output
    assert "Latency:" in output
    assert "- total: 0.250s" in output
    assert "Tool calls:" in output
    assert "- report.generate_charts: success, 0.020s" in output


def test_cli_report_energy_custom_engine_prints_report_metadata(
    tmp_path,
    monkeypatch,
    capsys,
):
    db_path = tmp_path / "analytics.duckdb"
    db_path.write_bytes(b"duckdb placeholder")
    monkeypatch.setattr(cli, "DEFAULT_REPORT_DB_PATH", db_path)
    monkeypatch.setattr(cli, "EnergyReportWorkflow", FakeReportWorkflow)

    exit_code = cli.main(["report", "energy", "--engine", "custom"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "- engine: custom" in output
    assert "- tex_success: True" in output


def test_cli_report_energy_langgraph_engine_prints_report_metadata(
    tmp_path,
    monkeypatch,
    capsys,
):
    db_path = tmp_path / "analytics.duckdb"
    db_path.write_bytes(b"duckdb placeholder")
    monkeypatch.setattr(cli, "DEFAULT_REPORT_DB_PATH", db_path)
    monkeypatch.setattr(
        cli,
        "_LANGGRAPH_REPORT_WORKFLOW_RUNNER",
        fake_langgraph_report_runner,
    )

    exit_code = cli.main(["report", "energy", "--engine", "langgraph"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "- engine: langgraph" in output
    assert "- success: False" in output
    assert "- tex_success: True" in output
    assert "- pdf_success: False" in output
    assert "- report.compile_pdf: error, 0.030s" in output
    assert "pdf_error: pdflatex not found" in output


def test_report_metadata_normalizer_adds_custom_defaults():
    metadata = cli.normalize_report_metadata(
        {
            "tex_success": True,
            "chart_count": 6,
        },
        engine="custom",
    )

    assert metadata["engine"] == "custom"
    assert metadata["log_path"] == ""
    assert metadata["latency"] == {}
    assert metadata["tool_calls"] == []


def test_print_report_metadata_shows_empty_latency_and_tool_calls(capsys):
    metadata = cli.normalize_report_metadata(
        {
            "success": True,
            "tex_success": True,
            "pdf_success": False,
            "tex_path": "reports/latex/energy_analysis_report.tex",
            "chart_count": 6,
        },
        engine="custom",
    )

    cli.print_report_metadata(metadata)

    output = capsys.readouterr().out
    assert "- engine: custom" in output
    assert "- tex_path: reports/latex/energy_analysis_report.tex" in output
    assert "Latency:\n{}" in output
    assert "Tool calls:\n[]" in output


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
