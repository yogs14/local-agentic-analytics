from pathlib import Path
import sys

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.profile_generator import (
    generate_profile_from_duckdb,
)
from local_agentic_analytics.graph import generic_report_workflow
from local_agentic_analytics.graph.generic_report_workflow import (
    GenericReportWorkflow,
    generate_generic_charts,
)
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool


class FakeInsightAgent:
    def __init__(self):
        self.calls = []

    def generate_insight(self, chart_context):
        self.calls.append(chart_context)
        return (
            f"Grafik {chart_context['chart_title']} menunjukkan pola ringkas. "
            "Untuk menyimpulkan anomali, diperlukan pembanding historis."
        )


class FailingInsightAgent:
    def generate_insight(self, chart_context):
        raise RuntimeError("ollama unavailable")


def _create_generic_database(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE TABLE data (tanggal DATE, kategori VARCHAR, "
            "nilai DOUBLE, jumlah INTEGER)"
        )
        for i in range(20):
            con.execute(
                "INSERT INTO data VALUES (?, ?, ?, ?)",
                [
                    f"2024-{(i % 12) + 1:02d}-01",
                    f"Kategori {i % 4}",
                    float((i + 1) * 10),
                    int(i + 1),
                ],
            )
    finally:
        con.close()


def _profile_for(db_path: Path, table_name: str = "data"):
    tool = DuckDBTool(str(db_path))
    tool.connect()
    try:
        return generate_profile_from_duckdb(tool, table_name)
    finally:
        tool.close()


def _fake_compile_pdf(table_name: str):
    def _compile(tex_path, output_dir):
        pdf_path = Path(output_dir) / f"{table_name}_report.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4")
        return pdf_path

    return _compile


def test_generic_workflow_produces_tex_and_pdf(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    _create_generic_database(db_path)
    profile = _profile_for(db_path)
    monkeypatch.setattr(
        generic_report_workflow, "compile_pdf", _fake_compile_pdf("data")
    )

    tool = DuckDBTool(str(db_path))
    tool.connect()
    try:
        workflow = GenericReportWorkflow(
            duckdb_tool=tool,
            table_name="data",
            profile=profile,
            output_dir=tmp_path / "out",
            insight_agent=FakeInsightAgent(),
        )
        metadata = workflow.run()
    finally:
        tool.close()

    assert metadata["tex_success"] is True
    assert metadata["pdf_success"] is True
    assert metadata["chart_count"] >= 2
    assert Path(metadata["tex_path"]).exists()


def test_generic_workflow_insight_failure_does_not_abort(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    _create_generic_database(db_path)
    profile = _profile_for(db_path)
    monkeypatch.setattr(
        generic_report_workflow, "compile_pdf", _fake_compile_pdf("data")
    )

    tool = DuckDBTool(str(db_path))
    tool.connect()
    try:
        workflow = GenericReportWorkflow(
            duckdb_tool=tool,
            table_name="data",
            profile=profile,
            output_dir=tmp_path / "out",
            insight_agent=FailingInsightAgent(),
        )
        metadata = workflow.run()
    finally:
        tool.close()

    assert metadata["tex_success"] is True
    assert metadata["insight_success_count"] == 0
    assert metadata["insight_failed_count"] == metadata["chart_count"]


def test_generic_workflow_pdf_failure_does_not_abort(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    _create_generic_database(db_path)
    profile = _profile_for(db_path)

    def failing_compile(tex_path, output_dir):
        raise RuntimeError("compiler missing")

    monkeypatch.setattr(generic_report_workflow, "compile_pdf", failing_compile)

    tool = DuckDBTool(str(db_path))
    tool.connect()
    try:
        workflow = GenericReportWorkflow(
            duckdb_tool=tool,
            table_name="data",
            profile=profile,
            output_dir=tmp_path / "out",
            insight_agent=FakeInsightAgent(),
        )
        metadata = workflow.run()
    finally:
        tool.close()

    assert metadata["tex_success"] is True
    assert metadata["pdf_success"] is False
    assert metadata["pdf_error"] != ""


def test_generic_workflow_no_datetime_column(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE TABLE data (produk VARCHAR, harga DOUBLE, kuantitas INTEGER)"
        )
        for i in range(15):
            con.execute(
                "INSERT INTO data VALUES (?, ?, ?)",
                [f"Produk {i % 3}", float(i * 5), int(i)],
            )
    finally:
        con.close()
    profile = _profile_for(db_path)
    monkeypatch.setattr(
        generic_report_workflow, "compile_pdf", _fake_compile_pdf("data")
    )

    tool = DuckDBTool(str(db_path))
    tool.connect()
    try:
        charts = generate_generic_charts(tool, "data", profile, tmp_path / "out")
    finally:
        tool.close()

    chart_ids = {chart["chart_id"] for chart in charts}
    assert "timeseries" not in chart_ids
    assert any(cid.startswith("distribution") for cid in chart_ids)
    assert "column_averages" in chart_ids


def test_generic_workflow_missing_database(tmp_path):
    tool = DuckDBTool(":memory:")
    tool.connect()
    tool.execute_query("CREATE TABLE other (x INTEGER)")
    profile = generate_profile_from_duckdb(tool, "other")
    # Build a profile that points at a table that does not exist.
    from dataclasses import replace

    missing_profile = replace(profile, table_name="ghost")
    try:
        workflow = GenericReportWorkflow(
            duckdb_tool=tool,
            table_name="ghost",
            profile=missing_profile,
            output_dir=tmp_path / "out",
        )
        metadata = workflow.run()
    finally:
        tool.close()

    assert metadata["success"] is False
    assert metadata["error_message"] != ""


def test_generate_generic_charts_creates_png_files(tmp_path):
    db_path = tmp_path / "analytics.duckdb"
    _create_generic_database(db_path)
    profile = _profile_for(db_path)

    tool = DuckDBTool(str(db_path))
    tool.connect()
    try:
        charts = generate_generic_charts(tool, "data", profile, tmp_path / "out")
    finally:
        tool.close()

    assert charts
    for chart in charts:
        assert Path(chart["path"]).exists()
        assert chart["stats"]


def test_generate_generic_charts_handles_null_heavy_column(tmp_path):
    db_path = tmp_path / "analytics.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE data (a DOUBLE, b DOUBLE)")
        for i in range(20):
            value = float(i) if i % 10 == 0 else None
            con.execute("INSERT INTO data VALUES (?, ?)", [value, float(i)])
    finally:
        con.close()
    profile = _profile_for(db_path)

    tool = DuckDBTool(str(db_path))
    tool.connect()
    try:
        charts = generate_generic_charts(tool, "data", profile, tmp_path / "out")
    finally:
        tool.close()

    # Should not raise; charts list is returned (possibly with skips).
    assert isinstance(charts, list)
