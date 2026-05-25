from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.reporting.latex_builder import (
    escape_latex,
    latex_label,
    normalize_figure_path,
    render_latex_report,
)
from local_agentic_analytics.reporting.pdf_compiler import compile_pdf
from local_agentic_analytics.reporting.report_schema import (
    AnalysisReport,
    ReportFigure,
    ReportSection,
)


TEMPLATE_PATH = (
    PROJECT_ROOT
    / "src"
    / "local_agentic_analytics"
    / "reporting"
    / "templates"
    / "analysis_report.tex.j2"
)


def test_escape_latex_handles_common_special_characters():
    assert escape_latex("a_b & 100% #1") == r"a\_b \& 100\% \#1"


def test_normalize_figure_path_uses_forward_slashes(tmp_path):
    figure_path = tmp_path / "reports" / "figures" / "chart_one.png"
    base_dir = tmp_path / "reports" / "latex"

    normalized = normalize_figure_path(figure_path, base_dir)

    assert normalized == "../figures/chart_one.png"
    assert "\\" not in normalized


def test_latex_label_removes_unsafe_characters():
    assert latex_label("daily_active_power trend") == "daily-active-power-trend"


def test_render_latex_report_writes_escaped_tex(tmp_path):
    figure_path = tmp_path / "reports" / "figures" / "chart_one.png"
    output_path = tmp_path / "reports" / "latex" / "sample_report.tex"
    report = AnalysisReport(
        title="Energy_Usage & 100%",
        author="local_agentic_analytics",
        abstract="Ringkasan 100% aman.",
        introduction="Dataset memakai kolom Global_active_power.",
        methodology="DuckDB & matplotlib.",
        sections=[
            ReportSection(
                title="Grafik #1",
                content="Nilai rata-rata memakai satuan kW.",
                figures=[
                    ReportFigure(
                        figure_id="chart_one",
                        title="Chart One",
                        path=str(figure_path),
                        caption="Caption dengan karakter _ dan %",
                    )
                ],
            )
        ],
        synthesis="Sintesis singkat.",
        conclusion="Kesimpulan singkat.",
    )

    rendered_path = render_latex_report(report, TEMPLATE_PATH, output_path)
    content = rendered_path.read_text(encoding="utf-8")

    assert r"Energy\_Usage \& 100\%" in content
    assert r"local\_agentic\_analytics" in content
    assert r"Global\_active\_power" in content
    assert r"\detokenize{../figures/chart_one.png}" in content
    assert r"\label{fig:chart-one}" in content


def test_report_schema_rejects_empty_required_fields():
    with pytest.raises(ValueError, match="title must not be empty"):
        ReportSection(title="", content="content")


def test_compile_pdf_uses_tectonic_when_available(tmp_path, monkeypatch):
    tex_path = tmp_path / "sample_report.tex"
    output_dir = tmp_path / "pdf"
    tex_path.write_text(r"\documentclass{article}\begin{document}Hi\end{document}")

    def fake_which(command):
        if command == "tectonic":
            return "C:/tools/tectonic.exe"
        return None

    def fake_run(command, capture_output, text, timeout, check, cwd):
        assert command[0] == "C:/tools/tectonic.exe"
        assert "--outdir" in command
        assert cwd == str(tex_path.parent)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sample_report.pdf").write_bytes(b"%PDF-1.4")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)

    pdf_path = compile_pdf(tex_path, output_dir)

    assert pdf_path == output_dir / "sample_report.pdf"
    assert pdf_path.exists()


def test_compile_pdf_falls_back_to_pdflatex(tmp_path, monkeypatch):
    tex_path = tmp_path / "sample_report.tex"
    output_dir = tmp_path / "pdf"
    tex_path.write_text(r"\documentclass{article}\begin{document}Hi\end{document}")

    def fake_which(command):
        if command == "pdflatex":
            return "C:/tools/pdflatex.exe"
        return None

    def fake_run(command, capture_output, text, timeout, check, cwd):
        assert command[0] == "C:/tools/pdflatex.exe"
        assert "-output-directory" in command
        assert cwd == str(tex_path.parent)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sample_report.pdf").write_bytes(b"%PDF-1.4")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)

    pdf_path = compile_pdf(tex_path, output_dir)

    assert pdf_path == output_dir / "sample_report.pdf"


def test_compile_pdf_raises_when_no_compiler_is_available(tmp_path, monkeypatch):
    tex_path = tmp_path / "sample_report.tex"
    tex_path.write_text(r"\documentclass{article}\begin{document}Hi\end{document}")
    monkeypatch.setattr("shutil.which", lambda command: None)

    with pytest.raises(RuntimeError, match="No LaTeX compiler found"):
        compile_pdf(tex_path, tmp_path / "pdf")
