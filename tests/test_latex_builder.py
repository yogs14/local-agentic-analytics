from pathlib import Path
import sys


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


def test_escape_latex_escapes_common_problem_characters():
    assert escape_latex("Global_active_power & 100% #1") == (
        r"Global\_active\_power \& 100\% \#1"
    )


def test_normalize_figure_path_is_windows_safe(tmp_path):
    figure_path = tmp_path / "reports" / "figures" / "chart.png"
    latex_dir = tmp_path / "reports" / "latex"

    normalized = normalize_figure_path(figure_path, latex_dir)

    assert normalized == "../figures/chart.png"
    assert "\\" not in normalized


def test_latex_label_keeps_label_safe():
    assert latex_label("daily_active_power trend!") == "daily-active-power-trend"


def test_render_latex_report_creates_tex_file(tmp_path):
    report = AnalysisReport(
        title="Energy_Report",
        author="Local_Agentic_Analytics",
        abstract="Ringkasan 100%.",
        introduction="Pendahuluan dengan Global_active_power.",
        methodology="DuckDB & matplotlib.",
        sections=[
            ReportSection(
                title="Chart Section",
                content="Konten singkat.",
                figures=[
                    ReportFigure(
                        figure_id="daily_active_power",
                        title="Daily Active Power",
                        path=str(tmp_path / "reports" / "figures" / "daily.png"),
                        caption="Caption _ aman.",
                    )
                ],
            )
        ],
        synthesis="Sintesis.",
        conclusion="Kesimpulan.",
    )

    output_path = tmp_path / "reports" / "latex" / "report.tex"
    rendered_path = render_latex_report(report, TEMPLATE_PATH, output_path)
    content = rendered_path.read_text(encoding="utf-8")

    assert rendered_path == output_path
    assert r"Energy\_Report" in content
    assert r"\detokenize{../figures/daily.png}" in content
    assert r"\label{fig:daily-active-power}" in content
