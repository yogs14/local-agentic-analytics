from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.reporting.report_schema import (
    AnalysisReport,
    ReportFigure,
    ReportSection,
)


def test_report_figure_validates_required_fields():
    with pytest.raises(ValueError, match="figure_id must not be empty"):
        ReportFigure(figure_id="", title="Title", path="figure.png", caption="Caption")


def test_report_section_defaults_to_empty_figures():
    section = ReportSection(title="Section", content="Content")

    assert section.figures == []


def test_analysis_report_accepts_sections():
    report = AnalysisReport(
        title="Title",
        author="Author",
        abstract="Abstract",
        introduction="Introduction",
        methodology="Methodology",
        sections=[ReportSection(title="Section", content="Content")],
        synthesis="Synthesis",
        conclusion="Conclusion",
    )

    assert report.sections[0].title == "Section"


def test_analysis_report_rejects_non_list_sections():
    with pytest.raises(ValueError, match="sections must be a list"):
        AnalysisReport(
            title="Title",
            author="Author",
            abstract="Abstract",
            introduction="Introduction",
            methodology="Methodology",
            sections="not-a-list",
            synthesis="Synthesis",
            conclusion="Conclusion",
        )
