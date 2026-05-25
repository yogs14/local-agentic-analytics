from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.reporting.latex_builder import render_latex_report
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
LATEX_OUTPUT_PATH = PROJECT_ROOT / "reports" / "latex" / "sample_report.tex"
PDF_OUTPUT_DIR = PROJECT_ROOT / "reports" / "pdf"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def build_sample_report() -> AnalysisReport:
    figures = []
    available_figures = sorted(FIGURES_DIR.glob("*.png"))
    if available_figures:
        figure_path = available_figures[0]
        figures.append(
            ReportFigure(
                figure_id=figure_path.stem,
                title="Contoh Grafik Energi",
                path=str(figure_path),
                caption=(
                    "Contoh grafik energi yang digunakan untuk validasi "
                    "template laporan LaTeX."
                ),
            )
        )

    return AnalysisReport(
        title="Sample Energy Analytics Report",
        author="local-agentic-analytics",
        abstract=(
            "Laporan contoh ini memvalidasi generator LaTeX berbasis template "
            "untuk hasil analisis energi lokal."
        ),
        introduction=(
            "Proyek local-agentic-analytics dirancang untuk menjalankan "
            "analisis data energi secara lokal, ringan, dan modular."
        ),
        methodology=(
            "Data terstruktur diproses menggunakan DuckDB. Grafik dan narasi "
            "analisis disusun sebagai artefak laporan tanpa memanggil LLM pada "
            "tahap rendering LaTeX."
        ),
        sections=[
            ReportSection(
                title="Contoh Analisis Grafik",
                content=(
                    "Bagian ini menunjukkan bagaimana satu grafik dapat "
                    "disisipkan ke dalam laporan menggunakan metadata figure."
                ),
                figures=figures,
            )
        ],
        synthesis=(
            "Template ini dapat menerima beberapa section dan beberapa figure "
            "untuk menyusun hasil eksperimen secara konsisten."
        ),
        conclusion=(
            "Generator laporan berhasil memisahkan struktur data laporan, "
            "rendering LaTeX, dan kompilasi PDF."
        ),
    )


def main() -> int:
    report = build_sample_report()
    tex_path = render_latex_report(
        report=report,
        template_path=TEMPLATE_PATH,
        output_path=LATEX_OUTPUT_PATH,
    )
    print(f"LaTeX report generated: {tex_path}")

    try:
        pdf_path = compile_pdf(tex_path=tex_path, output_dir=PDF_OUTPUT_DIR)
    except Exception as exc:
        print(f"PDF compilation failed: {exc}")
        return 1

    print(f"PDF report generated: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
