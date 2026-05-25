"""Sequential workflow for generating an energy analysis report."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import duckdb

from local_agentic_analytics.agents.insight_agent import InsightAgent
from local_agentic_analytics.reporting.latex_builder import render_latex_report
from local_agentic_analytics.reporting.pdf_compiler import compile_pdf
from local_agentic_analytics.reporting.report_schema import (
    AnalysisReport,
    ReportFigure,
    ReportSection,
)
from local_agentic_analytics.visualization.chart_registry import (
    ENERGY_CHART_REGISTRY,
    generate_all_energy_charts,
)
from local_agentic_analytics.visualization.chart_stats import CHART_STATS_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_LATEX_DIR = PROJECT_ROOT / "reports" / "latex"
DEFAULT_PDF_DIR = PROJECT_ROOT / "reports" / "pdf"
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "reports" / "experiments"
DEFAULT_TEMPLATE_PATH = (
    PROJECT_ROOT
    / "src"
    / "local_agentic_analytics"
    / "reporting"
    / "templates"
    / "analysis_report.tex.j2"
)
DEFAULT_TEX_PATH = DEFAULT_LATEX_DIR / "energy_analysis_report.tex"
DEFAULT_LOG_PATH = DEFAULT_EXPERIMENTS_DIR / "report_generation_log.json"


class EnergyReportWorkflow:
    """Generate charts, insights, LaTeX, and PDF for the energy dataset."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        figures_dir: str | Path = DEFAULT_FIGURES_DIR,
        latex_output_path: str | Path = DEFAULT_TEX_PATH,
        pdf_dir: str | Path = DEFAULT_PDF_DIR,
        template_path: str | Path = DEFAULT_TEMPLATE_PATH,
        log_path: str | Path = DEFAULT_LOG_PATH,
        insight_agent: InsightAgent | None = None,
    ):
        self.db_path = Path(db_path)
        self.figures_dir = Path(figures_dir)
        self.latex_output_path = Path(latex_output_path)
        self.pdf_dir = Path(pdf_dir)
        self.template_path = Path(template_path)
        self.log_path = Path(log_path)
        self.insight_agent = insight_agent or InsightAgent()

    def run(self) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        chart_metadata: list[dict] = []
        chart_contexts: list[dict] = []
        insight_records: list[dict] = []
        tex_path: Path | None = None
        pdf_path: Path | None = None
        workflow_error = ""
        pdf_error = ""

        try:
            chart_metadata = generate_all_energy_charts(
                db_path=self.db_path,
                output_dir=self.figures_dir,
            )
            chart_contexts = self._build_chart_contexts(chart_metadata)
            insight_records = self._generate_insights(chart_contexts)
            report = self._build_report(insight_records)
            tex_path = render_latex_report(
                report=report,
                template_path=self.template_path,
                output_path=self.latex_output_path,
            )
            try:
                pdf_path = compile_pdf(tex_path=tex_path, output_dir=self.pdf_dir)
            except Exception as exc:
                pdf_error = str(exc)
        except Exception as exc:
            workflow_error = str(exc)
        finally:
            finished_at = datetime.now(timezone.utc)

        metadata = {
            "timestamp_start": started_at.isoformat(),
            "timestamp_end": finished_at.isoformat(),
            "success": tex_path is not None and not pdf_error and not workflow_error,
            "tex_success": tex_path is not None,
            "pdf_success": pdf_path is not None,
            "error_message": workflow_error,
            "pdf_error": pdf_error,
            "db_path": str(self.db_path),
            "figures_dir": str(self.figures_dir),
            "tex_path": str(tex_path) if tex_path else "",
            "pdf_path": str(pdf_path) if pdf_path else "",
            "chart_count": len(chart_metadata),
            "insight_success_count": sum(
                1 for record in insight_records if record.get("success")
            ),
            "insight_failed_count": sum(
                1 for record in insight_records if not record.get("success")
            ),
            "charts": chart_metadata,
            "insights": insight_records,
        }
        self._write_log(metadata)
        return metadata

    def _build_chart_contexts(self, chart_metadata: list[dict]) -> list[dict]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB database not found: {self.db_path}")

        contexts = []
        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            for chart in chart_metadata:
                chart_id = str(chart["chart_id"])
                stats_function = CHART_STATS_REGISTRY.get(chart_id)
                if stats_function is None:
                    raise KeyError(f"No stats function registered for chart: {chart_id}")

                contexts.append(
                    {
                        "chart_id": chart_id,
                        "chart_title": str(chart["title"]),
                        "chart_path": str(chart["path"]),
                        "chart_description": str(chart.get("description", "")),
                        "stats": stats_function(con),
                    }
                )
        finally:
            con.close()

        return contexts

    def _generate_insights(self, chart_contexts: list[dict]) -> list[dict]:
        records = []
        for context in chart_contexts:
            try:
                insight = self.insight_agent.generate_insight(context)
                records.append(
                    {
                        **context,
                        "insight": insight,
                        "success": True,
                        "error_message": "",
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        **context,
                        "insight": (
                            "Insight otomatis tidak dapat dibuat untuk grafik ini. "
                            "Periksa koneksi Ollama atau konfigurasi model lokal."
                        ),
                        "success": False,
                        "error_message": str(exc),
                    }
                )
        return records

    def _build_report(self, insight_records: list[dict]) -> AnalysisReport:
        sections = []
        for record in insight_records:
            chart_id = str(record["chart_id"])
            chart_info = ENERGY_CHART_REGISTRY.get(chart_id, {})
            sections.append(
                ReportSection(
                    title=str(record["chart_title"]),
                    content=str(record["insight"]),
                    figures=[
                        ReportFigure(
                            figure_id=chart_id,
                            title=str(record["chart_title"]),
                            path=str(record["chart_path"]),
                            caption=str(
                                chart_info.get(
                                    "description",
                                    record.get("chart_description", ""),
                                )
                            ),
                        )
                    ],
                )
            )

        successful_insights = [
            str(record["insight"])
            for record in insight_records
            if record.get("success") and str(record.get("insight", "")).strip()
        ]
        abstract = _build_abstract(successful_insights)
        synthesis = _build_synthesis(successful_insights)
        conclusion = _build_conclusion(successful_insights)

        return AnalysisReport(
            title="Laporan Analisis Konsumsi Daya Listrik Rumah Tangga",
            author="Local Agentic Analytics",
            abstract=abstract,
            introduction=(
                "Laporan ini menganalisis dataset Individual Household Electric "
                "Power Consumption yang memuat pengukuran konsumsi listrik rumah "
                "tangga per menit, termasuk daya aktif global, daya reaktif, "
                "tegangan, arus, dan tiga kanal sub-metering."
            ),
            methodology=(
                "Data terstruktur disimpan dan diagregasi menggunakan DuckDB agar "
                "analisis tetap ringan di lingkungan lokal. Visualisasi dibuat "
                "secara deterministik dengan matplotlib dari hasil query agregat, "
                "sedangkan narasi tiap grafik dihasilkan secara sequential oleh "
                "InsightAgent berbasis model lokal Ollama menggunakan statistik "
                "ringkas, bukan dataframe mentah."
            ),
            sections=sections,
            synthesis=synthesis,
            conclusion=conclusion,
        )

    def _write_log(self, metadata: dict[str, Any]) -> Path:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.log_path


def _first_sentence(text: str) -> str:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return ""

    for delimiter in [". ", "! ", "? "]:
        if delimiter in normalized:
            return normalized.split(delimiter, maxsplit=1)[0].strip() + delimiter[0]

    return normalized


def _build_abstract(insights: list[str]) -> str:
    if not insights:
        return (
            "Laporan ini menyajikan visualisasi dan struktur analisis konsumsi "
            "daya listrik rumah tangga. Insight otomatis belum tersedia sehingga "
            "interpretasi kuantitatif perlu ditinjau melalui grafik dan statistik "
            "ringkas yang dihasilkan."
        )

    highlights = [_first_sentence(insight) for insight in insights[:2]]
    return (
        "Laporan ini menyajikan analisis konsumsi daya listrik rumah tangga "
        "berdasarkan grafik agregat dan insight otomatis. Ringkasan temuan utama: "
        + " ".join(highlights)
    )


def _build_synthesis(insights: list[str]) -> str:
    if not insights:
        return (
            "Temuan utama belum dapat disintesis secara otomatis karena insight "
            "per grafik tidak tersedia. Pemeriksaan lanjutan terhadap hasil "
            "visualisasi dan statistik ringkas tetap diperlukan."
        )

    selected = [_first_sentence(insight) for insight in insights[:4]]
    return (
        "Secara keseluruhan, hasil visualisasi menunjukkan beberapa pola yang "
        "perlu dibaca bersama, meliputi tren harian, pola per jam, distribusi "
        "daya dan tegangan, korelasi fitur, serta kontribusi sub-metering. "
        + " ".join(selected)
        + " Penilaian anomali tetap memerlukan pembanding historis atau batas "
        "operasional yang eksplisit."
    )


def _build_conclusion(insights: list[str]) -> str:
    if not insights:
        return (
            "Workflow pelaporan berhasil menyusun struktur laporan dan grafik, "
            "namun narasi insight perlu dibuat ulang setelah layanan Ollama siap."
        )

    return (
        "Laporan ini menunjukkan bahwa pipeline lokal dapat menghasilkan grafik, "
        "statistik ringkas, dan narasi analisis secara sequential. Hasil ini dapat "
        "digunakan sebagai dasar evaluasi konsumsi listrik rumah tangga, dengan "
        "catatan bahwa klaim anomali memerlukan pembanding historis tambahan."
    )
