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
from local_agentic_analytics.prompts.insight_prompt import build_conclusion_prompt
import re


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
        self.insight_agent = insight_agent or InsightAgent.for_energy_finetune()

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
            "engine": "custom",
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
            "log_path": str(self.log_path),
            "chart_count": len(chart_metadata),
            "insight_success_count": sum(
                1 for record in insight_records if record.get("success")
            ),
            "insight_failed_count": sum(
                1 for record in insight_records if not record.get("success")
            ),
            "latency": {},
            "tool_calls": [],
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
        conclusion = _build_conclusion(insight_records, self.insight_agent)

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


def _build_conclusion(insight_records: list[dict], insight_agent: InsightAgent) -> str:
    successful = [
        r for r in insight_records
        if r.get("success") and str(r.get("insight", "")).strip()
    ]
    if not successful:
        return (
            "Workflow pelaporan berhasil menyusun struktur laporan dan grafik, "
            "namun narasi insight perlu dibuat ulang setelah layanan Ollama siap."
        )

    try:
        prompt = build_conclusion_prompt(insight_records)
        response = insight_agent.ollama_tool.generate(
            prompt=prompt,
            temperature=insight_agent.temperature,
            max_tokens=768,
        )
        conclusion = response.strip()
        if conclusion:
            from local_agentic_analytics.agents.insight_agent import sanitize_narrative
            conclusion = sanitize_narrative(conclusion)
            if conclusion:
                conclusion = _paragraphize(conclusion)
                if conclusion:
                    return conclusion
    except Exception:
        pass

    return _build_conclusion_hardcoded(successful)


def _build_conclusion_hardcoded(successful: list[dict]) -> str:
    stats_by_chart = {str(r["chart_id"]): r.get("stats", {}) for r in successful}

    trend = stats_by_chart.get("daily_active_power_trend", {})
    hourly = stats_by_chart.get("hourly_consumption_pattern", {})
    power = stats_by_chart.get("power_distribution", {})
    voltage = stats_by_chart.get("voltage_distribution", {})
    corr = stats_by_chart.get("correlation_heatmap", {})
    submeter = stats_by_chart.get("sub_metering_comparison", {})

    def _fmt(val, decimals=2):
        try:
            v = float(val)
            if abs(v) < 0.001:
                return f"{v:.4f}"
            if v == int(v):
                return str(int(v))
            return f"{v:.{decimals}f}"
        except (TypeError, ValueError):
            return str(val)

    lines = [
        "Analisis konsumsi daya listrik rumah tangga pada periode Desember 2006 hingga "
        f"November 2010 ({_fmt(trend.get('day_count', '?'))} hari, {_fmt(power.get('record_count', '?'))} rekaman) "
        "mengungkapkan beberapa pola penting yang saling terkait.",
    ]

    lines.append(
        f"Rata-rata daya aktif harian tercatat sebesar {_fmt(trend.get('mean_daily_avg_kw', '?'))} kW "
        f"dengan rentang yang lebar dari {_fmt(trend.get('min_daily_avg_kw', '?'))} kW hingga "
        f"{_fmt(trend.get('max_daily_avg_kw', '?'))} kW, menunjukkan disparitas konsumsi yang tinggi "
        "antar hari yang dipengaruhi oleh faktor musiman dan aktivitas penghuni rumah."
    )

    lines.append(
        f"Pola konsumsi per jam memperlihatkan beban puncak sebesar {_fmt(hourly.get('max_hourly_avg_kw', '?'))} kW "
        f"pada pukul {_fmt(hourly.get('max_avg_hour', '?'))}:00 dan beban dasar terendah "
        f"{_fmt(hourly.get('min_hourly_avg_kw', '?'))} kW pada pukul {_fmt(hourly.get('min_avg_hour', '?'))}:00. "
        "Rasio puncak-ke-lembah yang signifikan ini mengindikasikan potensi penerapan strategi "
        "load shifting dan demand response untuk meratakan kurva beban, khususnya melalui "
        "pengalihan konsumsi dari jam sibuk malam ke jam beban rendah dini hari."
    )

    lines.append(
        f"Tegangan listrik relatif stabil dengan rata-rata {_fmt(voltage.get('avg_voltage_v', '?'))} V "
        f"dan simpangan baku {_fmt(voltage.get('stddev_voltage_v', '?'))} V, berada dalam rentang "
        f"{_fmt(voltage.get('min_voltage_v', '?'))} V hingga {_fmt(voltage.get('max_voltage_v', '?'))} V. "
        "Stabilitas tegangan ini sesuai dengan standar operasi jaringan distribusi residensial."
    )

    strongest = corr.get("strongest_absolute", {})
    if strongest:
        lines.append(
            f"Korelasi Pearson tertinggi ditemukan antara {strongest.get('pair', '?')} "
            f"dengan nilai {_fmt(strongest.get('correlation', '?'))}, menegaskan bahwa konsumsi "
            "daya aktif sangat erat ditentukan oleh intensitas arus yang mengalir. "
            "Sebaliknya, hubungan negatif moderat antara tegangan dan intensitas arus mengindikasikan "
            "efek pembebanan jaringan yang wajar pada sistem distribusi."
        )

    lines.append(
        f"Di antara tiga kanal sub-metering, Sub-metering 3 mendominasi dengan rata-rata "
        f"{_fmt(submeter.get('avg_sub_metering_3_wh', '?'))} Wh, jauh melampaui Sub-metering 1 "
        f"({_fmt(submeter.get('avg_sub_metering_1_wh', '?'))} Wh) dan Sub-metering 2 "
        f"({_fmt(submeter.get('avg_sub_metering_2_wh', '?'))} Wh). "
        "Hal ini mengonfirmasi bahwa beban terbesar berasal dari peralatan listrik tertentu "
        "(seperti pendingin ruangan atau pemanas air) yang terhubung ke kanal meter tersebut. "
        "Segmentasi konsumsi ini dapat menjadi dasar rekomendasi efisiensi energi yang tertarget."
    )

    lines.append(
        "Secara keseluruhan, profil konsumsi rumah tangga yang dianalisis menunjukkan karakteristik "
        "beban residensial tipikal: base load rendah yang didominasi peralatan elektronik dasar "
        "dengan lonjakan signifikan pada jam malam. Temuan ini mendukung perencanaan program "
        "manajemen sisi permintaan (demand side management) berbasis data untuk meningkatkan "
        "efisiensi energi dan mengurangi beban puncak jaringan."
    )

    return "\n\n".join(lines)


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _paragraphize(text: str, min_paragraphs: int = 3, max_sentences_per_para: int = 3) -> str:
    sentences = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
    if not sentences or len(sentences) <= 1:
        return text

    if "\n\n" in text or "\n" in text.strip():
        return text

    target_count = max(min_paragraphs, (len(sentences) + max_sentences_per_para - 1) // max_sentences_per_para)
    per_para = max(2, len(sentences) // target_count)

    paragraphs: list[str] = []
    for i in range(0, len(sentences), per_para):
        chunk = sentences[i:i + per_para]
        paragraphs.append(" ".join(chunk))

    return "\n\n".join(paragraphs)
