"""Report workflow for arbitrary CSV datasets.

This is a generalisation of :class:`EnergyReportWorkflow`. Its ``run()``
interface is identical so callers can swap the two without code changes. The
difference is that charts are derived from the numeric/categorical columns found
in a :class:`DatasetProfile` instead of a hard-coded energy chart registry, and
the report narrative is built from profile metadata.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from local_agentic_analytics.agents.insight_agent import InsightAgent
from local_agentic_analytics.core.config import PROJECT_ROOT
from local_agentic_analytics.core.dataset_profile import DatasetProfile
from local_agentic_analytics.reporting.latex_builder import render_latex_report
from local_agentic_analytics.reporting.pdf_compiler import compile_pdf
from local_agentic_analytics.reporting.report_schema import (
    AnalysisReport,
    ReportFigure,
    ReportSection,
)
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool
from local_agentic_analytics.tools.ollama_tool import OllamaTool


DEFAULT_TEMPLATE_PATH = (
    PROJECT_ROOT
    / "src"
    / "local_agentic_analytics"
    / "reporting"
    / "templates"
    / "analysis_report.tex.j2"
)

_MAX_TIMESERIES_SERIES = 3
_MAX_AVERAGE_COLUMNS = 8
_HISTOGRAM_BINS = 30
_CATEGORY_LIMIT = 15


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _numeric_columns(profile: DatasetProfile) -> list[str]:
    return [
        name
        for name, column in profile.columns.items()
        if column.semantic_type in ("numeric", "integer")
    ]


def _categorical_columns(profile: DatasetProfile) -> list[str]:
    return [
        name
        for name, column in profile.columns.items()
        if column.semantic_type == "categorical"
    ]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _log_chart_error(chart_label: str, exc: Exception) -> None:
    print(
        f"[generic_report_workflow] chart '{chart_label}' skipped: {exc}",
        file=sys.stderr,
    )


def _save_figure(fig: "plt.Figure", path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _chart_timeseries(
    duckdb_tool: DuckDBTool,
    table_name: str,
    profile: DatasetProfile,
    numeric_cols: list[str],
    output_dir: Path,
) -> dict[str, Any] | None:
    datetime_col = profile.datetime_column
    if not datetime_col or not numeric_cols:
        return None

    series_cols = numeric_cols[:_MAX_TIMESERIES_SERIES]
    quoted_table = _quote(table_name)
    avg_selects = ", ".join(
        f"AVG({_quote(col)}) AS {_quote('avg_' + col)}" for col in series_cols
    )
    sql = (
        f"SELECT DATE_TRUNC('day', CAST({_quote(datetime_col)} AS TIMESTAMP)) "
        f"AS period, {avg_selects} "
        f"FROM {quoted_table} "
        f"WHERE {_quote(datetime_col)} IS NOT NULL "
        "GROUP BY period ORDER BY period"
    )
    df = duckdb_tool.execute_query(sql)
    if df.empty:
        return None

    total_rows = int(
        duckdb_tool.execute_query(
            f"SELECT COUNT(*) AS n FROM {quoted_table}"
        )["n"].iloc[0]
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col in series_cols:
        ax.plot(df["period"], df[f"avg_{col}"], marker="o", label=col)
    ax.set_xlabel("Periode")
    ax.set_ylabel("Rata-rata")
    ax.set_title("Tren Rata-rata Harian")
    ax.legend()
    fig.autofmt_xdate()

    path = output_dir / "chart_timeseries.png"
    _save_figure(fig, path)

    periods = df["period"].tolist()
    stats = {
        "row_count": total_rows,
        "period_count": int(len(df)),
        "columns": series_cols,
        "min_date": str(periods[0]) if periods else "",
        "max_date": str(periods[-1]) if periods else "",
    }
    return {
        "chart_id": "timeseries",
        "title": "Tren Rata-rata Harian",
        "path": str(path.resolve()),
        "description": (
            "Tren rata-rata harian untuk kolom numerik terhadap kolom waktu "
            f"{datetime_col}."
        ),
        "stats": stats,
    }


def _chart_distribution(
    duckdb_tool: DuckDBTool,
    table_name: str,
    numeric_cols: list[str],
    output_dir: Path,
) -> dict[str, Any] | None:
    if not numeric_cols:
        return None

    column = numeric_cols[0]
    quoted_table = _quote(table_name)
    quoted_col = _quote(column)

    agg = duckdb_tool.execute_query(
        f"SELECT MIN({quoted_col}) AS mn, MAX({quoted_col}) AS mx, "
        f"AVG({quoted_col}) AS av, STDDEV_SAMP({quoted_col}) AS sd, "
        f"COUNT(*) AS total, "
        f"COUNT(*) FILTER (WHERE {quoted_col} IS NULL) AS nulls "
        f"FROM {quoted_table}"
    )
    minv = _to_float(agg["mn"].iloc[0])
    maxv = _to_float(agg["mx"].iloc[0])
    avg = _to_float(agg["av"].iloc[0])
    std = _to_float(agg["sd"].iloc[0])
    null_count = int(agg["nulls"].iloc[0])

    if minv is None or maxv is None:
        return None

    bin_width = max((maxv - minv) / float(_HISTOGRAM_BINS), 0.001)
    hist = duckdb_tool.execute_query(
        f"SELECT FLOOR(({quoted_col} - {minv}) / {bin_width}) AS bin_idx, "
        f"COUNT(*) AS freq FROM {quoted_table} "
        f"WHERE {quoted_col} IS NOT NULL GROUP BY bin_idx ORDER BY bin_idx"
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not hist.empty:
        centers = [minv + (idx + 0.5) * bin_width for idx in hist["bin_idx"]]
        ax.bar(centers, hist["freq"], width=bin_width * 0.9)
    ax.set_xlabel(column)
    ax.set_ylabel("Frekuensi")
    ax.set_title(f"Distribusi {column}")

    path = output_dir / "chart_distribution.png"
    _save_figure(fig, path)

    stats = {
        "column": column,
        "min": minv,
        "max": maxv,
        "avg": avg,
        "std": std,
        "bin_count": int(len(hist)),
        "null_count": null_count,
    }
    return {
        "chart_id": f"distribution_{column}",
        "title": f"Distribusi {column}",
        "path": str(path.resolve()),
        "description": f"Distribusi frekuensi nilai pada kolom {column}.",
        "stats": stats,
    }


def _chart_column_averages(
    duckdb_tool: DuckDBTool,
    table_name: str,
    numeric_cols: list[str],
    output_dir: Path,
) -> dict[str, Any] | None:
    if len(numeric_cols) < 2:
        return None

    columns = numeric_cols[:_MAX_AVERAGE_COLUMNS]
    quoted_table = _quote(table_name)
    selects = ", ".join(
        f"AVG({_quote(col)}) AS {_quote('avg_' + col)}" for col in columns
    )
    df = duckdb_tool.execute_query(f"SELECT {selects} FROM {quoted_table}")

    averages: dict[str, float | None] = {}
    for col in columns:
        averages[col] = _to_float(df[f"avg_{col}"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_cols = [c for c in columns if averages[c] is not None]
    plot_vals = [averages[c] for c in plot_cols]
    ax.barh(plot_cols, plot_vals)
    ax.set_xlabel("Rata-rata")
    ax.set_title("Rata-rata per Kolom Numerik")

    path = output_dir / "chart_averages.png"
    _save_figure(fig, path)

    stats = {
        "columns": columns,
        "averages": averages,
    }
    return {
        "chart_id": "column_averages",
        "title": "Rata-rata per Kolom Numerik",
        "path": str(path.resolve()),
        "description": "Perbandingan rata-rata antar kolom numerik.",
        "stats": stats,
    }


def _chart_categories(
    duckdb_tool: DuckDBTool,
    table_name: str,
    profile: DatasetProfile,
    categorical_cols: list[str],
    output_dir: Path,
) -> dict[str, Any] | None:
    datetime_col = profile.datetime_column
    column = None
    for candidate in categorical_cols:
        if candidate == datetime_col:
            continue
        if "id" in candidate.lower():
            continue
        column = candidate
        break
    if column is None:
        return None

    quoted_table = _quote(table_name)
    quoted_col = _quote(column)
    df = duckdb_tool.execute_query(
        f"SELECT {quoted_col} AS category, COUNT(*) AS n FROM {quoted_table} "
        f"WHERE {quoted_col} IS NOT NULL "
        f"GROUP BY {quoted_col} ORDER BY n DESC LIMIT {_CATEGORY_LIMIT}"
    )
    if df.empty:
        return None

    unique_df = duckdb_tool.execute_query(
        f"SELECT COUNT(DISTINCT {quoted_col}) AS u, "
        f"COUNT({quoted_col}) AS t FROM {quoted_table}"
    )
    unique_count = int(unique_df["u"].iloc[0])
    total_count = int(unique_df["t"].iloc[0])

    categories = [str(value) for value in df["category"].tolist()]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(categories[::-1], df["n"].tolist()[::-1])
    ax.set_xlabel("Jumlah")
    ax.set_title(f"Frekuensi Kategori: {column}")

    path = output_dir / "chart_categories.png"
    _save_figure(fig, path)

    stats = {
        "column": column,
        "top_category": categories[0] if categories else None,
        "unique_count": unique_count,
        "total_count": total_count,
    }
    return {
        "chart_id": f"categories_{column}",
        "title": f"Frekuensi Kategori: {column}",
        "path": str(path.resolve()),
        "description": f"Distribusi 15 kategori teratas pada kolom {column}.",
        "stats": stats,
    }


def generate_generic_charts(
    duckdb_tool: DuckDBTool,
    table_name: str,
    profile: DatasetProfile,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Build deterministic charts from a :class:`DatasetProfile`.

    Each chart that fails is logged to stderr and skipped; the function never
    raises for chart-specific errors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = _numeric_columns(profile)
    categorical_cols = _categorical_columns(profile)

    builders = [
        ("timeseries", lambda: _chart_timeseries(
            duckdb_tool, table_name, profile, numeric_cols, output_dir
        )),
        ("distribution", lambda: _chart_distribution(
            duckdb_tool, table_name, numeric_cols, output_dir
        )),
        ("column_averages", lambda: _chart_column_averages(
            duckdb_tool, table_name, numeric_cols, output_dir
        )),
        ("categories", lambda: _chart_categories(
            duckdb_tool, table_name, profile, categorical_cols, output_dir
        )),
    ]

    charts: list[dict[str, Any]] = []
    for label, builder in builders:
        try:
            chart = builder()
        except Exception as exc:  # noqa: BLE001 - resilient per-chart handling
            _log_chart_error(label, exc)
            plt.close("all")
            continue
        if chart is not None:
            charts.append(chart)
    return charts


class GenericReportWorkflow:
    """Generate a PDF report for an arbitrary CSV dataset.

    The ``run()`` interface mirrors :class:`EnergyReportWorkflow` so it can be
    swapped without changing caller code.
    """

    def __init__(
        self,
        duckdb_tool: DuckDBTool,
        table_name: str,
        profile: DatasetProfile,
        output_dir: Path,
        insight_agent: InsightAgent | None = None,
        model_name: str | None = None,
    ):
        self.duckdb_tool = duckdb_tool
        self.table_name = table_name
        self.profile = profile
        self.output_dir = Path(output_dir)
        # See SequentialAnalyticsWorkflow.model_name — same override contract,
        # applied to the report's InsightAgent instead of the SQL/planner path.
        override_tool = (
            OllamaTool.from_config(model_override=model_name) if model_name else None
        )
        self.insight_agent = insight_agent or InsightAgent(ollama_tool=override_tool)
        self.template_path = DEFAULT_TEMPLATE_PATH

    def run(self) -> dict[str, Any]:
        chart_metadata: list[dict] = []
        insight_records: list[dict] = []
        tex_path: Path | None = None
        pdf_path: Path | None = None
        workflow_error = ""
        pdf_error = ""

        try:
            self._ensure_table_exists()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            chart_metadata = generate_generic_charts(
                self.duckdb_tool,
                self.table_name,
                self.profile,
                self.output_dir,
            )
            insight_records = self._generate_insights(chart_metadata)
            report = self._build_report(insight_records)
            tex_output_path = self.output_dir / f"{self.table_name}_report.tex"
            tex_path = render_latex_report(
                report=report,
                template_path=self.template_path,
                output_path=tex_output_path,
            )
            try:
                pdf_path = compile_pdf(
                    tex_path=tex_path, output_dir=self.output_dir
                )
            except Exception as exc:  # noqa: BLE001
                pdf_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            workflow_error = str(exc)

        insight_success = sum(
            1 for record in insight_records if record.get("success")
        )
        insight_failed = sum(
            1 for record in insight_records if not record.get("success")
        )

        return {
            "success": tex_path is not None,
            "tex_success": tex_path is not None,
            "pdf_success": pdf_path is not None,
            "tex_path": str(tex_path) if tex_path else "",
            "pdf_path": str(pdf_path) if pdf_path else "",
            "chart_count": len(chart_metadata),
            "insight_success_count": insight_success,
            "insight_failed_count": insight_failed,
            "error_message": workflow_error,
            "pdf_error": pdf_error,
            "latency": {},
            "tool_calls": [],
        }

    def _ensure_table_exists(self) -> None:
        escaped = self.table_name.replace("'", "''")
        result = self.duckdb_tool.execute_query(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_name = '{escaped}'"
        )
        if result.empty:
            raise ValueError(
                f"Table not found in database: {self.table_name}"
            )

    def _generate_insights(self, chart_metadata: list[dict]) -> list[dict]:
        records = []
        for chart in chart_metadata:
            context = {
                "chart_id": chart["chart_id"],
                "chart_title": chart["title"],
                "chart_path": chart["path"],
                "chart_description": chart.get("description", ""),
                "stats": chart.get("stats", {}),
            }
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
            except Exception as exc:  # noqa: BLE001
                records.append(
                    {
                        **context,
                        "insight": (
                            "Insight otomatis tidak dapat dibuat untuk grafik "
                            "ini. Periksa koneksi Ollama atau konfigurasi model "
                            "lokal."
                        ),
                        "success": False,
                        "error_message": str(exc),
                    }
                )
        return records

    def _build_report(self, insight_records: list[dict]) -> AnalysisReport:
        sections = []
        for record in insight_records:
            caption = str(
                record.get("chart_description")
                or record.get("chart_title")
                or "Grafik analisis"
            )
            sections.append(
                ReportSection(
                    title=str(record["chart_title"]),
                    content=str(record["insight"]),
                    figures=[
                        ReportFigure(
                            figure_id=str(record["chart_id"]),
                            title=str(record["chart_title"]),
                            path=str(record["chart_path"]),
                            caption=caption,
                        )
                    ],
                )
            )

        successful = [
            str(record["insight"])
            for record in insight_records
            if record.get("success") and str(record.get("insight", "")).strip()
        ]
        numeric_count = len(_numeric_columns(self.profile))

        return AnalysisReport(
            title=f"Laporan Analisis Dataset: {self.profile.name}",
            author="Local Agentic Analytics",
            abstract=_build_abstract(successful),
            introduction=(
                f"Laporan ini menganalisis dataset '{self.profile.name}' yang "
                f"disimpan pada tabel '{self.table_name}'. Dataset memiliki "
                f"{numeric_count} kolom numerik yang menjadi dasar visualisasi "
                "dan analisis statistik ringkas."
            ),
            methodology=(
                "Data terstruktur disimpan dan diagregasi menggunakan DuckDB "
                "agar analisis tetap ringan di lingkungan lokal. Visualisasi "
                "dibuat secara deterministik dengan matplotlib dari hasil query "
                "agregat, sedangkan narasi tiap grafik dihasilkan oleh "
                "InsightAgent berbasis model lokal menggunakan statistik "
                "ringkas, bukan data mentah."
            ),
            sections=sections,
            synthesis=_build_synthesis(successful),
            conclusion=_build_conclusion(successful),
        )


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
            "Laporan ini menyajikan visualisasi dan struktur analisis untuk "
            "dataset yang diunggah. Insight otomatis belum tersedia sehingga "
            "interpretasi kuantitatif perlu ditinjau melalui grafik dan "
            "statistik ringkas yang dihasilkan."
        )
    highlights = [_first_sentence(insight) for insight in insights[:2]]
    return (
        "Laporan ini menyajikan analisis dataset berdasarkan grafik agregat "
        "dan insight otomatis. Ringkasan temuan utama: " + " ".join(highlights)
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
        "perlu dibaca bersama. " + " ".join(selected) + " Penilaian anomali "
        "tetap memerlukan pembanding historis atau batas operasional eksplisit."
    )


def _build_conclusion(insights: list[str]) -> str:
    if not insights:
        return (
            "Workflow pelaporan berhasil menyusun struktur laporan dan grafik, "
            "namun narasi insight perlu dibuat ulang setelah layanan model lokal "
            "siap."
        )
    return (
        "Laporan ini menunjukkan bahwa pipeline lokal dapat menghasilkan grafik, "
        "statistik ringkas, dan narasi analisis secara sequential untuk dataset "
        "sembarang. Hasil ini dapat digunakan sebagai dasar eksplorasi data "
        "lanjutan, dengan catatan bahwa klaim anomali memerlukan pembanding "
        "historis tambahan."
    )
