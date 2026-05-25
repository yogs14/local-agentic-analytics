from pathlib import Path
from typing import Callable

import duckdb

from local_agentic_analytics.visualization.energy_charts import (
    plot_correlation_heatmap,
    plot_daily_active_power_trend,
    plot_hourly_consumption_pattern,
    plot_power_distribution,
    plot_sub_metering_comparison,
    plot_voltage_distribution,
)


ChartFunction = Callable[[duckdb.DuckDBPyConnection, str | Path], Path]


ENERGY_CHART_REGISTRY: dict[str, dict[str, str | ChartFunction]] = {
    "daily_active_power_trend": {
        "function": plot_daily_active_power_trend,
        "filename": "daily_active_power_trend.png",
        "title": "Tren Rata-rata Daya Aktif Harian",
        "description": "Rata-rata Global_active_power per tanggal.",
    },
    "hourly_consumption_pattern": {
        "function": plot_hourly_consumption_pattern,
        "filename": "hourly_consumption_pattern.png",
        "title": "Pola Konsumsi Daya per Jam",
        "description": "Rata-rata Global_active_power berdasarkan jam.",
    },
    "power_distribution": {
        "function": plot_power_distribution,
        "filename": "power_distribution.png",
        "title": "Distribusi Global Active Power",
        "description": "Histogram agregat Global_active_power.",
    },
    "voltage_distribution": {
        "function": plot_voltage_distribution,
        "filename": "voltage_distribution.png",
        "title": "Distribusi Voltage",
        "description": "Histogram agregat Voltage.",
    },
    "correlation_heatmap": {
        "function": plot_correlation_heatmap,
        "filename": "correlation_heatmap.png",
        "title": "Korelasi Fitur Numerik Energi",
        "description": "Korelasi antar kolom numerik utama pada tabel electric_power.",
    },
    "sub_metering_comparison": {
        "function": plot_sub_metering_comparison,
        "filename": "sub_metering_comparison.png",
        "title": "Perbandingan Rata-rata Sub Metering",
        "description": "Rata-rata Sub_metering_1, Sub_metering_2, dan Sub_metering_3.",
    },
}


def generate_all_energy_charts(db_path: str | Path, output_dir: str | Path) -> list[dict]:
    database_path = Path(db_path)
    if not database_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {database_path}")

    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    chart_metadata = []
    con = duckdb.connect(str(database_path), read_only=True)
    try:
        for chart_id, chart_info in ENERGY_CHART_REGISTRY.items():
            chart_function = chart_info["function"]
            if not callable(chart_function):
                raise TypeError(f"Chart function for {chart_id} is not callable.")

            output_path = figures_dir / str(chart_info["filename"])
            saved_path = chart_function(con, output_path)
            chart_metadata.append(
                {
                    "chart_id": chart_id,
                    "title": str(chart_info["title"]),
                    "path": str(saved_path),
                    "description": str(chart_info["description"]),
                }
            )
    finally:
        con.close()

    return chart_metadata
