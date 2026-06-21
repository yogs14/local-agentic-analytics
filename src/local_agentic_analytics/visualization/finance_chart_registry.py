from pathlib import Path
from typing import Callable

import duckdb

from local_agentic_analytics.visualization.finance_charts import (
    plot_average_close_by_ticker,
    plot_average_volume_by_ticker,
    plot_close_price_trend,
    plot_daily_return_distribution,
    plot_ticker_close_correlation,
)


ChartFunction = Callable[[duckdb.DuckDBPyConnection, str | Path], Path]


FINANCE_CHART_REGISTRY: dict[str, dict[str, str | ChartFunction]] = {
    "close_price_trend": {
        "function": plot_close_price_trend,
        "filename": "finance_close_price_trend.png",
        "title": "Tren Harga Penutupan Harian per Ticker",
        "description": "Harga penutupan harian (USD) untuk NVDA, NFLX, TSLA, dan GOOGL.",
    },
    "average_close_by_ticker": {
        "function": plot_average_close_by_ticker,
        "filename": "finance_average_close_by_ticker.png",
        "title": "Rata-rata Harga Penutupan per Ticker",
        "description": "Rata-rata harga penutupan (USD) tiap ticker selama periode dataset.",
    },
    "average_volume_by_ticker": {
        "function": plot_average_volume_by_ticker,
        "filename": "finance_average_volume_by_ticker.png",
        "title": "Rata-rata Volume Perdagangan per Ticker",
        "description": "Rata-rata volume perdagangan harian (lembar) tiap ticker.",
    },
    "daily_return_distribution": {
        "function": plot_daily_return_distribution,
        "filename": "finance_daily_return_distribution.png",
        "title": "Distribusi Return Harian",
        "description": "Histogram return harian (%) seluruh ticker.",
    },
    "ticker_close_correlation": {
        "function": plot_ticker_close_correlation,
        "filename": "finance_ticker_close_correlation.png",
        "title": "Korelasi Harga Penutupan Antar Ticker",
        "description": "Korelasi harga penutupan harian antar ticker.",
    },
}


def generate_all_finance_charts(db_path: str | Path, output_dir: str | Path) -> list[dict]:
    database_path = Path(db_path)
    if not database_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {database_path}")

    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    chart_metadata = []
    con = duckdb.connect(str(database_path), read_only=True)
    try:
        for chart_id, chart_info in FINANCE_CHART_REGISTRY.items():
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
