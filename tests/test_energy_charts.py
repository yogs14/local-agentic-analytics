from pathlib import Path
import sys

import duckdb
import pytest


pytest.importorskip("matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.visualization.energy_charts import (
    plot_correlation_heatmap,
    plot_daily_active_power_trend,
    plot_hourly_consumption_pattern,
    plot_power_distribution,
    plot_sub_metering_comparison,
    plot_voltage_distribution,
)
from local_agentic_analytics.visualization.chart_registry import (
    ENERGY_CHART_REGISTRY,
    generate_all_energy_charts,
)


def _create_energy_table(con):
    con.execute(
        """
        CREATE TABLE electric_power AS
        SELECT * FROM (
            VALUES
                (TIMESTAMP '2006-12-16 00:00:00', 1.0, 0.1, 240.0, 4.0, 0.0, 1.0, 17.0),
                (TIMESTAMP '2006-12-16 01:00:00', 2.0, 0.2, 241.0, 8.0, 1.0, 2.0, 18.0),
                (TIMESTAMP '2006-12-17 00:00:00', 3.0, 0.3, 242.0, 12.0, 2.0, 3.0, 19.0),
                (TIMESTAMP '2006-12-17 01:00:00', 4.0, 0.4, 243.0, 16.0, 3.0, 4.0, 20.0)
        ) AS t(
            datetime,
            Global_active_power,
            Global_reactive_power,
            Voltage,
            Global_intensity,
            Sub_metering_1,
            Sub_metering_2,
            Sub_metering_3
        )
        """
    )


def test_energy_chart_functions_create_output_files(tmp_path):
    con = duckdb.connect(":memory:")
    _create_energy_table(con)

    chart_functions = [
        plot_daily_active_power_trend,
        plot_hourly_consumption_pattern,
        plot_power_distribution,
        plot_voltage_distribution,
        plot_correlation_heatmap,
        plot_sub_metering_comparison,
    ]

    try:
        for chart_function in chart_functions:
            output_path = tmp_path / f"{chart_function.__name__}.png"
            saved_path = chart_function(con, output_path)

            assert saved_path == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    finally:
        con.close()


def test_generate_all_energy_charts_returns_metadata(tmp_path):
    db_path = tmp_path / "analytics.duckdb"
    output_dir = tmp_path / "figures"
    con = duckdb.connect(str(db_path))
    try:
        _create_energy_table(con)
    finally:
        con.close()

    chart_metadata = generate_all_energy_charts(db_path=db_path, output_dir=output_dir)

    assert len(chart_metadata) == len(ENERGY_CHART_REGISTRY)
    assert {chart["chart_id"] for chart in chart_metadata} == set(ENERGY_CHART_REGISTRY)
    for chart in chart_metadata:
        chart_path = Path(chart["path"])
        assert chart_path.exists()
        assert chart_path.stat().st_size > 0
        assert chart["title"]
        assert chart["description"]
