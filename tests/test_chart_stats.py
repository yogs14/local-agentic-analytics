from pathlib import Path
import sys

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.visualization.chart_stats import (
    CHART_STATS_REGISTRY,
    get_correlation_stats,
    get_daily_active_power_stats,
    get_hourly_consumption_stats,
    get_power_distribution_stats,
    get_sub_metering_stats,
    get_voltage_distribution_stats,
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


def test_chart_stats_functions_return_compact_metrics():
    con = duckdb.connect(":memory:")
    _create_energy_table(con)

    try:
        daily_stats = get_daily_active_power_stats(con)
        hourly_stats = get_hourly_consumption_stats(con)
        power_stats = get_power_distribution_stats(con)
        voltage_stats = get_voltage_distribution_stats(con)
        correlation_stats = get_correlation_stats(con)
        sub_metering_stats = get_sub_metering_stats(con)
    finally:
        con.close()

    assert daily_stats["day_count"] == 2
    assert daily_stats["max_daily_avg_date"] == "2006-12-17"
    assert daily_stats["max_daily_avg_kw"] == 3.5
    assert daily_stats["unit"] == "kW"

    assert hourly_stats["hour_count"] == 2
    assert hourly_stats["max_avg_hour"] == 1
    assert hourly_stats["max_hourly_avg_kw"] == 3.0

    assert power_stats["record_count"] == 4
    assert power_stats["avg_global_active_power_kw"] == 2.5
    assert power_stats["mode_bin_kw"] == 1.0

    assert voltage_stats["avg_voltage_v"] == 241.5
    assert voltage_stats["unit"] == "Volt"

    assert correlation_stats["pair_count"] == 21
    assert correlation_stats["correlations"][
        "Global_active_power vs Global_intensity"
    ] == 1.0

    assert sub_metering_stats["highest_average_meter"] == "Sub_metering_3"
    assert sub_metering_stats["highest_average_wh"] == 18.5


def test_chart_stats_registry_covers_all_energy_charts():
    assert set(CHART_STATS_REGISTRY) == {
        "daily_active_power_trend",
        "hourly_consumption_pattern",
        "power_distribution",
        "voltage_distribution",
        "correlation_heatmap",
        "sub_metering_comparison",
    }
