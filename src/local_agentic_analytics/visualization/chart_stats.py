"""Compact DuckDB statistics for energy chart insights."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import math
from typing import Any


ENERGY_TABLE = "electric_power"
NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def _fetch_one(con: Any, sql: str) -> dict[str, Any]:
    try:
        cursor = con.execute(sql)
        row = cursor.fetchone()
    except Exception as exc:
        raise ValueError(f"Failed to run chart stats query: {exc}") from exc

    if row is None:
        return {}

    columns = [description[0] for description in cursor.description]
    return {
        column: _to_json_value(value)
        for column, value in zip(columns, row)
    }


def _to_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    return value


def get_daily_active_power_stats(con: Any) -> dict:
    query = f"""
        WITH daily AS (
            SELECT
                CAST(datetime AS DATE) AS reading_date,
                AVG(Global_active_power) AS avg_global_active_power_kw
            FROM {ENERGY_TABLE}
            WHERE datetime IS NOT NULL
              AND Global_active_power IS NOT NULL
            GROUP BY reading_date
        )
        SELECT
            COUNT(*) AS day_count,
            CAST(MIN(reading_date) AS VARCHAR) AS first_date,
            CAST(MAX(reading_date) AS VARCHAR) AS last_date,
            AVG(avg_global_active_power_kw) AS mean_daily_avg_kw,
            MIN(avg_global_active_power_kw) AS min_daily_avg_kw,
            MAX(avg_global_active_power_kw) AS max_daily_avg_kw,
            arg_min(CAST(reading_date AS VARCHAR), avg_global_active_power_kw)
                AS min_daily_avg_date,
            arg_max(CAST(reading_date AS VARCHAR), avg_global_active_power_kw)
                AS max_daily_avg_date
        FROM daily
    """
    stats = _fetch_one(con, query)
    stats["unit"] = "kW"
    return stats


def get_hourly_consumption_stats(con: Any) -> dict:
    query = f"""
        WITH hourly AS (
            SELECT
                CAST(EXTRACT(hour FROM datetime) AS INTEGER) AS reading_hour,
                AVG(Global_active_power) AS avg_global_active_power_kw
            FROM {ENERGY_TABLE}
            WHERE datetime IS NOT NULL
              AND Global_active_power IS NOT NULL
            GROUP BY reading_hour
        )
        SELECT
            COUNT(*) AS hour_count,
            AVG(avg_global_active_power_kw) AS mean_hourly_avg_kw,
            MIN(avg_global_active_power_kw) AS min_hourly_avg_kw,
            MAX(avg_global_active_power_kw) AS max_hourly_avg_kw,
            arg_min(reading_hour, avg_global_active_power_kw) AS min_avg_hour,
            arg_max(reading_hour, avg_global_active_power_kw) AS max_avg_hour
        FROM hourly
    """
    stats = _fetch_one(con, query)
    stats["unit"] = "kW"
    return stats


def get_power_distribution_stats(con: Any) -> dict:
    query = f"""
        SELECT
            COUNT(*) AS record_count,
            MIN(Global_active_power) AS min_global_active_power_kw,
            MAX(Global_active_power) AS max_global_active_power_kw,
            AVG(Global_active_power) AS avg_global_active_power_kw,
            stddev_samp(Global_active_power) AS stddev_global_active_power_kw,
            (
                SELECT FLOOR(Global_active_power * 10) / 10.0
                FROM {ENERGY_TABLE}
                WHERE Global_active_power IS NOT NULL
                GROUP BY 1
                ORDER BY COUNT(*) DESC, 1
                LIMIT 1
            ) AS mode_bin_kw,
            (
                SELECT COUNT(*)
                FROM {ENERGY_TABLE}
                WHERE Global_active_power IS NOT NULL
                GROUP BY FLOOR(Global_active_power * 10) / 10.0
                ORDER BY COUNT(*) DESC, FLOOR(Global_active_power * 10) / 10.0
                LIMIT 1
            ) AS mode_bin_count
        FROM {ENERGY_TABLE}
        WHERE Global_active_power IS NOT NULL
    """
    stats = _fetch_one(con, query)
    stats["unit"] = "kW"
    stats["bin_width_kw"] = 0.1
    return stats


def get_voltage_distribution_stats(con: Any) -> dict:
    query = f"""
        SELECT
            COUNT(*) AS record_count,
            MIN(Voltage) AS min_voltage_v,
            MAX(Voltage) AS max_voltage_v,
            AVG(Voltage) AS avg_voltage_v,
            stddev_samp(Voltage) AS stddev_voltage_v,
            (
                SELECT FLOOR(Voltage)
                FROM {ENERGY_TABLE}
                WHERE Voltage IS NOT NULL
                GROUP BY 1
                ORDER BY COUNT(*) DESC, 1
                LIMIT 1
            ) AS mode_bin_v,
            (
                SELECT COUNT(*)
                FROM {ENERGY_TABLE}
                WHERE Voltage IS NOT NULL
                GROUP BY FLOOR(Voltage)
                ORDER BY COUNT(*) DESC, FLOOR(Voltage)
                LIMIT 1
            ) AS mode_bin_count
        FROM {ENERGY_TABLE}
        WHERE Voltage IS NOT NULL
    """
    stats = _fetch_one(con, query)
    stats["unit"] = "Volt"
    stats["bin_width_v"] = 1.0
    return stats


def get_correlation_stats(con: Any) -> dict:
    select_parts = []
    aliases = {}
    for left_idx, left_column in enumerate(NUMERIC_COLUMNS):
        for right_idx, right_column in enumerate(NUMERIC_COLUMNS):
            alias = f"corr_{left_idx}_{right_idx}"
            aliases[(left_idx, right_idx)] = alias
            select_parts.append(f"corr({left_column}, {right_column}) AS {alias}")

    query = f"""
        SELECT
            {", ".join(select_parts)}
        FROM {ENERGY_TABLE}
    """
    row = _fetch_one(con, query)

    correlations = {}
    pair_values = []
    for left_idx, left_column in enumerate(NUMERIC_COLUMNS):
        for right_idx, right_column in enumerate(NUMERIC_COLUMNS):
            if left_idx >= right_idx:
                continue
            pair_name = f"{left_column} vs {right_column}"
            value = row.get(aliases[(left_idx, right_idx)])
            correlations[pair_name] = value
            if isinstance(value, (int, float)):
                pair_values.append(
                    {
                        "pair": pair_name,
                        "correlation": value,
                    }
                )

    strongest_positive = None
    strongest_negative = None
    strongest_absolute = None
    if pair_values:
        strongest_positive = max(pair_values, key=lambda item: item["correlation"])
        strongest_negative = min(pair_values, key=lambda item: item["correlation"])
        strongest_absolute = max(
            pair_values,
            key=lambda item: abs(item["correlation"]),
        )

    return {
        "numeric_columns": NUMERIC_COLUMNS,
        "pair_count": len(correlations),
        "correlations": correlations,
        "strongest_positive": strongest_positive,
        "strongest_negative": strongest_negative,
        "strongest_absolute": strongest_absolute,
        "unit_note": (
            "Korelasi tidak memiliki satuan; kolom sumber memakai kW, Volt, "
            "Ampere, dan Wh sesuai definisi dataset."
        ),
    }


def get_sub_metering_stats(con: Any) -> dict:
    query = f"""
        SELECT
            COUNT(*) AS record_count,
            AVG(Sub_metering_1) AS avg_sub_metering_1_wh,
            AVG(Sub_metering_2) AS avg_sub_metering_2_wh,
            AVG(Sub_metering_3) AS avg_sub_metering_3_wh,
            SUM(Sub_metering_1) AS total_sub_metering_1_wh,
            SUM(Sub_metering_2) AS total_sub_metering_2_wh,
            SUM(Sub_metering_3) AS total_sub_metering_3_wh
        FROM {ENERGY_TABLE}
    """
    stats = _fetch_one(con, query)
    average_values = {
        "Sub_metering_1": stats.get("avg_sub_metering_1_wh"),
        "Sub_metering_2": stats.get("avg_sub_metering_2_wh"),
        "Sub_metering_3": stats.get("avg_sub_metering_3_wh"),
    }
    valid_average_values = {
        meter: value
        for meter, value in average_values.items()
        if isinstance(value, (int, float))
    }
    if valid_average_values:
        highest_meter = max(valid_average_values, key=valid_average_values.get)
        stats["highest_average_meter"] = highest_meter
        stats["highest_average_wh"] = valid_average_values[highest_meter]
    else:
        stats["highest_average_meter"] = None
        stats["highest_average_wh"] = None
    stats["unit"] = "Wh"
    return stats


CHART_STATS_REGISTRY = {
    "daily_active_power_trend": get_daily_active_power_stats,
    "hourly_consumption_pattern": get_hourly_consumption_stats,
    "power_distribution": get_power_distribution_stats,
    "voltage_distribution": get_voltage_distribution_stats,
    "correlation_heatmap": get_correlation_stats,
    "sub_metering_comparison": get_sub_metering_stats,
}
