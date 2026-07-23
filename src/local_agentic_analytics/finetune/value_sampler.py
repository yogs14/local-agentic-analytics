"""Programmatic value variation for energy chart statistics.

Each sampler produces a *flat*, internally consistent statistics block for one
chart type. Values are sampled around the real ``chart_insights.json`` ranges so
the teacher (and later the student) see plausible-but-varied inputs. Everything
is driven by an injected ``random.Random`` so generation is fully deterministic
under a fixed seed.

Invariants guaranteed per variant:
- ``min < mean/avg < max`` wherever those fields coexist
- standard deviation ``> 0``
- correlation coefficients ``r in [-1, 1]``
- peak/min hours fall in plausible day-parts
"""

from __future__ import annotations

from datetime import date, timedelta
import random
from typing import Any, Callable


ENERGY_CHARTS: tuple[str, ...] = (
    "daily_active_power_trend",
    "hourly_consumption_pattern",
    "power_distribution",
    "voltage_distribution",
    "correlation_heatmap",
    "sub_metering_comparison",
)

CHART_TITLES: dict[str, str] = {
    "daily_active_power_trend": "Tren Rata-rata Daya Aktif Harian",
    "hourly_consumption_pattern": "Pola Konsumsi Daya per Jam",
    "power_distribution": "Distribusi Global Active Power",
    "voltage_distribution": "Distribusi Voltage",
    "correlation_heatmap": "Korelasi Fitur Numerik Energi",
    "sub_metering_comparison": "Perbandingan Rata-rata Sub Metering",
}

# Charts whose narratives must carry an electrical unit near a number. The
# correlation heatmap is exempt because Pearson r is dimensionless.
UNIT_REQUIRED: dict[str, bool] = {chart: True for chart in ENERGY_CHARTS}
UNIT_REQUIRED["correlation_heatmap"] = False

_DATASET_START = date(2006, 12, 16)
_DATASET_END = date(2010, 11, 26)


def _round(value: float, digits: int = 4) -> float:
    return round(value, digits)


def _random_date(rng: random.Random, start: date, end: date) -> date:
    span = (end - start).days
    if span <= 0:
        return start
    return start + timedelta(days=rng.randint(0, span))


def _iso(d: date) -> str:
    return d.isoformat()


def sample_daily_active_power(rng: random.Random) -> dict[str, Any]:
    mean = _round(rng.uniform(0.8, 1.3))
    minimum = _round(rng.uniform(0.15, mean * 0.6))
    maximum = _round(rng.uniform(mean * 1.8, 3.5))
    day_count = rng.randint(320, 1450)

    first = _random_date(rng, _DATASET_START, _DATASET_START + timedelta(days=40))
    span = min(day_count + rng.randint(0, 60), (_DATASET_END - first).days)
    last = first + timedelta(days=max(span, 1))
    min_date = _random_date(rng, first, last)
    max_date = _random_date(rng, first, last)

    return {
        "chart_id": "daily_active_power_trend",
        "day_count": day_count,
        "first_date": _iso(first),
        "last_date": _iso(last),
        "mean_daily_avg_kw": mean,
        "min_daily_avg_kw": minimum,
        "min_daily_avg_date": _iso(min_date),
        "max_daily_avg_kw": maximum,
        "max_daily_avg_date": _iso(max_date),
        "unit": "kW",
    }


def sample_hourly_consumption(rng: random.Random) -> dict[str, Any]:
    mean = _round(rng.uniform(0.9, 1.2))
    minimum = _round(rng.uniform(0.4, mean * 0.7))
    maximum = _round(rng.uniform(mean * 1.4, 2.2))
    return {
        "chart_id": "hourly_consumption_pattern",
        "hour_count": 24,
        "mean_hourly_avg_kw": mean,
        "min_hourly_avg_kw": minimum,
        "min_avg_hour": rng.randint(2, 5),
        "max_hourly_avg_kw": maximum,
        "max_avg_hour": rng.randint(18, 21),
        "unit": "kW",
    }


def sample_power_distribution(rng: random.Random) -> dict[str, Any]:
    avg = _round(rng.uniform(0.9, 1.3))
    stddev = _round(rng.uniform(0.8, 1.3))
    minimum = _round(rng.uniform(0.05, 0.2), 3)
    maximum = _round(rng.uniform(8.0, 11.5), 3)
    mode_bin = rng.choice([0.1, 0.2, 0.3])
    return {
        "chart_id": "power_distribution",
        "record_count": rng.randint(1_900_000, 2_100_000),
        "min_global_active_power_kw": minimum,
        "max_global_active_power_kw": maximum,
        "avg_global_active_power_kw": avg,
        "stddev_global_active_power_kw": stddev,
        "mode_bin_kw": mode_bin,
        "mode_bin_count": rng.randint(250_000, 400_000),
        "bin_width_kw": 0.1,
        "unit": "kW",
    }


def sample_voltage_distribution(rng: random.Random) -> dict[str, Any]:
    avg = _round(rng.uniform(238.0, 242.0))
    stddev = _round(rng.uniform(2.5, 4.0))
    minimum = _round(rng.uniform(222.0, 228.0), 2)
    maximum = _round(rng.uniform(250.0, 256.0), 2)
    mode_bin = float(rng.choice([round(avg) - 1, round(avg), round(avg) + 1]))
    return {
        "chart_id": "voltage_distribution",
        "record_count": rng.randint(1_900_000, 2_100_000),
        "min_voltage_v": minimum,
        "max_voltage_v": maximum,
        "avg_voltage_v": avg,
        "stddev_voltage_v": stddev,
        "mode_bin_v": mode_bin,
        "mode_bin_count": rng.randint(200_000, 320_000),
        "bin_width_v": 1.0,
        "unit": "Volt",
    }


def sample_correlation_heatmap(rng: random.Random) -> dict[str, Any]:
    pos_r = _round(rng.uniform(0.95, 0.999))
    neg_r = _round(rng.uniform(-0.5, -0.3))
    secondary_r = _round(rng.uniform(0.5, 0.7))
    return {
        "chart_id": "correlation_heatmap",
        "pair_count": 21,
        "corr_active_power_vs_intensity": pos_r,
        "corr_voltage_vs_intensity": neg_r,
        "corr_active_power_vs_submetering3": secondary_r,
        "strongest_positive_pair": "Global_active_power vs Global_intensity",
        "strongest_positive_r": pos_r,
        "strongest_negative_pair": "Voltage vs Global_intensity",
        "strongest_negative_r": neg_r,
        "unit_note": "tanpa satuan (korelasi tidak berdimensi)",
    }


def sample_sub_metering(rng: random.Random) -> dict[str, Any]:
    avg1 = _round(rng.uniform(0.8, 1.6))
    avg2 = _round(rng.uniform(0.9, 1.8))
    avg3 = _round(rng.uniform(4.5, 8.0))
    record_count = rng.randint(2_000_000, 2_100_000)
    return {
        "chart_id": "sub_metering_comparison",
        "record_count": record_count,
        "avg_sub_metering_1_wh": avg1,
        "avg_sub_metering_2_wh": avg2,
        "avg_sub_metering_3_wh": avg3,
        "total_sub_metering_1_wh": _round(avg1 * record_count, 1),
        "total_sub_metering_2_wh": _round(avg2 * record_count, 1),
        "total_sub_metering_3_wh": _round(avg3 * record_count, 1),
        "highest_average_meter": "Sub_metering_3",
        "highest_average_wh": avg3,
        "unit": "Wh",
    }


SAMPLER_REGISTRY: dict[str, Callable[[random.Random], dict[str, Any]]] = {
    "daily_active_power_trend": sample_daily_active_power,
    "hourly_consumption_pattern": sample_hourly_consumption,
    "power_distribution": sample_power_distribution,
    "voltage_distribution": sample_voltage_distribution,
    "correlation_heatmap": sample_correlation_heatmap,
    "sub_metering_comparison": sample_sub_metering,
}


def sample_variant(chart_id: str, rng: random.Random) -> dict[str, Any]:
    """Sample one internally consistent statistics variant for ``chart_id``."""
    try:
        sampler = SAMPLER_REGISTRY[chart_id]
    except KeyError as exc:
        raise KeyError(f"Unknown chart id: {chart_id}") from exc
    return sampler(rng)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        # Indonesian locale: comma decimal separator, no thousand grouping. This
        # keeps the "tuliskan angka persis seperti di input" rule coherent.
        return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", ",")
    return str(value)


def format_stats_block(variant: dict[str, Any]) -> str:
    """Render a variant as a compact ``key: value`` block.

    This block is both the teacher prompt's injected statistics and the JSONL
    ``input`` field, so the only numbers a faithful narrative may use are the
    ones rendered here (plus whitelisted derived values).
    """
    return "\n".join(f"{key}: {_format_value(value)}" for key, value in variant.items())


def parse_stats_block(block: str) -> dict[str, str]:
    """Parse a ``key: value`` block back into an ordered string mapping."""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields
