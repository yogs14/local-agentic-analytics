from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.finetune.value_sampler import (
    ENERGY_CHARTS,
    SAMPLER_REGISTRY,
    format_stats_block,
    parse_stats_block,
    sample_variant,
)


def test_registry_covers_all_energy_charts():
    assert set(SAMPLER_REGISTRY) == set(ENERGY_CHARTS)


def test_sampling_is_deterministic_for_a_fixed_seed():
    for chart_id in ENERGY_CHARTS:
        a = sample_variant(chart_id, random.Random(7))
        b = sample_variant(chart_id, random.Random(7))
        assert a == b


def test_min_mean_max_ordering_holds_for_distribution_charts():
    rng = random.Random(123)
    for _ in range(50):
        daily = sample_variant("daily_active_power_trend", rng)
        assert (
            daily["min_daily_avg_kw"]
            < daily["mean_daily_avg_kw"]
            < daily["max_daily_avg_kw"]
        )

        hourly = sample_variant("hourly_consumption_pattern", rng)
        assert (
            hourly["min_hourly_avg_kw"]
            < hourly["mean_hourly_avg_kw"]
            < hourly["max_hourly_avg_kw"]
        )

        power = sample_variant("power_distribution", rng)
        assert (
            power["min_global_active_power_kw"]
            < power["avg_global_active_power_kw"]
            < power["max_global_active_power_kw"]
        )
        assert power["stddev_global_active_power_kw"] > 0

        voltage = sample_variant("voltage_distribution", rng)
        assert (
            voltage["min_voltage_v"]
            < voltage["avg_voltage_v"]
            < voltage["max_voltage_v"]
        )
        assert voltage["stddev_voltage_v"] > 0


def test_hours_fall_in_plausible_dayparts():
    rng = random.Random(99)
    for _ in range(30):
        hourly = sample_variant("hourly_consumption_pattern", rng)
        assert 2 <= hourly["min_avg_hour"] <= 5
        assert 18 <= hourly["max_avg_hour"] <= 21


def test_correlations_stay_in_valid_range():
    rng = random.Random(5)
    for _ in range(50):
        corr = sample_variant("correlation_heatmap", rng)
        for key, value in corr.items():
            if key.startswith("corr_") or key.endswith("_r"):
                assert -1.0 <= value <= 1.0
        # The strongest positive is meaningfully positive; negative is negative.
        assert corr["strongest_positive_r"] > 0
        assert corr["strongest_negative_r"] < 0


def test_sub_metering_three_dominates():
    rng = random.Random(11)
    for _ in range(20):
        sub = sample_variant("sub_metering_comparison", rng)
        assert sub["highest_average_meter"] == "Sub_metering_3"
        assert sub["avg_sub_metering_3_wh"] >= sub["avg_sub_metering_1_wh"]
        assert sub["avg_sub_metering_3_wh"] >= sub["avg_sub_metering_2_wh"]


def test_format_and_parse_block_round_trip():
    variant = sample_variant("daily_active_power_trend", random.Random(3))
    block = format_stats_block(variant)
    assert "chart_id: daily_active_power_trend" in block

    parsed = parse_stats_block(block)
    assert parsed["chart_id"] == "daily_active_power_trend"
    assert parsed["unit"] == "kW"
    # Numeric values survive as their compact string form.
    assert parsed["day_count"] == str(variant["day_count"])
