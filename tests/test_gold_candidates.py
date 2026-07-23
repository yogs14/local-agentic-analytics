import datetime as dt
import random
from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.gold_candidates import (
    SUPPORTED_TEMPLATES,
    EnergyPools,
    FinancePools,
    compute_hardness,
    format_date_id,
    generate_energy_candidates,
    generate_finance_candidates,
    recompute_expected,
)


def _energy_pools() -> EnergyPools:
    clean_days = [
        dt.date(2007, 1, 1) + dt.timedelta(days=index * 7)
        for index in range(60)
    ]
    full_months = [
        (year, month) for year in (2007, 2008, 2009) for month in range(1, 13)
    ]
    return EnergyPools(
        clean_days=clean_days,
        full_months=full_months,
        full_years=[2007, 2008, 2009],
        gap_median=0.6,
        gap_p90=2.5,
        voltage_median=241.0,
        intensity_p90=10.0,
        monthly_avg_gap_median=1.1,
        monthly_avg_intensity_median=4.6,
    )


def _finance_pools() -> FinancePools:
    return FinancePools(
        tickers=["GOOGL", "NFLX", "NVDA", "TSLA"],
        months=[(2019, month) for month in range(1, 13)]
        + [(2020, month) for month in range(1, 6)],
        min_date=dt.date(2019, 1, 2),
        max_date=dt.date(2020, 6, 9),
        close_median={"GOOGL": 60.0, "NFLX": 320.0, "NVDA": 45.0, "TSLA": 50.0},
    )


# ---------------------------------------------------------------------------
# Hardness
# ---------------------------------------------------------------------------


def test_compute_hardness_levels():
    assert compute_hardness("SELECT AVG(v) FROM t WHERE d = '2007-01-01'") == "easy"
    assert (
        compute_hardness(
            "SELECT AVG(v) FROM t WHERE y = 2007 AND m = 1"
        )
        == "medium"
    )
    assert (
        compute_hardness(
            "SELECT d, SUM(v) FROM t WHERE y = 2007 GROUP BY d "
            "ORDER BY 2 DESC LIMIT 5"
        )
        == "hard"
    )
    window_sql = (
        "WITH daily AS (SELECT d, SUM(v) AS e FROM t GROUP BY d) "
        "SELECT d, e - LAG(e) OVER (ORDER BY d) AS delta FROM daily "
        "ORDER BY delta DESC LIMIT 1"
    )
    assert compute_hardness(window_sql) == "extra"


def test_format_date_id():
    assert format_date_id(dt.date(2007, 1, 15)) == "15 Januari 2007"


# ---------------------------------------------------------------------------
# Generation determinism and template coverage
# ---------------------------------------------------------------------------


def test_generation_is_deterministic():
    first = generate_energy_candidates(_energy_pools(), random.Random(42))
    second = generate_energy_candidates(_energy_pools(), random.Random(42))

    assert [c.id for c in first] == [c.id for c in second]
    assert [c.sql for c in first] == [c.sql for c in second]
    assert [c.question for c in first] == [c.question for c in second]


def test_generated_counts_meet_targets():
    energy = generate_energy_candidates(_energy_pools(), random.Random(42))
    finance = generate_finance_candidates(_finance_pools(), random.Random(42))

    assert len(energy) >= 64  # 36 inherited elsewhere -> >= 100 total
    assert len(finance) >= 22  # 8 inherited elsewhere -> >= 30 total
    assert len({c.id for c in energy}) == len(energy)
    assert len({c.id for c in finance}) == len(finance)


def test_every_generated_template_has_a_pandas_handler():
    energy = generate_energy_candidates(_energy_pools(), random.Random(1))
    finance = generate_finance_candidates(_finance_pools(), random.Random(1))

    for candidate in energy + finance:
        assert candidate.template in SUPPORTED_TEMPLATES
        assert candidate.question
        assert candidate.sql
        assert candidate.expected_unit
        item = candidate.to_manifest_item("references/sql_gold")
        assert item["verified"] is False
        assert item["hardness"] in ("easy", "medium", "hard", "extra")


def test_recompute_expected_rejects_unknown_template():
    with pytest.raises(KeyError):
        recompute_expected("nope", {}, pd.DataFrame())


# ---------------------------------------------------------------------------
# Pandas handlers (independent recomputation) spot checks
# ---------------------------------------------------------------------------


def _tiny_energy_df() -> pd.DataFrame:
    timestamps = (
        [pd.Timestamp(2007, 1, 15, 0, minute) for minute in range(3)]
        + [pd.Timestamp(2007, 1, 16, 18, minute) for minute in range(3)]
    )
    return pd.DataFrame(
        {
            "datetime": timestamps,
            "Global_active_power": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Voltage": [240.0, 241.0, 242.0, 239.0, 238.0, 240.0],
            "Global_intensity": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "Sub_metering_1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Sub_metering_2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Sub_metering_3": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )


def test_energy_daily_agg_handler():
    result = recompute_expected(
        "energy_daily_agg",
        {"agg": "AVG", "column": "Global_active_power", "date": "2007-01-15"},
        _tiny_energy_df(),
    )
    assert result.iat[0, 0] == pytest.approx(2.0)


def test_energy_count_threshold_handler():
    result = recompute_expected(
        "energy_count_threshold",
        {"column": "Global_active_power", "threshold": 4.5,
         "date": "2007-01-16"},
        _tiny_energy_df(),
    )
    assert result.iat[0, 0] == 2


def test_energy_hour_filter_handler():
    result = recompute_expected(
        "energy_hour_filter",
        {"column": "Voltage", "date": "2007-01-16", "h1": 18, "h2": 19},
        _tiny_energy_df(),
    )
    assert result.iat[0, 0] == pytest.approx((239.0 + 238.0 + 240.0) / 3)


def test_energy_multi_col_avg_handler():
    result = recompute_expected(
        "energy_multi_col_avg",
        {"date": "2007-01-15"},
        _tiny_energy_df(),
    )
    assert list(result.columns) == ["avg_sub1", "avg_sub2", "avg_sub3"]
    assert result.iat[0, 0] == pytest.approx(1.0)


def _tiny_finance_df() -> pd.DataFrame:
    dates = [dt.date(2019, 1, 2), dt.date(2019, 1, 3), dt.date(2019, 1, 4)]
    return pd.DataFrame(
        {
            "date": dates * 2,
            "ticker": ["NVDA"] * 3 + ["TSLA"] * 3,
            "open": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            "high": [11.0, 12.0, 13.0, 21.0, 22.0, 23.0],
            "low": [9.0, 10.0, 11.0, 19.0, 20.0, 21.0],
            "close": [10.5, 11.5, 12.5, 20.5, 19.5, 21.5],
            "volume": [100, 200, 300, 400, 500, 600],
        }
    )


def test_finance_range_agg_handler():
    result = recompute_expected(
        "finance_range_agg",
        {"agg": "AVG", "field": "close", "ticker": "NVDA",
         "start": "2019-01-02", "end": "2019-01-04"},
        _tiny_finance_df(),
    )
    assert result.iat[0, 0] == pytest.approx((10.5 + 11.5 + 12.5) / 3)


def test_finance_daily_return_handler():
    result = recompute_expected(
        "finance_daily_return",
        {"ticker": "TSLA", "direction": "DESC"},
        _tiny_finance_df(),
    )
    # TSLA returns: -4.878% then +10.256%; DESC -> the +10.256% day.
    assert result.iloc[0]["date"] == dt.date(2019, 1, 4)
    assert result.iloc[0]["daily_return_pct"] == pytest.approx(
        100.0 * (21.5 - 19.5) / 19.5
    )


def test_finance_ticker_compare_handler():
    result = recompute_expected(
        "finance_ticker_compare",
        {"start": "2019-01-02", "end": "2019-01-04"},
        _tiny_finance_df(),
    )
    assert list(result["ticker"]) == ["NVDA", "TSLA"]
    assert result.iloc[1]["avg_close"] == pytest.approx((20.5 + 19.5 + 21.5) / 3)
