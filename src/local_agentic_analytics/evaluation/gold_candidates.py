"""Template-based gold SQL candidate generation for the expanded gold sets.

Fase 3 expands the gold sets (energy 36 -> 100+, finance 8 -> 30+). Every new
candidate is produced from a parameterized template, so that:

- generation is deterministic (seeded RNG + parameter pools read from the
  real database),
- ``scripts/verify_gold_v3.py`` can recompute the expected result with an
  INDEPENDENT pandas implementation of the same template (never by executing
  the SQL again), and
- SPIDER-like ``hardness`` is derived from the SQL AST, not hand-assigned.

All generated candidates carry ``verified: false`` — a human flips them to
``true`` after reviewing the executed + cross-checked results. Existing gold
sets (energy v2, finance v1) are inherited untouched into the new manifests
with their original SQL files.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
import sqlglot
from sqlglot import expressions as exp


ENERGY_TABLE = "electric_power"
FINANCE_TABLE = "stock_prices"

# measure column -> (Indonesian phrase, unit)
ENERGY_MEASURES: dict[str, tuple[str, str]] = {
    "Global_active_power": ("daya aktif", "kW"),
    "Global_reactive_power": ("daya reaktif", "kW"),
    "Voltage": ("tegangan", "V"),
    "Global_intensity": ("intensitas arus", "A"),
    "Sub_metering_1": ("Sub_metering_1", "Wh"),
    "Sub_metering_2": ("Sub_metering_2", "Wh"),
    "Sub_metering_3": ("Sub_metering_3", "Wh"),
}

AGG_NAMES_ID: dict[str, str] = {
    "AVG": "rata-rata",
    "MAX": "maksimum",
    "MIN": "minimum",
}

MONTH_NAMES_ID = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}

FINANCE_PRICE_FIELDS: dict[str, str] = {
    "close": "harga penutupan",
    "open": "harga pembukaan",
    "high": "harga tertinggi harian",
    "low": "harga terendah harian",
}

HARDNESS_LEVELS = ("easy", "medium", "hard", "extra")


@dataclass
class GoldCandidate:
    """One generated gold question with its machine-checkable spec."""

    id: str
    question: str
    sql: str
    expected_unit: str
    category: str
    result_shape: str
    template: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_manifest_item(self, sql_dir_prefix: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "gold_sql_file": f"{sql_dir_prefix}/{self.id}.sql",
            "expected_unit": self.expected_unit,
            "category": self.category,
            "difficulty": compute_hardness(self.sql),
            "hardness": compute_hardness(self.sql),
            "result_shape": self.result_shape,
            "resolver_eligible": "no",
            "verified": False,
            "source": "generated_v3",
            "template": self.template,
            "params": self.params,
        }


# ---------------------------------------------------------------------------
# SPIDER-like hardness from the SQL AST
# ---------------------------------------------------------------------------


def compute_hardness(sql: str) -> str:
    """Approximate SPIDER hardness from SQL component counts.

    Documented approximation of the SPIDER criteria (component counting):

    - ``comp1`` counts "simple" components beyond a minimal query: extra
      SELECT expressions, extra WHERE predicates, GROUP BY, ORDER BY, LIMIT,
      extra aggregates.
    - ``comp2`` counts "advanced" components: HAVING, window functions,
      subqueries/CTEs, set operations.

    easy: comp1 == 0 and comp2 == 0; medium: comp1 <= 2 and comp2 == 0;
    hard: comp2 == 1 or comp1 >= 3; extra: comp2 >= 2, or comp2 >= 1 with
    comp1 >= 3.
    """
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        return "hard"
    if tree is None:
        return "hard"

    selects = list(tree.find_all(exp.Select))
    outer = selects[0] if selects else None

    n_select = len(outer.expressions) if outer is not None else 1
    n_agg = len(list(tree.find_all(exp.AggFunc)))
    n_where_predicates = _count_where_predicates(tree)
    has_group = tree.find(exp.Group) is not None
    has_order = tree.find(exp.Order) is not None
    has_limit = tree.find(exp.Limit) is not None
    has_having = tree.find(exp.Having) is not None
    has_window = tree.find(exp.Window) is not None
    n_subqueries = max(len(selects) - 1, 0) + len(list(tree.find_all(exp.Union)))

    comp1 = (
        max(n_select - 1, 0)
        + max(n_where_predicates - 1, 0)
        + max(n_agg - 1, 0)
        + (1 if has_group else 0)
        + (1 if has_order else 0)
        + (1 if has_limit else 0)
    )
    comp2 = (
        (1 if has_having else 0)
        + (1 if has_window else 0)
        + min(n_subqueries, 2)
    )

    if comp2 >= 2 or (comp2 >= 1 and comp1 >= 3):
        return "extra"
    if comp2 == 1 or comp1 >= 3:
        return "hard"
    if comp1 >= 1:
        return "medium"
    return "easy"


def _count_where_predicates(tree: exp.Expression) -> int:
    where = tree.find(exp.Where)
    if where is None:
        return 0
    predicates = 1
    for node in where.walk():
        if isinstance(node, (exp.And, exp.Or)):
            predicates += 1
    return predicates


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_date_id(date: dt.date) -> str:
    return f"{date.day} {MONTH_NAMES_ID[date.month]} {date.year}"


def format_month_id(year: int, month: int) -> str:
    return f"{MONTH_NAMES_ID[month]} {year}"


def _iso(date: dt.date) -> str:
    return date.isoformat()


def _num(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


# ---------------------------------------------------------------------------
# Parameter pools (queried from the real database by the generator script)
# ---------------------------------------------------------------------------


@dataclass
class EnergyPools:
    """Deterministic parameter pools for the energy templates.

    ``clean_days`` must be full days (1440 rows) with no NULL in any measure
    column so every generated query has a well-defined, non-degenerate answer.
    """

    clean_days: list[dt.date]
    full_months: list[tuple[int, int]]  # (year, month) with >= 27 clean days
    full_years: list[int]
    gap_median: float
    gap_p90: float
    voltage_median: float
    intensity_p90: float
    # HAVING thresholds must be on the monthly-average scale (a minute-level
    # p90 is far above any monthly mean and yields empty results).
    monthly_avg_gap_median: float = 1.0
    monthly_avg_intensity_median: float = 4.0


@dataclass
class FinancePools:
    tickers: list[str]
    months: list[tuple[int, int]]  # (year, month) fully inside the data range
    min_date: dt.date
    max_date: dt.date
    close_median: dict[str, float]


ENERGY_POOLS_SQL = {
    "clean_days": f"""
        SELECT CAST(datetime AS DATE) AS day
        FROM {ENERGY_TABLE}
        GROUP BY day
        HAVING COUNT(*) = 1440
           AND COUNT(Global_active_power) = 1440
           AND COUNT(Global_reactive_power) = 1440
           AND COUNT(Voltage) = 1440
           AND COUNT(Global_intensity) = 1440
           AND COUNT(Sub_metering_1) = 1440
           AND COUNT(Sub_metering_2) = 1440
           AND COUNT(Sub_metering_3) = 1440
        ORDER BY day
    """,
    "gap_quantiles": f"""
        SELECT
            quantile_cont(Global_active_power, 0.5) AS median,
            quantile_cont(Global_active_power, 0.9) AS p90
        FROM {ENERGY_TABLE}
    """,
    "voltage_median": f"""
        SELECT quantile_cont(Voltage, 0.5) FROM {ENERGY_TABLE}
    """,
    "intensity_p90": f"""
        SELECT quantile_cont(Global_intensity, 0.9) FROM {ENERGY_TABLE}
    """,
    "monthly_avg_medians": f"""
        WITH monthly AS (
            SELECT
                EXTRACT(YEAR FROM datetime) AS year,
                EXTRACT(MONTH FROM datetime) AS month,
                AVG(Global_active_power) AS avg_gap,
                AVG(Global_intensity) AS avg_intensity
            FROM {ENERGY_TABLE}
            GROUP BY year, month
        )
        SELECT
            quantile_cont(avg_gap, 0.5),
            quantile_cont(avg_intensity, 0.5)
        FROM monthly
    """,
}


def build_energy_pools(
    clean_days: list[dt.date],
    gap_median: float,
    gap_p90: float,
    voltage_median: float,
    intensity_p90: float,
    data_min_day: dt.date | None = None,
    data_max_day: dt.date | None = None,
    monthly_avg_gap_median: float = 1.0,
    monthly_avg_intensity_median: float = 4.0,
) -> EnergyPools:
    """Derive month/year pools from the clean-day list.

    ``full_months`` (used by day-precision templates) require >= 27 clean
    days. ``full_years`` only require the calendar year to lie inside the
    data range: year-level aggregates skip NULLs, so gap days do not make
    them degenerate.
    """
    by_month: dict[tuple[int, int], int] = {}
    for day in clean_days:
        key = (day.year, day.month)
        by_month[key] = by_month.get(key, 0) + 1
    full_months = sorted(key for key, count in by_month.items() if count >= 27)

    data_min_day = data_min_day or min(clean_days)
    data_max_day = data_max_day or max(clean_days)
    full_years = [
        year
        for year in range(data_min_day.year, data_max_day.year + 1)
        if dt.date(year, 1, 1) >= data_min_day
        and dt.date(year, 12, 31) <= data_max_day
    ]

    return EnergyPools(
        clean_days=clean_days,
        full_months=full_months,
        full_years=full_years,
        gap_median=round(gap_median, 1),
        gap_p90=round(gap_p90, 1),
        voltage_median=round(voltage_median, 0),
        intensity_p90=round(intensity_p90, 0),
        monthly_avg_gap_median=round(monthly_avg_gap_median, 2),
        monthly_avg_intensity_median=round(monthly_avg_intensity_median, 2),
    )


# ---------------------------------------------------------------------------
# Energy templates
# ---------------------------------------------------------------------------


def generate_energy_candidates(
    pools: EnergyPools,
    rng: Any,
    start_number: int = 201,
) -> list[GoldCandidate]:
    """Generate the energy v3 candidates (deterministic given pools + rng)."""
    candidates: list[GoldCandidate] = []
    counter = [start_number]

    def next_id() -> str:
        value = counter[0]
        counter[0] += 1
        return f"E{value}"

    day_pool = list(pools.clean_days)
    rng.shuffle(day_pool)
    day_iter = iter(day_pool)

    def next_day() -> dt.date:
        return next(day_iter)

    month_pool = list(pools.full_months)
    rng.shuffle(month_pool)
    month_iter = iter(month_pool)

    def next_month() -> tuple[int, int]:
        return next(month_iter)

    measures = list(ENERGY_MEASURES)

    # 1. daily_agg (12): AGG(measure) on one clean day -> scalar
    for agg in ("AVG", "MAX", "MIN"):
        for column in ("Global_active_power", "Voltage", "Global_intensity",
                       "Global_reactive_power"):
            day = next_day()
            name_id, unit = ENERGY_MEASURES[column]
            candidates.append(GoldCandidate(
                id=next_id(),
                question=(
                    f"Berapa {AGG_NAMES_ID[agg]} {name_id} pada tanggal "
                    f"{format_date_id(day)}?"
                ),
                sql=(
                    f"SELECT {agg}({column}) AS value\n"
                    f"FROM {ENERGY_TABLE}\n"
                    f"WHERE CAST(datetime AS DATE) = DATE '{_iso(day)}';"
                ),
                expected_unit=unit,
                category="daily_aggregation",
                result_shape="scalar",
                template="energy_daily_agg",
                params={"agg": agg, "column": column, "date": _iso(day)},
            ))

    # 2. daily_energy (3): SUM(gap)/60 kWh on one day -> scalar
    for _ in range(3):
        day = next_day()
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa total energi kWh pada tanggal {format_date_id(day)} "
                "berdasarkan Global_active_power?"
            ),
            sql=(
                f"SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE CAST(datetime AS DATE) = DATE '{_iso(day)}';"
            ),
            expected_unit="kWh",
            category="daily_aggregation",
            result_shape="scalar",
            template="energy_daily_energy",
            params={"date": _iso(day)},
        ))

    # 3. daily_sub_sum (3): SUM(Sub_metering_k) on one day -> scalar (Wh)
    for k in (1, 2, 3):
        day = next_day()
        column = f"Sub_metering_{k}"
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa total konsumsi {column} dalam Wh pada tanggal "
                f"{format_date_id(day)}?"
            ),
            sql=(
                f"SELECT SUM({column}) AS total_wh\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE CAST(datetime AS DATE) = DATE '{_iso(day)}';"
            ),
            expected_unit="Wh",
            category="daily_aggregation",
            result_shape="scalar",
            template="energy_daily_sub_sum",
            params={"column": column, "date": _iso(day)},
        ))

    # 4. daily_range (2): MAX - MIN on one day -> scalar
    for column in ("Voltage", "Global_active_power"):
        day = next_day()
        name_id, unit = ENERGY_MEASURES[column]
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa selisih antara {name_id} maksimum dan minimum pada "
                f"tanggal {format_date_id(day)}?"
            ),
            sql=(
                f"SELECT MAX({column}) - MIN({column}) AS value_range\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE CAST(datetime AS DATE) = DATE '{_iso(day)}';"
            ),
            expected_unit=unit,
            category="statistics",
            result_shape="scalar",
            template="energy_daily_range",
            params={"column": column, "date": _iso(day)},
        ))

    # 5. daily_stat (3): STDDEV / MEDIAN on one day -> scalar
    for func, phrase in (
        ("STDDEV_SAMP", "standar deviasi"),
        ("MEDIAN", "median"),
        ("STDDEV_SAMP", "standar deviasi"),
    ):
        day = next_day()
        column = "Voltage" if func == "MEDIAN" else "Global_active_power"
        name_id, unit = ENERGY_MEASURES[column]
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa {phrase} {name_id} pada tanggal "
                f"{format_date_id(day)}?"
            ),
            sql=(
                f"SELECT {func}({column}) AS value\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE CAST(datetime AS DATE) = DATE '{_iso(day)}';"
            ),
            expected_unit=unit,
            category="statistics",
            result_shape="scalar",
            template="energy_daily_stat",
            params={"func": func, "column": column, "date": _iso(day)},
        ))

    # 6. monthly_agg (6): AVG/MAX(measure) over one month -> scalar
    for agg in ("AVG", "MAX"):
        for column in ("Global_active_power", "Voltage", "Global_intensity"):
            year, month = next_month()
            name_id, unit = ENERGY_MEASURES[column]
            candidates.append(GoldCandidate(
                id=next_id(),
                question=(
                    f"Berapa {AGG_NAMES_ID[agg]} {name_id} sepanjang bulan "
                    f"{format_month_id(year, month)}?"
                ),
                sql=(
                    f"SELECT {agg}({column}) AS value\n"
                    f"FROM {ENERGY_TABLE}\n"
                    f"WHERE EXTRACT(YEAR FROM datetime) = {year}\n"
                    f"  AND EXTRACT(MONTH FROM datetime) = {month};"
                ),
                expected_unit=unit,
                category="monthly_aggregation",
                result_shape="scalar",
                template="energy_monthly_agg",
                params={"agg": agg, "column": column, "year": year,
                        "month": month},
            ))

    # 7. monthly_energy (3): SUM/60 over one month -> scalar
    for _ in range(3):
        year, month = next_month()
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa total energi kWh sepanjang bulan "
                f"{format_month_id(year, month)}?"
            ),
            sql=(
                f"SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE EXTRACT(YEAR FROM datetime) = {year}\n"
                f"  AND EXTRACT(MONTH FROM datetime) = {month};"
            ),
            expected_unit="kWh",
            category="monthly_aggregation",
            result_shape="scalar",
            template="energy_monthly_energy",
            params={"year": year, "month": month},
        ))

    # 8. hour_filter (4): AVG(measure) between two hours on one day -> scalar
    for column in ("Global_active_power", "Voltage",
                   "Global_intensity", "Global_active_power"):
        day = next_day()
        h1 = rng.choice([6, 8, 10, 17, 18])
        h2 = h1 + rng.choice([2, 3])
        name_id, unit = ENERGY_MEASURES[column]
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa rata-rata {name_id} pada rentang jam {h1} sampai "
                f"{h2} di tanggal {format_date_id(day)}?"
            ),
            sql=(
                f"SELECT AVG({column}) AS value\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE CAST(datetime AS DATE) = DATE '{_iso(day)}'\n"
                f"  AND EXTRACT(HOUR FROM datetime) BETWEEN {h1} AND {h2};"
            ),
            expected_unit=unit,
            category="hour_filter",
            result_shape="scalar",
            template="energy_hour_filter",
            params={"column": column, "date": _iso(day), "h1": h1, "h2": h2},
        ))

    # 9. count_threshold (4): COUNT(*) above threshold, day or month -> scalar
    threshold_specs = [
        ("Global_active_power", pools.gap_p90, "kW", "day"),
        ("Global_intensity", pools.intensity_p90, "A", "day"),
        ("Global_active_power", pools.gap_p90, "kW", "month"),
        ("Voltage", pools.voltage_median, "V", "month"),
    ]
    for column, threshold, unit, scope in threshold_specs:
        name_id, _ = ENERGY_MEASURES[column]
        if scope == "day":
            day = next_day()
            where = f"CAST(datetime AS DATE) = DATE '{_iso(day)}'"
            scope_id = f"pada tanggal {format_date_id(day)}"
            params: dict[str, Any] = {"date": _iso(day)}
        else:
            year, month = next_month()
            where = (
                f"EXTRACT(YEAR FROM datetime) = {year}\n"
                f"  AND EXTRACT(MONTH FROM datetime) = {month}"
            )
            scope_id = f"sepanjang bulan {format_month_id(year, month)}"
            params = {"year": year, "month": month}
        params.update({"column": column, "threshold": threshold})
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa jumlah record dengan {name_id} di atas "
                f"{_num(threshold)} {unit} {scope_id}?"
            ),
            sql=(
                f"SELECT COUNT(*) AS record_count\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE {where}\n"
                f"  AND {column} > {threshold};"
            ),
            expected_unit="records",
            category="multi_condition",
            result_shape="scalar",
            template="energy_count_threshold",
            params=params,
        ))

    # 10. percent_threshold (2): percentage of minutes above threshold -> scalar
    for _ in range(2):
        year, month = next_month()
        threshold = pools.gap_median
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa persentase menit dengan daya aktif di atas "
                f"{_num(threshold)} kW sepanjang bulan "
                f"{format_month_id(year, month)}?"
            ),
            sql=(
                "SELECT 100.0 * SUM(CASE WHEN Global_active_power > "
                f"{threshold} THEN 1 ELSE 0 END) / COUNT(*) AS pct\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE EXTRACT(YEAR FROM datetime) = {year}\n"
                f"  AND EXTRACT(MONTH FROM datetime) = {month};"
            ),
            expected_unit="percent",
            category="threshold",
            result_shape="scalar",
            template="energy_percent_threshold",
            params={"year": year, "month": month, "threshold": threshold},
        ))

    # 11. per_day_in_month (3): per-day AVG in one month -> table
    for column in ("Global_active_power", "Voltage", "Global_intensity"):
        year, month = next_month()
        name_id, unit = ENERGY_MEASURES[column]
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Tampilkan rata-rata {name_id} per hari sepanjang bulan "
                f"{format_month_id(year, month)}."
            ),
            sql=(
                f"SELECT CAST(datetime AS DATE) AS day, "
                f"AVG({column}) AS avg_value\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE EXTRACT(YEAR FROM datetime) = {year}\n"
                f"  AND EXTRACT(MONTH FROM datetime) = {month}\n"
                f"GROUP BY day\n"
                f"ORDER BY day;"
            ),
            expected_unit=unit,
            category="daily_aggregation",
            result_shape="table",
            template="energy_per_day_in_month",
            params={"column": column, "year": year, "month": month},
        ))

    # 12. top_n_days (2): top-N days by total energy in one year -> table
    for n in (5, 3):
        year = rng.choice(pools.full_years)
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Tampilkan {n} tanggal dengan total energi kWh tertinggi "
                f"sepanjang tahun {year}."
            ),
            sql=(
                f"SELECT CAST(datetime AS DATE) AS day, "
                f"SUM(Global_active_power) / 60.0 AS total_energy_kwh\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE EXTRACT(YEAR FROM datetime) = {year}\n"
                f"GROUP BY day\n"
                f"ORDER BY total_energy_kwh DESC, day ASC\n"
                f"LIMIT {n};"
            ),
            expected_unit="kWh",
            category="groupby_limit",
            result_shape="table",
            template="energy_top_n_days",
            params={"year": year, "n": n},
        ))

    # 13. top_n_hours (2): top-N hours by avg power in one month -> table
    for n in (3, 5):
        year, month = next_month()
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Tampilkan {n} jam dengan rata-rata daya aktif tertinggi "
                f"sepanjang bulan {format_month_id(year, month)}."
            ),
            sql=(
                f"SELECT EXTRACT(HOUR FROM datetime) AS hour, "
                f"AVG(Global_active_power) AS avg_kw\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE EXTRACT(YEAR FROM datetime) = {year}\n"
                f"  AND EXTRACT(MONTH FROM datetime) = {month}\n"
                f"GROUP BY hour\n"
                f"ORDER BY avg_kw DESC, hour ASC\n"
                f"LIMIT {n};"
            ),
            expected_unit="kW",
            category="groupby_limit",
            result_shape="table",
            template="energy_top_n_hours",
            params={"year": year, "month": month, "n": n},
        ))

    # 14. having_months (3): months in a year with AVG above threshold -> table
    # Thresholds come from the median of MONTHLY AVERAGES so roughly half the
    # months qualify (discriminative, never empty).
    for column, threshold in (
        ("Global_active_power", pools.monthly_avg_gap_median),
        ("Global_intensity", pools.monthly_avg_intensity_median),
        ("Voltage", pools.voltage_median),
    ):
        year = rng.choice(pools.full_years)
        name_id, unit = ENERGY_MEASURES[column]
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Tampilkan bulan-bulan pada tahun {year} yang "
                f"{AGG_NAMES_ID['AVG']} {name_id}-nya di atas "
                f"{_num(threshold)} {unit}."
            ),
            sql=(
                f"SELECT EXTRACT(MONTH FROM datetime) AS month, "
                f"AVG({column}) AS avg_value\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE EXTRACT(YEAR FROM datetime) = {year}\n"
                f"GROUP BY month\n"
                f"HAVING AVG({column}) > {threshold}\n"
                f"ORDER BY month;"
            ),
            expected_unit=unit,
            category="groupby_having",
            result_shape="table",
            template="energy_having_months",
            params={"column": column, "year": year, "threshold": threshold},
        ))

    # 15. year_compare (2): AVG per year over the full years -> table
    for column in ("Global_active_power", "Voltage"):
        years = list(pools.full_years)
        name_id, unit = ENERGY_MEASURES[column]
        year_list = ", ".join(str(year) for year in years)
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Bandingkan rata-rata {name_id} untuk setiap tahun "
                f"{year_list}."
            ),
            sql=(
                f"SELECT EXTRACT(YEAR FROM datetime) AS year, "
                f"AVG({column}) AS avg_value\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE EXTRACT(YEAR FROM datetime) IN ({year_list})\n"
                f"GROUP BY year\n"
                f"ORDER BY year;"
            ),
            expected_unit=unit,
            category="period_comparison",
            result_shape="table",
            template="energy_year_compare",
            params={"column": column, "years": years},
        ))

    # 16. month_across_years (2): AVG of one month across years -> table
    for column in ("Global_active_power", "Voltage"):
        month = rng.choice([1, 2, 7])
        years = list(pools.full_years)
        name_id, unit = ENERGY_MEASURES[column]
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Bandingkan rata-rata {name_id} bulan "
                f"{MONTH_NAMES_ID[month]} pada tahun "
                f"{', '.join(str(y) for y in years)}."
            ),
            sql=(
                f"SELECT EXTRACT(YEAR FROM datetime) AS year, "
                f"AVG({column}) AS avg_value\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE EXTRACT(MONTH FROM datetime) = {month}\n"
                f"  AND EXTRACT(YEAR FROM datetime) IN "
                f"({', '.join(str(y) for y in years)})\n"
                f"GROUP BY year\n"
                f"ORDER BY year;"
            ),
            expected_unit=unit,
            category="period_comparison",
            result_shape="table",
            template="energy_month_across_years",
            params={"column": column, "month": month, "years": years},
        ))

    # 17. multi_col_avg (2): AVG of the three sub meterings -> row
    for scope in ("day", "month"):
        if scope == "day":
            day = next_day()
            where = f"CAST(datetime AS DATE) = DATE '{_iso(day)}'"
            scope_id = f"pada tanggal {format_date_id(day)}"
            params = {"date": _iso(day)}
        else:
            year, month = next_month()
            where = (
                f"EXTRACT(YEAR FROM datetime) = {year}\n"
                f"  AND EXTRACT(MONTH FROM datetime) = {month}"
            )
            scope_id = f"sepanjang bulan {format_month_id(year, month)}"
            params = {"year": year, "month": month}
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa rata-rata masing-masing Sub_metering_1, "
                f"Sub_metering_2, dan Sub_metering_3 {scope_id}?"
            ),
            sql=(
                "SELECT AVG(Sub_metering_1) AS avg_sub1, "
                "AVG(Sub_metering_2) AS avg_sub2, "
                "AVG(Sub_metering_3) AS avg_sub3\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE {where};"
            ),
            expected_unit="Wh",
            category="aggregation",
            result_shape="row",
            template="energy_multi_col_avg",
            params=params,
        ))

    # 18. argmax_time (4): timestamp of extreme value in month/year -> row
    argmax_specs = [
        ("Global_active_power", "MAX", "tertinggi"),
        ("Voltage", "MIN", "terendah"),
        ("Global_intensity", "MAX", "tertinggi"),
        ("Global_active_power", "MAX", "tertinggi"),
    ]
    for index, (column, direction, phrase) in enumerate(argmax_specs):
        name_id, _ = ENERGY_MEASURES[column]
        if index % 2 == 0:
            year, month = next_month()
            where = (
                f"EXTRACT(YEAR FROM datetime) = {year}\n"
                f"  AND EXTRACT(MONTH FROM datetime) = {month}"
            )
            scope_id = f"pada bulan {format_month_id(year, month)}"
            params = {"year": year, "month": month}
        else:
            year = rng.choice(pools.full_years)
            where = f"EXTRACT(YEAR FROM datetime) = {year}"
            scope_id = f"pada tahun {year}"
            params = {"year": year}
        order = "DESC" if direction == "MAX" else "ASC"
        params.update({"column": column, "direction": direction})
        candidates.append(GoldCandidate(
            id=next_id(),
            question=f"Kapan {name_id} {phrase} {scope_id} terjadi?",
            sql=(
                f"SELECT datetime, {column}\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE {where}\n"
                f"  AND {column} IS NOT NULL\n"
                f"ORDER BY {column} {order}, datetime ASC\n"
                f"LIMIT 1;"
            ),
            expected_unit="timestamp",
            category="extremes",
            result_shape="row",
            template="energy_argmax_time",
            params=params,
        ))

    # 19. window_delta_day (3): largest day-over-day energy increase -> row
    for _ in range(3):
        year, month = next_month()
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                "Pada tanggal berapa terjadi kenaikan total energi harian "
                "(kWh) terbesar dibanding hari sebelumnya sepanjang bulan "
                f"{format_month_id(year, month)}?"
            ),
            sql=(
                "WITH daily AS (\n"
                "    SELECT CAST(datetime AS DATE) AS day,\n"
                "           SUM(Global_active_power) / 60.0 AS energy_kwh\n"
                f"    FROM {ENERGY_TABLE}\n"
                f"    WHERE EXTRACT(YEAR FROM datetime) = {year}\n"
                f"      AND EXTRACT(MONTH FROM datetime) = {month}\n"
                "    GROUP BY day\n"
                ")\n"
                "SELECT day, energy_kwh - LAG(energy_kwh) OVER (ORDER BY day) "
                "AS delta_kwh\n"
                "FROM daily\n"
                "QUALIFY delta_kwh IS NOT NULL\n"
                "ORDER BY delta_kwh DESC, day ASC\n"
                "LIMIT 1;"
            ),
            expected_unit="kWh",
            category="window_function",
            result_shape="row",
            template="energy_window_delta_day",
            params={"year": year, "month": month},
        ))

    # 20. distinct_days (1): unique days recorded in one year -> scalar
    year = rng.choice(pools.full_years)
    candidates.append(GoldCandidate(
        id=next_id(),
        question=(
            f"Berapa jumlah hari unik yang tercatat dalam dataset pada "
            f"tahun {year}?"
        ),
        sql=(
            "SELECT COUNT(DISTINCT CAST(datetime AS DATE)) AS unique_days\n"
            f"FROM {ENERGY_TABLE}\n"
            f"WHERE EXTRACT(YEAR FROM datetime) = {year};"
        ),
        expected_unit="count",
        category="statistics",
        result_shape="scalar",
        template="energy_distinct_days",
        params={"year": year},
    ))

    # 21. weekday_avg (2): AVG per ISO weekday in one year -> table
    for column in ("Global_active_power", "Voltage"):
        year = rng.choice(pools.full_years)
        name_id, unit = ENERGY_MEASURES[column]
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Tampilkan rata-rata {name_id} untuk setiap hari dalam "
                f"seminggu (ISO, 1=Senin) sepanjang tahun {year}."
            ),
            sql=(
                f"SELECT EXTRACT(ISODOW FROM datetime) AS weekday, "
                f"AVG({column}) AS avg_value\n"
                f"FROM {ENERGY_TABLE}\n"
                f"WHERE EXTRACT(YEAR FROM datetime) = {year}\n"
                f"GROUP BY weekday\n"
                f"ORDER BY weekday;"
            ),
            expected_unit=unit,
            category="weekly_aggregation",
            result_shape="table",
            template="energy_weekday_avg",
            params={"column": column, "year": year},
        ))

    return candidates


# ---------------------------------------------------------------------------
# Finance templates
# ---------------------------------------------------------------------------


def generate_finance_candidates(
    pools: FinancePools,
    rng: Any,
    start_number: int = 101,
) -> list[GoldCandidate]:
    """Generate the finance v2 candidates (deterministic given pools + rng)."""
    candidates: list[GoldCandidate] = []
    counter = [start_number]

    def next_id() -> str:
        value = counter[0]
        counter[0] += 1
        return f"F{value}"

    def month_range(year: int, month: int) -> tuple[str, str]:
        first = dt.date(year, month, 1)
        if month == 12:
            last = dt.date(year, 12, 31)
        else:
            last = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
        return _iso(first), _iso(last)

    month_pool = list(pools.months)
    rng.shuffle(month_pool)
    month_iter = iter(month_pool)

    def next_month() -> tuple[int, int]:
        return next(month_iter)

    def pick_ticker() -> str:
        return rng.choice(pools.tickers)

    # 1. range_agg_close (6): AVG/MAX/MIN price field over a month -> scalar
    for agg, phrase in (
        ("AVG", "rata-rata"),
        ("MAX", "tertinggi"),
        ("MIN", "terendah"),
    ):
        for price_field in ("close", "open"):
            ticker = pick_ticker()
            year, month = next_month()
            start, end = month_range(year, month)
            field_id = FINANCE_PRICE_FIELDS[price_field]
            candidates.append(GoldCandidate(
                id=next_id(),
                question=(
                    f"Berapa {field_id} {phrase} {ticker} sepanjang bulan "
                    f"{format_month_id(year, month)}?"
                ),
                sql=(
                    f"SELECT {agg}({price_field}) AS value\n"
                    f"FROM {FINANCE_TABLE}\n"
                    f"WHERE ticker = '{ticker}'\n"
                    f"  AND date BETWEEN DATE '{start}' AND DATE '{end}';"
                ),
                expected_unit="USD",
                category="range_aggregation",
                result_shape="scalar",
                template="finance_range_agg",
                params={"agg": agg, "field": price_field, "ticker": ticker,
                        "start": start, "end": end},
            ))

    # 2. volume_agg (3): AVG/SUM volume over a month -> scalar
    for agg, phrase in (("AVG", "rata-rata"), ("SUM", "total"),
                        ("MAX", "tertinggi")):
        ticker = pick_ticker()
        year, month = next_month()
        start, end = month_range(year, month)
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa volume perdagangan {phrase} {ticker} sepanjang "
                f"bulan {format_month_id(year, month)}?"
            ),
            sql=(
                f"SELECT {agg}(volume) AS value\n"
                f"FROM {FINANCE_TABLE}\n"
                f"WHERE ticker = '{ticker}'\n"
                f"  AND date BETWEEN DATE '{start}' AND DATE '{end}';"
            ),
            expected_unit="shares",
            category="range_aggregation",
            result_shape="scalar",
            template="finance_volume_agg",
            params={"agg": agg, "ticker": ticker, "start": start, "end": end},
        ))

    # 3. count_days (2): trading days for a ticker in a month -> scalar
    for _ in range(2):
        ticker = pick_ticker()
        year, month = next_month()
        start, end = month_range(year, month)
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa jumlah hari perdagangan {ticker} pada bulan "
                f"{format_month_id(year, month)}?"
            ),
            sql=(
                "SELECT COUNT(*) AS trading_days\n"
                f"FROM {FINANCE_TABLE}\n"
                f"WHERE ticker = '{ticker}'\n"
                f"  AND date BETWEEN DATE '{start}' AND DATE '{end}';"
            ),
            expected_unit="count",
            category="range_aggregation",
            result_shape="scalar",
            template="finance_count_days",
            params={"ticker": ticker, "start": start, "end": end},
        ))

    # 4. monthly_close (3): monthly AVG close for a ticker in 2019 -> table
    for _ in range(3):
        ticker = pick_ticker()
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Tampilkan rata-rata harga penutupan {ticker} per bulan "
                f"sepanjang tahun 2019."
            ),
            sql=(
                "SELECT EXTRACT(MONTH FROM date) AS month, "
                "AVG(close) AS avg_close\n"
                f"FROM {FINANCE_TABLE}\n"
                f"WHERE ticker = '{ticker}'\n"
                "  AND EXTRACT(YEAR FROM date) = 2019\n"
                "GROUP BY month\n"
                "ORDER BY month;"
            ),
            expected_unit="USD",
            category="monthly_aggregation",
            result_shape="table",
            template="finance_monthly_close",
            params={"ticker": ticker, "year": 2019},
        ))

    # 5. top_volume_days (2): top-N dates by volume -> table
    for n in (5, 3):
        ticker = pick_ticker()
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Tampilkan {n} tanggal dengan volume perdagangan {ticker} "
                f"tertinggi selama periode dataset."
            ),
            sql=(
                "SELECT date, volume\n"
                f"FROM {FINANCE_TABLE}\n"
                f"WHERE ticker = '{ticker}'\n"
                "ORDER BY volume DESC, date ASC\n"
                f"LIMIT {n};"
            ),
            expected_unit="shares",
            category="groupby_limit",
            result_shape="table",
            template="finance_top_volume_days",
            params={"ticker": ticker, "n": n},
        ))

    # 6. argmax_close (2): date of the highest/lowest close -> row
    for direction, phrase in (("MAX", "tertinggi"), ("MIN", "terendah")):
        ticker = pick_ticker()
        order = "DESC" if direction == "MAX" else "ASC"
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Pada tanggal berapa harga penutupan {phrase} {ticker} "
                f"selama periode dataset terjadi?"
            ),
            sql=(
                "SELECT date, close\n"
                f"FROM {FINANCE_TABLE}\n"
                f"WHERE ticker = '{ticker}'\n"
                f"ORDER BY close {order}, date ASC\n"
                "LIMIT 1;"
            ),
            expected_unit="timestamp",
            category="extremes",
            result_shape="row",
            template="finance_argmax_close",
            params={"ticker": ticker, "direction": direction},
        ))

    # 7. ticker_compare (2): AVG close per ticker over a month -> table
    for _ in range(2):
        year, month = next_month()
        start, end = month_range(year, month)
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                "Bandingkan rata-rata harga penutupan semua ticker "
                f"sepanjang bulan {format_month_id(year, month)}."
            ),
            sql=(
                "SELECT ticker, AVG(close) AS avg_close\n"
                f"FROM {FINANCE_TABLE}\n"
                f"WHERE date BETWEEN DATE '{start}' AND DATE '{end}'\n"
                "GROUP BY ticker\n"
                "ORDER BY ticker;"
            ),
            expected_unit="USD",
            category="period_comparison",
            result_shape="table",
            template="finance_ticker_compare",
            params={"start": start, "end": end},
        ))

    # 8. having_months_close (2): months with AVG close above median -> table
    for _ in range(2):
        ticker = pick_ticker()
        threshold = round(pools.close_median[ticker], 0)
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Tampilkan bulan-bulan pada tahun 2019 yang rata-rata "
                f"harga penutupan {ticker}-nya di atas {_num(threshold)} USD."
            ),
            sql=(
                "SELECT EXTRACT(MONTH FROM date) AS month, "
                "AVG(close) AS avg_close\n"
                f"FROM {FINANCE_TABLE}\n"
                f"WHERE ticker = '{ticker}'\n"
                "  AND EXTRACT(YEAR FROM date) = 2019\n"
                "GROUP BY month\n"
                f"HAVING AVG(close) > {threshold}\n"
                "ORDER BY month;"
            ),
            expected_unit="USD",
            category="groupby_having",
            result_shape="table",
            template="finance_having_months_close",
            params={"ticker": ticker, "year": 2019, "threshold": threshold},
        ))

    # 9. daily_return (3): largest daily percentage move (window) -> row
    for direction, phrase in (
        ("DESC", "kenaikan"),
        ("ASC", "penurunan"),
        ("DESC", "kenaikan"),
    ):
        ticker = pick_ticker()
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Pada tanggal berapa {phrase} harian harga penutupan "
                f"{ticker} terbesar (persen) selama periode dataset terjadi?"
            ),
            sql=(
                "WITH ordered AS (\n"
                "    SELECT date, close,\n"
                "           LAG(close) OVER (ORDER BY date) AS prev_close\n"
                f"    FROM {FINANCE_TABLE}\n"
                f"    WHERE ticker = '{ticker}'\n"
                ")\n"
                "SELECT date, 100.0 * (close - prev_close) / prev_close "
                "AS daily_return_pct\n"
                "FROM ordered\n"
                "WHERE prev_close IS NOT NULL\n"
                f"ORDER BY daily_return_pct {direction}, date ASC\n"
                "LIMIT 1;"
            ),
            expected_unit="percent",
            category="window_function",
            result_shape="row",
            template="finance_daily_return",
            params={"ticker": ticker, "direction": direction},
        ))

    # 10. spread_avg (2): AVG (high - low) over a month -> scalar
    for _ in range(2):
        ticker = pick_ticker()
        year, month = next_month()
        start, end = month_range(year, month)
        candidates.append(GoldCandidate(
            id=next_id(),
            question=(
                f"Berapa rata-rata selisih harga tertinggi dan terendah "
                f"harian {ticker} sepanjang bulan "
                f"{format_month_id(year, month)}?"
            ),
            sql=(
                "SELECT AVG(high - low) AS avg_spread\n"
                f"FROM {FINANCE_TABLE}\n"
                f"WHERE ticker = '{ticker}'\n"
                f"  AND date BETWEEN DATE '{start}' AND DATE '{end}';"
            ),
            expected_unit="USD",
            category="range_aggregation",
            result_shape="scalar",
            template="finance_spread_avg",
            params={"ticker": ticker, "start": start, "end": end},
        ))

    # 11. close_above_count (1): days with close above threshold -> scalar
    ticker = pick_ticker()
    threshold = round(pools.close_median[ticker], 0)
    candidates.append(GoldCandidate(
        id=next_id(),
        question=(
            f"Berapa jumlah hari perdagangan dengan harga penutupan {ticker} "
            f"di atas {_num(threshold)} USD selama periode dataset?"
        ),
        sql=(
            "SELECT COUNT(*) AS day_count\n"
            f"FROM {FINANCE_TABLE}\n"
            f"WHERE ticker = '{ticker}'\n"
            f"  AND close > {threshold};"
        ),
        expected_unit="count",
        category="threshold",
        result_shape="scalar",
        template="finance_close_above_count",
        params={"ticker": ticker, "threshold": threshold},
    ))

    return candidates


# ---------------------------------------------------------------------------
# Independent pandas recomputation (used by scripts/verify_gold_v3.py)
# ---------------------------------------------------------------------------


def recompute_expected(
    template: str,
    params: dict[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute a candidate's expected result with pandas only.

    ``df`` is the full raw table (energy or finance) loaded once by the
    verifier. The SQL is never consulted here — this is the independent
    cross-check. Returns a DataFrame shaped like the SQL result.
    """
    handler = _PANDAS_HANDLERS.get(template)
    if handler is None:
        raise KeyError(f"No pandas handler for template '{template}'")
    return handler(params, df)


def _frame(value: Any, column: str = "value") -> pd.DataFrame:
    return pd.DataFrame({column: [value]})


def _energy_day_slice(df: pd.DataFrame, date_iso: str) -> pd.DataFrame:
    # dt.normalize() keeps datetime64 vectors; dt.date would materialize ~2M
    # Python date objects, which can exhaust memory on the 8 GB target laptop.
    return df[df["datetime"].dt.normalize() == pd.Timestamp(date_iso)]


def _energy_month_slice(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    return df[(df["datetime"].dt.year == year) & (df["datetime"].dt.month == month)]


def _daily_energy(df_slice: pd.DataFrame) -> pd.Series:
    grouped = df_slice.groupby(df_slice["datetime"].dt.normalize())[
        "Global_active_power"
    ].sum(min_count=1)
    return grouped / 60.0


def _agg(series: pd.Series, agg: str) -> float:
    if agg == "AVG":
        return series.mean()
    if agg == "MAX":
        return series.max()
    if agg == "MIN":
        return series.min()
    if agg == "SUM":
        return series.sum(min_count=1)
    raise ValueError(f"Unsupported aggregate: {agg}")


def _h_energy_daily_agg(params, df):
    day = _energy_day_slice(df, params["date"])
    return _frame(_agg(day[params["column"]], params["agg"]))


def _h_energy_daily_energy(params, df):
    day = _energy_day_slice(df, params["date"])
    return _frame(day["Global_active_power"].sum(min_count=1) / 60.0)


def _h_energy_daily_sub_sum(params, df):
    day = _energy_day_slice(df, params["date"])
    return _frame(day[params["column"]].sum(min_count=1))


def _h_energy_daily_range(params, df):
    day = _energy_day_slice(df, params["date"])
    series = day[params["column"]]
    return _frame(series.max() - series.min())


def _h_energy_daily_stat(params, df):
    day = _energy_day_slice(df, params["date"])
    series = day[params["column"]].dropna()
    if params["func"] == "MEDIAN":
        return _frame(series.median())
    return _frame(series.std(ddof=1))


def _h_energy_monthly_agg(params, df):
    month = _energy_month_slice(df, params["year"], params["month"])
    return _frame(_agg(month[params["column"]], params["agg"]))


def _h_energy_monthly_energy(params, df):
    month = _energy_month_slice(df, params["year"], params["month"])
    return _frame(month["Global_active_power"].sum(min_count=1) / 60.0)


def _h_energy_hour_filter(params, df):
    day = _energy_day_slice(df, params["date"])
    hours = day["datetime"].dt.hour
    mask = (hours >= params["h1"]) & (hours <= params["h2"])
    return _frame(day.loc[mask, params["column"]].mean())


def _h_energy_count_threshold(params, df):
    if "date" in params:
        scope = _energy_day_slice(df, params["date"])
    else:
        scope = _energy_month_slice(df, params["year"], params["month"])
    count = int((scope[params["column"]] > params["threshold"]).sum())
    return _frame(count)


def _h_energy_percent_threshold(params, df):
    scope = _energy_month_slice(df, params["year"], params["month"])
    above = (scope["Global_active_power"] > params["threshold"]).sum()
    return _frame(100.0 * above / len(scope))


def _h_energy_per_day_in_month(params, df):
    scope = _energy_month_slice(df, params["year"], params["month"])
    grouped = scope.groupby(scope["datetime"].dt.normalize())[
        params["column"]
    ].mean()
    return pd.DataFrame(
        {"day": list(grouped.index), "avg_value": grouped.to_numpy()}
    )


def _h_energy_top_n_days(params, df):
    scope = df[df["datetime"].dt.year == params["year"]]
    daily = _daily_energy(scope).reset_index()
    daily.columns = ["day", "total_energy_kwh"]
    daily = daily.sort_values(
        ["total_energy_kwh", "day"], ascending=[False, True]
    )
    return daily.head(params["n"]).reset_index(drop=True)


def _h_energy_top_n_hours(params, df):
    scope = _energy_month_slice(df, params["year"], params["month"])
    grouped = scope.groupby(scope["datetime"].dt.hour)[
        "Global_active_power"
    ].mean().reset_index()
    grouped.columns = ["hour", "avg_kw"]
    grouped = grouped.sort_values(["avg_kw", "hour"], ascending=[False, True])
    return grouped.head(params["n"]).reset_index(drop=True)


def _h_energy_having_months(params, df):
    scope = df[df["datetime"].dt.year == params["year"]]
    grouped = scope.groupby(scope["datetime"].dt.month)[params["column"]].mean()
    grouped = grouped[grouped > params["threshold"]].reset_index()
    grouped.columns = ["month", "avg_value"]
    return grouped.sort_values("month").reset_index(drop=True)


def _h_energy_year_compare(params, df):
    scope = df[df["datetime"].dt.year.isin(params["years"])]
    grouped = scope.groupby(scope["datetime"].dt.year)[params["column"]].mean()
    result = grouped.reset_index()
    result.columns = ["year", "avg_value"]
    return result.sort_values("year").reset_index(drop=True)


def _h_energy_month_across_years(params, df):
    scope = df[
        (df["datetime"].dt.month == params["month"])
        & (df["datetime"].dt.year.isin(params["years"]))
    ]
    grouped = scope.groupby(scope["datetime"].dt.year)[params["column"]].mean()
    result = grouped.reset_index()
    result.columns = ["year", "avg_value"]
    return result.sort_values("year").reset_index(drop=True)


def _h_energy_multi_col_avg(params, df):
    if "date" in params:
        scope = _energy_day_slice(df, params["date"])
    else:
        scope = _energy_month_slice(df, params["year"], params["month"])
    return pd.DataFrame(
        {
            "avg_sub1": [scope["Sub_metering_1"].mean()],
            "avg_sub2": [scope["Sub_metering_2"].mean()],
            "avg_sub3": [scope["Sub_metering_3"].mean()],
        }
    )


def _h_energy_argmax_time(params, df):
    if "month" in params:
        scope = _energy_month_slice(df, params["year"], params["month"])
    else:
        scope = df[df["datetime"].dt.year == params["year"]]
    scope = scope.dropna(subset=[params["column"]])
    ascending = params["direction"] == "MIN"
    ordered = scope.sort_values(
        [params["column"], "datetime"], ascending=[ascending, True]
    )
    top = ordered.iloc[0]
    return pd.DataFrame(
        {"datetime": [top["datetime"]], params["column"]: [top[params["column"]]]}
    )


def _h_energy_window_delta_day(params, df):
    scope = _energy_month_slice(df, params["year"], params["month"])
    daily = _daily_energy(scope).sort_index()
    delta = daily.diff().dropna()
    ordered = delta.reset_index()
    ordered.columns = ["day", "delta_kwh"]
    ordered = ordered.sort_values(["delta_kwh", "day"], ascending=[False, True])
    return ordered.head(1).reset_index(drop=True)


def _h_energy_distinct_days(params, df):
    scope = df[df["datetime"].dt.year == params["year"]]
    return _frame(int(scope["datetime"].dt.normalize().nunique()), "unique_days")


def _h_energy_weekday_avg(params, df):
    scope = df[df["datetime"].dt.year == params["year"]]
    weekday = scope["datetime"].dt.dayofweek + 1  # ISO: 1 = Monday
    grouped = scope.groupby(weekday)[params["column"]].mean().reset_index()
    grouped.columns = ["weekday", "avg_value"]
    return grouped.sort_values("weekday").reset_index(drop=True)


def _finance_ticker_slice(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return df[df["ticker"] == ticker]


def _finance_range_slice(
    df: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"]).dt.date
    return df[
        (dates >= pd.Timestamp(start).date()) & (dates <= pd.Timestamp(end).date())
    ]


def _h_finance_range_agg(params, df):
    scope = _finance_range_slice(
        _finance_ticker_slice(df, params["ticker"]), params["start"], params["end"]
    )
    return _frame(_agg(scope[params["field"]], params["agg"]))


def _h_finance_volume_agg(params, df):
    scope = _finance_range_slice(
        _finance_ticker_slice(df, params["ticker"]), params["start"], params["end"]
    )
    return _frame(_agg(scope["volume"], params["agg"]))


def _h_finance_count_days(params, df):
    scope = _finance_range_slice(
        _finance_ticker_slice(df, params["ticker"]), params["start"], params["end"]
    )
    return _frame(int(len(scope)), "trading_days")


def _h_finance_monthly_close(params, df):
    scope = _finance_ticker_slice(df, params["ticker"])
    dates = pd.to_datetime(scope["date"])
    scope = scope[dates.dt.year == params["year"]]
    months = pd.to_datetime(scope["date"]).dt.month
    grouped = scope.groupby(months)["close"].mean().reset_index()
    grouped.columns = ["month", "avg_close"]
    return grouped.sort_values("month").reset_index(drop=True)


def _h_finance_top_volume_days(params, df):
    scope = _finance_ticker_slice(df, params["ticker"])
    ordered = scope.sort_values(["volume", "date"], ascending=[False, True])
    return ordered[["date", "volume"]].head(params["n"]).reset_index(drop=True)


def _h_finance_argmax_close(params, df):
    scope = _finance_ticker_slice(df, params["ticker"])
    ascending = params["direction"] == "MIN"
    ordered = scope.sort_values(["close", "date"], ascending=[ascending, True])
    top = ordered.iloc[0]
    return pd.DataFrame({"date": [top["date"]], "close": [top["close"]]})


def _h_finance_ticker_compare(params, df):
    scope = _finance_range_slice(df, params["start"], params["end"])
    grouped = scope.groupby("ticker")["close"].mean().reset_index()
    grouped.columns = ["ticker", "avg_close"]
    return grouped.sort_values("ticker").reset_index(drop=True)


def _h_finance_having_months_close(params, df):
    scope = _finance_ticker_slice(df, params["ticker"])
    dates = pd.to_datetime(scope["date"])
    scope = scope[dates.dt.year == params["year"]]
    months = pd.to_datetime(scope["date"]).dt.month
    grouped = scope.groupby(months)["close"].mean()
    grouped = grouped[grouped > params["threshold"]].reset_index()
    grouped.columns = ["month", "avg_close"]
    return grouped.sort_values("month").reset_index(drop=True)


def _h_finance_daily_return(params, df):
    scope = _finance_ticker_slice(df, params["ticker"]).sort_values("date")
    returns = 100.0 * scope["close"].diff() / scope["close"].shift(1)
    frame = pd.DataFrame(
        {"date": scope["date"], "daily_return_pct": returns}
    ).dropna()
    ascending = params["direction"] == "ASC"
    frame = frame.sort_values(
        ["daily_return_pct", "date"], ascending=[ascending, True]
    )
    return frame.head(1).reset_index(drop=True)


def _h_finance_spread_avg(params, df):
    scope = _finance_range_slice(
        _finance_ticker_slice(df, params["ticker"]), params["start"], params["end"]
    )
    return _frame((scope["high"] - scope["low"]).mean(), "avg_spread")


def _h_finance_close_above_count(params, df):
    scope = _finance_ticker_slice(df, params["ticker"])
    return _frame(int((scope["close"] > params["threshold"]).sum()), "day_count")


_PANDAS_HANDLERS: dict[str, Callable[[dict[str, Any], pd.DataFrame], pd.DataFrame]] = {
    "energy_daily_agg": _h_energy_daily_agg,
    "energy_daily_energy": _h_energy_daily_energy,
    "energy_daily_sub_sum": _h_energy_daily_sub_sum,
    "energy_daily_range": _h_energy_daily_range,
    "energy_daily_stat": _h_energy_daily_stat,
    "energy_monthly_agg": _h_energy_monthly_agg,
    "energy_monthly_energy": _h_energy_monthly_energy,
    "energy_hour_filter": _h_energy_hour_filter,
    "energy_count_threshold": _h_energy_count_threshold,
    "energy_percent_threshold": _h_energy_percent_threshold,
    "energy_per_day_in_month": _h_energy_per_day_in_month,
    "energy_top_n_days": _h_energy_top_n_days,
    "energy_top_n_hours": _h_energy_top_n_hours,
    "energy_having_months": _h_energy_having_months,
    "energy_year_compare": _h_energy_year_compare,
    "energy_month_across_years": _h_energy_month_across_years,
    "energy_multi_col_avg": _h_energy_multi_col_avg,
    "energy_argmax_time": _h_energy_argmax_time,
    "energy_window_delta_day": _h_energy_window_delta_day,
    "energy_distinct_days": _h_energy_distinct_days,
    "energy_weekday_avg": _h_energy_weekday_avg,
    "finance_range_agg": _h_finance_range_agg,
    "finance_volume_agg": _h_finance_volume_agg,
    "finance_count_days": _h_finance_count_days,
    "finance_monthly_close": _h_finance_monthly_close,
    "finance_top_volume_days": _h_finance_top_volume_days,
    "finance_argmax_close": _h_finance_argmax_close,
    "finance_ticker_compare": _h_finance_ticker_compare,
    "finance_having_months_close": _h_finance_having_months_close,
    "finance_daily_return": _h_finance_daily_return,
    "finance_spread_avg": _h_finance_spread_avg,
    "finance_close_above_count": _h_finance_close_above_count,
}

SUPPORTED_TEMPLATES = tuple(_PANDAS_HANDLERS)
