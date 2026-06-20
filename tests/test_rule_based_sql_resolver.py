from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.rule_based_sql_resolver import (
    RuleBasedSQLResolver,
)
from local_agentic_analytics.core.dataset_profile import load_dataset_profile


def test_resolver_generates_average_active_power_sql_for_indonesian_date():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile()

    sql = resolver.resolve(
        "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?",
        profile,
    )

    assert sql == (
        "SELECT AVG(Global_active_power) AS avg_global_active_power_kw\n"
        "FROM electric_power\n"
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-16';"
    )


def test_resolver_generates_total_energy_sql_for_iso_date():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile()

    sql = resolver.resolve(
        "Berapa total energi kWh pada tanggal 2006-12-17?",
        profile,
    )

    assert sql == (
        "SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh\n"
        "FROM electric_power\n"
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-17';"
    )


def test_resolver_generates_missing_value_sql_when_column_is_named():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile()

    sql = resolver.resolve(
        "Berapa jumlah missing value pada kolom Global_active_power?",
        profile,
    )

    assert sql == (
        "SELECT COUNT(*) FILTER (WHERE Global_active_power IS NULL) "
        "AS missing_global_active_power_count\n"
        "FROM electric_power;"
    )


def test_resolver_returns_none_when_rule_or_date_does_not_match():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile()

    assert resolver.resolve("Berapa tegangan rata-rata?", profile) is None
    assert resolver.resolve("Berapa total energi kWh?", profile) is None
    assert (
        resolver.resolve(
            "Berapa jumlah nilai hilang pada kolom kolom_tidak_ada?",
            profile,
        )
        is None
    )


def test_resolver_generates_finance_average_close_price_sql():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve("Berapa rata-rata harga penutupan?", profile)

    assert sql == (
        "SELECT AVG(close_price) AS avg_close_price_usd\n"
        "FROM stock_prices;"
    )


def test_resolver_generates_finance_average_close_price_sql_with_date():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve(
        "Berapa rata-rata harga penutupan pada tanggal 2024-01-05?",
        profile,
    )

    assert sql == (
        "SELECT AVG(close_price) AS avg_close_price_usd\n"
        "FROM stock_prices\n"
        "WHERE CAST(date AS DATE) = DATE '2024-01-05';"
    )


def test_resolver_generates_finance_max_close_price_sql():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve("Berapa harga penutupan maksimum?", profile)

    assert sql == (
        "SELECT MAX(close_price) AS max_close_price_usd\n"
        "FROM stock_prices;"
    )


def test_resolver_generates_finance_average_volume_sql():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve("Berapa volume rata-rata?", profile)

    assert sql == (
        "SELECT AVG(volume) AS avg_volume_shares\n"
        "FROM stock_prices;"
    )


def test_resolver_generates_finance_average_return_sql():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve("Berapa return rata-rata?", profile)

    assert sql == (
        "SELECT AVG(return_pct) AS avg_return_pct\n"
        "FROM stock_prices;"
    )


def test_resolver_generates_finance_missing_value_sql_when_column_is_named():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve(
        "Berapa jumlah missing value pada kolom close_price?",
        profile,
    )

    assert sql == (
        "SELECT COUNT(*) FILTER (WHERE close_price IS NULL) "
        "AS missing_close_price_count\n"
        "FROM stock_prices;"
    )


def test_resolver_generates_finance_all_missing_value_sql_without_named_column():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve("Berapa jumlah missing value?", profile)

    assert sql == (
        "SELECT COUNT(*) FILTER (WHERE close_price IS NULL) "
        "AS missing_close_price_count, "
        "COUNT(*) FILTER (WHERE volume IS NULL) AS missing_volume_count, "
        "COUNT(*) FILTER (WHERE return_pct IS NULL) AS missing_return_pct_count\n"
        "FROM stock_prices;"
    )
