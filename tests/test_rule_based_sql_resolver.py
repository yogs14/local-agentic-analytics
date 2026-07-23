from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.rule_based_sql_resolver import (
    RuleBasedSQLResolver,
)
from local_agentic_analytics.core.dataset_profile import (
    ColumnProfile,
    DatasetProfile,
    load_dataset_profile,
)


def _mixed_profile_without_datetime() -> DatasetProfile:
    """A custom CSV profile with numeric and text columns and no datetime."""
    columns = {
        "net_manager": ColumnProfile(
            name="net_manager", type="integer", semantic_type="integer"
        ),
        "street": ColumnProfile(
            name="street", type="string", semantic_type="categorical"
        ),
        "zipcode_from": ColumnProfile(
            name="zipcode_from", type="string", semantic_type="categorical"
        ),
        "num_connections": ColumnProfile(
            name="num_connections", type="numeric", semantic_type="numeric"
        ),
        "annual_consume": ColumnProfile(
            name="annual_consume", type="integer", semantic_type="integer"
        ),
    }
    return DatasetProfile(
        name="rendo",
        domain="custom",
        table_name="rendo_electricity_2013_2",
        datetime_column="",  # type: ignore[arg-type]
        columns=columns,
    )


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


def test_resolver_generates_finance_average_close_sql():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve("Berapa rata-rata harga penutupan?", profile)

    assert sql == (
        "SELECT AVG(close) AS avg_close_usd\n"
        "FROM stock_prices;"
    )


def test_resolver_generates_finance_average_close_sql_with_date():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve(
        "Berapa rata-rata harga penutupan pada tanggal 2019-06-03?",
        profile,
    )

    assert sql == (
        "SELECT AVG(close) AS avg_close_usd\n"
        "FROM stock_prices\n"
        "WHERE CAST(date AS DATE) = DATE '2019-06-03';"
    )


def test_resolver_generates_finance_average_close_sql_with_ticker():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve("Berapa rata-rata harga penutupan NVDA?", profile)

    assert sql == (
        "SELECT AVG(close) AS avg_close_usd\n"
        "FROM stock_prices\n"
        "WHERE ticker = 'NVDA';"
    )


def test_resolver_generates_finance_max_close_sql():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve("Berapa harga penutupan maksimum?", profile)

    assert sql == (
        "SELECT MAX(close) AS max_close_usd\n"
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


def test_resolver_generates_finance_missing_value_sql_when_column_is_named():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve(
        "Berapa jumlah missing value pada kolom close?",
        profile,
    )

    assert sql == (
        "SELECT COUNT(*) FILTER (WHERE close IS NULL) "
        "AS missing_close_count\n"
        "FROM stock_prices;"
    )


def test_resolver_averages_only_numeric_columns_without_datetime():
    resolver = RuleBasedSQLResolver()
    profile = _mixed_profile_without_datetime()

    sql = resolver.resolve("Tampilkan rata-rata dari setiap kolom numerik.", profile)

    # Text columns (street, zipcode_from) must be excluded to avoid avg(VARCHAR).
    assert sql == (
        'SELECT AVG("net_manager") AS avg_net_manager, '
        'AVG("num_connections") AS avg_num_connections, '
        'AVG("annual_consume") AS avg_annual_consume\n'
        "FROM rendo_electricity_2013_2;"
    )


def test_resolver_numeric_summary_matches_english_phrasing():
    resolver = RuleBasedSQLResolver()
    profile = _mixed_profile_without_datetime()

    sql = resolver.resolve("Show the average of all numeric columns", profile)

    assert sql is not None
    assert sql.startswith('SELECT AVG("net_manager")')


def test_resolver_numeric_summary_quotes_spaced_columns():
    resolver = RuleBasedSQLResolver()
    profile = DatasetProfile(
        name="renewable",
        domain="custom",
        table_name="renewable",
        datetime_column="",  # type: ignore[arg-type]
        columns={
            "Sector": ColumnProfile(
                name="Sector", type="string", semantic_type="categorical"
            ),
            "Solar Energy": ColumnProfile(
                name="Solar Energy", type="numeric", semantic_type="numeric"
            ),
        },
    )

    sql = resolver.resolve("rata-rata setiap kolom numerik", profile)

    assert sql == 'SELECT AVG("Solar Energy") AS avg_solar_energy\nFROM renewable;'


def test_resolver_null_per_column_without_datetime():
    resolver = RuleBasedSQLResolver()
    profile = _mixed_profile_without_datetime()

    sql = resolver.resolve("adakah null value dari tiap kolom?", profile)

    # Every column (numeric and text) gets a null count, with SELECT and FROM.
    assert sql == (
        "SELECT "
        "COUNT(*) FILTER (WHERE net_manager IS NULL) AS missing_net_manager_count, "
        "COUNT(*) FILTER (WHERE street IS NULL) AS missing_street_count, "
        "COUNT(*) FILTER (WHERE zipcode_from IS NULL) AS missing_zipcode_from_count, "
        "COUNT(*) FILTER (WHERE num_connections IS NULL) "
        "AS missing_num_connections_count, "
        "COUNT(*) FILTER (WHERE annual_consume IS NULL) "
        "AS missing_annual_consume_count\n"
        "FROM rendo_electricity_2013_2;"
    )


def test_resolver_null_recognized_as_missing_value_keyword():
    resolver = RuleBasedSQLResolver()
    profile = _mixed_profile_without_datetime()

    # "null" alone (no "missing value"/"nilai hilang") must still be handled.
    assert resolver.resolve("berapa null di semua kolom?", profile) is not None


def test_resolver_numeric_summary_does_not_shadow_energy_rule():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile()

    # "rata-rata daya aktif" must still resolve to the energy-specific rule,
    # not the generic numeric-summary path.
    sql = resolver.resolve(
        "Berapa rata-rata daya aktif pada tanggal 16 Desember 2006?",
        profile,
    )

    assert sql == (
        "SELECT AVG(Global_active_power) AS avg_global_active_power_kw\n"
        "FROM electric_power\n"
        "WHERE CAST(datetime AS DATE) = DATE '2006-12-16';"
    )


def test_resolver_generates_finance_all_missing_value_sql_without_named_column():
    resolver = RuleBasedSQLResolver()
    profile = load_dataset_profile("finance")

    sql = resolver.resolve("Berapa jumlah missing value?", profile)

    assert sql == (
        "SELECT COUNT(*) FILTER (WHERE ticker IS NULL) AS missing_ticker_count, "
        "COUNT(*) FILTER (WHERE open IS NULL) AS missing_open_count, "
        "COUNT(*) FILTER (WHERE high IS NULL) AS missing_high_count, "
        "COUNT(*) FILTER (WHERE low IS NULL) AS missing_low_count, "
        "COUNT(*) FILTER (WHERE close IS NULL) AS missing_close_count, "
        "COUNT(*) FILTER (WHERE volume IS NULL) AS missing_volume_count\n"
        "FROM stock_prices;"
    )
