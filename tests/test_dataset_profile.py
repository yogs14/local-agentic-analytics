from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core import dataset_profile
from local_agentic_analytics.core.dataset_profile import (
    ColumnProfile,
    DatasetProfile,
    load_dataset_profile,
    profile_to_compact_sql_context,
    profile_to_prompt_context,
)


def _generic_profile() -> DatasetProfile:
    return DatasetProfile(
        name="rendo",
        domain="custom",
        table_name="rendo_electricity_2013_2",
        datetime_column="",  # type: ignore[arg-type]
        columns={
            "net_manager": ColumnProfile(
                name="net_manager", type="integer", semantic_type="integer"
            ),
            "street": ColumnProfile(
                name="street", type="string", semantic_type="categorical"
            ),
            "num_connections": ColumnProfile(
                name="num_connections", type="numeric", semantic_type="numeric"
            ),
        },
    )


def test_load_dataset_profile_reads_energy_profile():
    profile = load_dataset_profile("energy")

    assert profile.name == "household_power_consumption"
    assert profile.domain == "energy"
    assert profile.table_name == "electric_power"
    assert profile.datetime_column == "datetime"
    assert profile.sampling_frequency == "per_minute"
    assert profile.columns["Global_active_power"].unit == "kW"
    assert (
        profile.columns["Global_active_power"].rules["total_energy_kwh_sql"]
        == "SUM(Global_active_power) / 60.0"
    )
    assert (
        profile.sql_rules["missing_value_count"]
        == "COUNT(*) FILTER (WHERE <column> IS NULL)"
    )


def test_load_dataset_profile_reads_finance_profile():
    profile = load_dataset_profile("finance")

    assert profile.name == "sp500_analyst_ratings"
    assert profile.domain == "finance"
    assert profile.table_name == "stock_prices"
    assert profile.datetime_column == "date"
    assert profile.sampling_frequency == "daily"
    assert profile.columns["ticker"].semantic_type == "ticker_symbol"
    assert profile.columns["close"].unit == "USD"
    assert profile.columns["open"].unit == "USD"
    assert profile.columns["volume"].unit == "shares"


def test_profile_to_prompt_context_contains_domain_semantics():
    profile = load_dataset_profile()
    context = profile_to_prompt_context(profile)

    assert "Dataset name: household_power_consumption" in context
    assert "Canonical table name: electric_power" in context
    assert "Datetime column: datetime" in context
    assert "Global_active_power: numeric; unit=kW; semantic_type=active_power" in context
    assert "total_energy_kwh_sql: SUM(Global_active_power) / 60.0" in context
    assert "date_filter: CAST(datetime AS DATE) = DATE 'YYYY-MM-DD'" in context


def test_profile_to_compact_sql_context_keeps_only_sql_essentials():
    profile = load_dataset_profile()
    context = profile_to_compact_sql_context(profile)

    assert "table_name: electric_power" in context
    assert "datetime_column: datetime" in context
    assert "available_columns:" in context
    assert "Global_active_power" in context
    assert "Voltage=V" in context
    assert "AVG(Global_active_power) AS avg_global_active_power_kw" in context
    assert "SUM(Global_active_power) / 60.0 AS total_energy_kwh" in context
    assert "COUNT(*) FILTER (WHERE <column> IS NULL)" in context
    assert "CAST(datetime AS DATE) = DATE 'YYYY-MM-DD'" in context
    assert "Dataset konsumsi listrik rumah tangga per menit" not in context
    assert "Daya aktif global rata-rata dalam kilowatt" not in context


def test_finance_compact_sql_context_uses_finance_table_and_rules():
    profile = load_dataset_profile("finance")
    context = profile_to_compact_sql_context(profile)

    assert "table_name: stock_prices" in context
    assert "datetime_column: date" in context
    assert (
        "available_columns: date, ticker, open, high, low, close, volume" in context
    )
    assert "close=USD" in context
    assert "volume=shares" in context
    assert "AVG(close) AS avg_close_usd" in context
    assert "MAX(close) AS max_close_usd" in context
    assert "AVG(volume) AS avg_volume_shares" in context
    assert "CAST(date AS DATE) = DATE 'YYYY-MM-DD'" in context
    assert "electric_power" not in context
    assert "Global_active_power" not in context


def test_generic_compact_context_lists_numeric_and_text_columns():
    context = profile_to_compact_sql_context(_generic_profile())

    assert "numeric_columns: net_manager, num_connections" in context
    assert "text_columns: street" in context
    assert "Always start the query with SELECT" in context
    assert "read FROM rendo_electricity_2013_2" in context
    assert "never aggregate text_columns" in context


def test_curated_domains_do_not_get_generic_type_guidance():
    energy = profile_to_compact_sql_context(load_dataset_profile("energy"))
    finance = profile_to_compact_sql_context(load_dataset_profile("finance"))

    # The gate keeps curated-domain prompts byte-identical (no new sections).
    for context in (energy, finance):
        assert "numeric_columns:" not in context
        assert "text_columns:" not in context
        assert "Always start the query with SELECT" not in context


def test_load_dataset_profile_raises_for_missing_domain(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_profile, "DOMAINS_DIR", tmp_path / "domains")

    with pytest.raises(FileNotFoundError, match="Domain not found"):
        load_dataset_profile("missing")


def test_load_dataset_profile_raises_for_missing_profile_yaml(tmp_path, monkeypatch):
    domains_dir = tmp_path / "domains"
    (domains_dir / "energy").mkdir(parents=True)
    monkeypatch.setattr(dataset_profile, "DOMAINS_DIR", domains_dir)

    with pytest.raises(FileNotFoundError, match="Dataset profile file not found"):
        load_dataset_profile("energy")


def test_load_dataset_profile_rejects_empty_table_name(tmp_path, monkeypatch):
    domains_dir = tmp_path / "domains"
    profile_dir = domains_dir / "energy"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        """
dataset:
  name: test
  domain: energy
  table_name: ""
  datetime_column: datetime
columns: {}
sql_rules: {}
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(dataset_profile, "DOMAINS_DIR", domains_dir)

    with pytest.raises(ValueError, match="table_name"):
        load_dataset_profile("energy")


def test_load_dataset_profile_rejects_empty_datetime_column(tmp_path, monkeypatch):
    domains_dir = tmp_path / "domains"
    profile_dir = domains_dir / "energy"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        """
dataset:
  name: test
  domain: energy
  table_name: electric_power
  datetime_column: ""
columns: {}
sql_rules: {}
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(dataset_profile, "DOMAINS_DIR", domains_dir)

    with pytest.raises(ValueError, match="datetime_column"):
        load_dataset_profile("energy")
