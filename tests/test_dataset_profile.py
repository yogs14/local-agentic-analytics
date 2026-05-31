from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core import dataset_profile
from local_agentic_analytics.core.dataset_profile import (
    load_dataset_profile,
    profile_to_compact_sql_context,
    profile_to_prompt_context,
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
