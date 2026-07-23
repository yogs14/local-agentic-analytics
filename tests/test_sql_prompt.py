from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.prompts.sql_prompt import build_sql_prompt


def test_build_sql_prompt_includes_dataset_profile_and_energy_rules():
    prompt = build_sql_prompt(
        question="Berapa total energi kWh pada tanggal 16 Desember 2006?",
        schema="Table: electric_power\nGlobal_active_power: DOUBLE\ndatetime: TIMESTAMP",
    )

    assert "Dataset profile and semantic rules" in prompt
    assert "table_name: electric_power" in prompt
    assert "datetime_column: datetime" in prompt
    assert "available_columns:" in prompt
    assert "SUM(Global_active_power) / 60.0" in prompt
    assert "COUNT(*) FILTER" in prompt
    assert "CAST(datetime AS DATE)" in prompt
    assert "Dataset konsumsi listrik rumah tangga per menit" not in prompt


def test_build_sql_prompt_includes_required_few_shot_examples():
    prompt = build_sql_prompt(
        question="Berapa rata-rata daya aktif?",
        schema="Table: electric_power\nGlobal_active_power: DOUBLE\ndatetime: TIMESTAMP",
    )

    assert "Example 1:" in prompt
    assert "AVG(Global_active_power) AS avg_global_active_power_kw" in prompt
    assert "SUM(Global_active_power) / 60.0 AS total_energy_kwh" in prompt
    assert (
        "COUNT(*) FILTER (WHERE Global_active_power IS NULL) "
        "AS missing_global_active_power_count"
    ) in prompt


def test_build_sql_prompt_warns_against_selecting_output_aliases():
    prompt = build_sql_prompt(
        question="Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?",
        schema="Table: electric_power\nGlobal_active_power: DOUBLE\ndatetime: TIMESTAMP",
    )

    assert "Alias names are output names only" in prompt
    assert "Never SELECT an alias as a source column" in prompt
    assert "AVG(Global_active_power) AS avg_global_active_power_kw" in prompt
    assert "SELECT avg_global_active_power_kw FROM electric_power" in prompt
    assert "Wrong:" in prompt
    assert "Correct:" in prompt


def test_build_sql_prompt_accepts_dataset_profile_context_override():
    prompt = build_sql_prompt(
        question="Berapa total penjualan?",
        schema="sales(amount DOUBLE)",
        dataset_profile_context=(
            "Dataset name: sales\n"
            "Canonical table name: sales\n"
            "SQL rules:\n"
            "- total_sales: SUM(amount) AS total_amount"
        ),
    )

    assert "SUM(amount) AS total_amount" in prompt
    assert "Example 1:" not in prompt


def test_build_sql_prompt_adds_quoting_rule_when_schema_has_quoted_column():
    prompt = build_sql_prompt(
        question="Tampilkan rata-rata setiap kolom numerik.",
        schema='Table: renewable\n"Hydroelectric Power": DOUBLE\nYear: BIGINT',
        dataset_profile_context="table_name: renewable",
    )

    assert 'AVG("Hydroelectric Power")' in prompt


def test_build_sql_prompt_omits_quoting_rule_for_clean_schema():
    prompt = build_sql_prompt(
        question="Berapa rata-rata daya aktif?",
        schema="Table: electric_power\nGlobal_active_power: DOUBLE\ndatetime: TIMESTAMP",
    )

    assert "contain spaces or special characters" not in prompt


def test_build_sql_prompt_rejects_empty_inputs():
    with pytest.raises(ValueError, match="question must not be empty"):
        build_sql_prompt(question="", schema="sales(amount DOUBLE)")

    with pytest.raises(ValueError, match="schema must not be empty"):
        build_sql_prompt(question="Berapa total?", schema="")
