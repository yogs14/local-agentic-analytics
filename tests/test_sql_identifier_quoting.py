from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.sql_cleaning import (
    identifier_needs_quoting,
    quote_known_identifiers,
)
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool


# The columns that reproduced the front-end parser error on the
# "U.S. Renewable Energy Consumption" dataset.
RENEWABLE_COLUMNS = [
    "Year",
    "Month",
    "Sector",
    "Hydroelectric Power",
    "Geothermal Energy",
    "Solar Energy",
    "Wind Energy",
    "Fuel Ethanol, Excluding Denaturant",
    "Conventional Hydroelectric Power",
]


# -- identifier_needs_quoting ----------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["Global_active_power", "datetime", "Year", "Sector", "_private", "col1"],
)
def test_clean_identifiers_do_not_need_quoting(name):
    assert identifier_needs_quoting(name) is False


@pytest.mark.parametrize(
    "name",
    [
        "Hydroelectric Power",
        "Fuel Ethanol, Excluding Denaturant",
        "1st_quarter",
        "price (usd)",
    ],
)
def test_special_identifiers_need_quoting(name):
    assert identifier_needs_quoting(name) is True


def test_pure_digit_name_is_not_quoted():
    # Quoting a digit run would collide with numeric literals.
    assert identifier_needs_quoting("2020") is False


def test_empty_name_is_not_quoted():
    assert identifier_needs_quoting("") is False


# -- quote_known_identifiers -----------------------------------------------

def test_quotes_single_spaced_column():
    sql = "SELECT AVG(Hydroelectric Power) AS avg_h FROM t"
    out = quote_known_identifiers(sql, ["Hydroelectric Power"])
    assert out == 'SELECT AVG("Hydroelectric Power") AS avg_h FROM t'


def test_quotes_the_reported_failing_query():
    sql = (
        "SELECT AVG(Hydroelectric Power) AS avg_hydroelectric_power, "
        "AVG(Geothermal Energy) AS avg_geothermal_energy, "
        "AVG(Solar Energy) AS avg_solar_energy FROM renewable"
    )
    out = quote_known_identifiers(sql, RENEWABLE_COLUMNS)
    assert '"Hydroelectric Power"' in out
    assert '"Geothermal Energy"' in out
    assert '"Solar Energy"' in out
    # Aliases must stay untouched.
    assert "AS avg_hydroelectric_power" in out


def test_leaves_clean_columns_untouched():
    sql = "SELECT AVG(Global_active_power) AS p FROM electric_power"
    assert quote_known_identifiers(sql, ["Global_active_power"]) == sql


def test_does_not_double_quote_already_quoted():
    sql = 'SELECT AVG("Hydroelectric Power") FROM t'
    assert quote_known_identifiers(sql, ["Hydroelectric Power"]) == sql


def test_does_not_touch_string_literals():
    sql = "SELECT COUNT(*) FROM t WHERE Sector = 'Solar Energy'"
    # "Solar Energy" appears only inside a string literal, not as a column ref.
    assert quote_known_identifiers(sql, ["Solar Energy"]) == sql


def test_handles_column_name_with_comma():
    sql = "SELECT AVG(Fuel Ethanol, Excluding Denaturant) AS x FROM t"
    out = quote_known_identifiers(sql, ["Fuel Ethanol, Excluding Denaturant"])
    assert out == 'SELECT AVG("Fuel Ethanol, Excluding Denaturant") AS x FROM t'


def test_longer_name_not_corrupted_by_shorter_substring():
    sql = (
        "SELECT AVG(Conventional Hydroelectric Power) AS a, "
        "AVG(Hydroelectric Power) AS b FROM t"
    )
    out = quote_known_identifiers(
        sql, ["Hydroelectric Power", "Conventional Hydroelectric Power"]
    )
    assert '"Conventional Hydroelectric Power"' in out
    # No nested/broken quoting such as "Conventional "Hydroelectric Power"".
    assert '"Conventional "' not in out
    assert out.count('"') == 4  # two well-formed quoted identifiers


def test_empty_sql_returns_input():
    assert quote_known_identifiers("", ["Solar Energy"]) == ""


def test_no_special_columns_returns_input_unchanged():
    sql = "SELECT * FROM t"
    assert quote_known_identifiers(sql, ["Year", "Month"]) == sql


# -- End-to-end against DuckDB ---------------------------------------------

def test_quoted_sql_executes_against_duckdb(tmp_path):
    """The bare model SQL fails; the quoted rewrite runs successfully."""
    csv_path = tmp_path / "renewable.csv"
    csv_path.write_text(
        "Year,Hydroelectric Power,Solar Energy\n"
        "1973,1.0,0.0\n"
        "1974,2.0,0.5\n",
        encoding="utf-8",
    )
    tool = DuckDBTool(str(tmp_path / "a.duckdb"))
    try:
        tool.connect()
        tool.register_csv_as_table(str(csv_path), "renewable")

        bare_sql = (
            "SELECT AVG(Hydroelectric Power) AS avg_h, "
            "AVG(Solar Energy) AS avg_s FROM renewable"
        )
        with pytest.raises(ValueError, match="Invalid SQL query"):
            tool.execute_query(bare_sql)

        fixed_sql = quote_known_identifiers(
            bare_sql, ["Hydroelectric Power", "Solar Energy"]
        )
        result = tool.execute_query(fixed_sql)
        assert result["avg_h"].iloc[0] == pytest.approx(1.5)
        assert result["avg_s"].iloc[0] == pytest.approx(0.25)
    finally:
        tool.close()
