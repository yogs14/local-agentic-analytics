from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.sql_cleaning import clean_sql_response


def test_clean_sql_response_removes_sql_language_line():
    assert clean_sql_response("sql\nSELECT 1;\n") == "SELECT 1;"


def test_clean_sql_response_removes_trailing_markdown_fence():
    assert clean_sql_response("SELECT 1;\n```") == "SELECT 1;"


def test_clean_sql_response_extracts_select_after_prose():
    assert clean_sql_response("Here is the SQL:\nSELECT 1;\n```") == "SELECT 1;"


def test_clean_sql_response_removes_sql_label_prefix():
    response = (
        "SQL:\nSELECT AVG(Global_active_power) AS avg_global_active_power_kw "
        "FROM electric_power;"
    )

    assert clean_sql_response(response) == (
        "SELECT AVG(Global_active_power) AS avg_global_active_power_kw "
        "FROM electric_power;"
    )


def test_clean_sql_response_removes_duckdb_language_line_without_semicolon():
    assert clean_sql_response("duckdb\nSELECT 1\n") == "SELECT 1"


def test_clean_sql_response_keeps_only_first_statement():
    assert clean_sql_response("SELECT 1; SELECT 2;") == "SELECT 1;"
