from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.profile_generator import (
    generate_profile_from_duckdb,
    get_table_summary,
)
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool


def _make_tool(create_sql: str) -> DuckDBTool:
    tool = DuckDBTool(":memory:")
    tool.connect()
    tool.execute_query(create_sql)
    return tool


def test_generate_profile_detects_timestamp_as_datetime_column():
    tool = _make_tool(
        "CREATE TABLE t (id INTEGER, created_at TIMESTAMP, revenue DOUBLE)"
    )
    try:
        profile = generate_profile_from_duckdb(tool, "t")
        assert profile.datetime_column == "created_at"
        assert "date_filter" in profile.sql_rules
        assert "CAST(created_at AS DATE)" in profile.sql_rules["date_filter"]
    finally:
        tool.close()


def test_generate_profile_detects_date_type_as_datetime_column():
    tool = _make_tool("CREATE TABLE t (tanggal DATE, jumlah INTEGER)")
    try:
        profile = generate_profile_from_duckdb(tool, "t")
        assert profile.datetime_column == "tanggal"
        assert "CAST" not in profile.sql_rules["date_filter"]
        assert "tanggal = DATE" in profile.sql_rules["date_filter"]
    finally:
        tool.close()


def test_generate_profile_falls_back_to_varchar_name_keyword():
    tool = _make_tool(
        "CREATE TABLE t (tgl VARCHAR, produk VARCHAR, nilai DOUBLE)"
    )
    try:
        profile = generate_profile_from_duckdb(tool, "t")
        assert profile.datetime_column == "tgl"
    finally:
        tool.close()


def test_generate_profile_no_datetime_when_no_match():
    tool = _make_tool(
        "CREATE TABLE t (produk VARCHAR, jumlah INTEGER, harga DOUBLE)"
    )
    try:
        profile = generate_profile_from_duckdb(tool, "t")
        assert profile.datetime_column is None
        assert "date_filter" not in profile.sql_rules
    finally:
        tool.close()


def test_generate_profile_prefers_native_datetime_over_varchar_keyword():
    tool = _make_tool("CREATE TABLE t (tanggal VARCHAR, waktu TIMESTAMP)")
    try:
        profile = generate_profile_from_duckdb(tool, "t")
        assert profile.datetime_column == "waktu"
    finally:
        tool.close()


def test_generate_profile_numeric_types_map_correctly():
    tool = _make_tool(
        "CREATE TABLE t (a DOUBLE, b FLOAT, c BIGINT, d INTEGER, e SMALLINT)"
    )
    try:
        profile = generate_profile_from_duckdb(tool, "t")
        assert profile.columns["a"].semantic_type == "numeric"
        assert profile.columns["b"].semantic_type == "numeric"
        assert profile.columns["c"].semantic_type == "integer"
        assert profile.columns["d"].semantic_type == "integer"
        assert profile.columns["e"].semantic_type == "integer"
    finally:
        tool.close()


def test_generate_profile_column_names_and_count():
    tool = _make_tool(
        "CREATE TABLE t (w VARCHAR, x INTEGER, y DOUBLE, z DATE)"
    )
    try:
        profile = generate_profile_from_duckdb(tool, "t")
        assert len(profile.columns) == 4
        for name in ("w", "x", "y", "z"):
            assert name in profile.columns
    finally:
        tool.close()


def test_generate_profile_raises_for_nonexistent_table():
    tool = DuckDBTool(":memory:")
    tool.connect()
    try:
        with pytest.raises(ValueError):
            generate_profile_from_duckdb(tool, "does_not_exist")
    finally:
        tool.close()


def test_get_table_summary_returns_correct_row_count():
    tool = _make_tool("CREATE TABLE t (n INTEGER)")
    try:
        for i in range(10):
            tool.execute_query(f"INSERT INTO t VALUES ({i})")
        summary = get_table_summary(tool, "t")
        assert summary["row_count"] == 10
    finally:
        tool.close()


def test_get_table_summary_lists_numeric_columns():
    tool = _make_tool("CREATE TABLE t (a INTEGER, b DOUBLE, c VARCHAR)")
    try:
        summary = get_table_summary(tool, "t")
        assert summary["numeric_column_names"] == ["a", "b"]
        assert summary["categorical_column_names"] == ["c"]
    finally:
        tool.close()
