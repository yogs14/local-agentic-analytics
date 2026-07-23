from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.tools.duckdb_tool import DuckDBTool


def test_register_csv_get_schema_and_sample_rows(tmp_path):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "product,quantity,price\n"
        "book,2,10.5\n"
        "pen,5,1.25\n"
        "bag,1,25.0\n",
        encoding="utf-8",
    )
    tool = DuckDBTool(str(tmp_path / "analytics.duckdb"))

    try:
        tool.connect()
        tool.register_csv_as_table(str(csv_path), "sales")

        schema = tool.get_schema("sales")
        sample = tool.get_sample_rows("sales", limit=2)

        assert "Table: sales" in schema
        assert "product: VARCHAR" in schema
        assert "quantity:" in schema
        assert isinstance(sample, pd.DataFrame)
        assert list(sample["product"]) == ["book", "pen"]
    finally:
        tool.close()


def test_get_schema_quotes_columns_with_spaces(tmp_path):
    csv_path = tmp_path / "renewable.csv"
    csv_path.write_text(
        "Year,Hydroelectric Power,Solar Energy\n1973,1.0,0.0\n",
        encoding="utf-8",
    )
    tool = DuckDBTool(str(tmp_path / "analytics.duckdb"))

    try:
        tool.connect()
        tool.register_csv_as_table(str(csv_path), "renewable")
        schema = tool.get_schema("renewable")

        # Names needing quotes are shown quoted; clean names are left bare.
        assert '"Hydroelectric Power":' in schema
        assert '"Solar Energy":' in schema
        assert "\nYear:" in schema
    finally:
        tool.close()


def test_execute_query_returns_dataframe(tmp_path):
    tool = DuckDBTool(str(tmp_path / "analytics.duckdb"))

    try:
        result = tool.execute_query("SELECT 1 AS value")

        assert isinstance(result, pd.DataFrame)
        assert result.to_dict(orient="records") == [{"value": 1}]
    finally:
        tool.close()


def test_execute_query_raises_for_invalid_sql(tmp_path):
    tool = DuckDBTool(str(tmp_path / "analytics.duckdb"))

    try:
        with pytest.raises(ValueError, match="Invalid SQL query"):
            tool.execute_query("SELECT * FROM missing_table")
    finally:
        tool.close()


def test_register_csv_raises_for_missing_file(tmp_path):
    tool = DuckDBTool(str(tmp_path / "analytics.duckdb"))

    try:
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            tool.register_csv_as_table(str(tmp_path / "missing.csv"), "missing")
    finally:
        tool.close()
