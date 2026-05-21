"""Small DuckDB utility wrapper for structured analytics."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


class DuckDBTool:
    """Lightweight DuckDB helper for local analytics workflows."""

    def __init__(self, db_path: str):
        if not db_path:
            raise ValueError("db_path must not be empty")

        self.db_path = db_path
        self._connection: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Open a DuckDB connection if one is not already active."""
        if self._connection is not None:
            return self._connection

        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

        try:
            self._connection = duckdb.connect(self.db_path)
            self._connection.execute("PRAGMA threads=1")
        except duckdb.Error as exc:
            raise RuntimeError(
                f"Failed to connect to DuckDB database: {self.db_path}"
            ) from exc

        return self._connection

    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return the result as a pandas DataFrame."""
        if not sql or not sql.strip():
            raise ValueError("sql must not be empty")

        connection = self._ensure_connection()
        try:
            return connection.execute(sql).fetchdf()
        except duckdb.Error as exc:
            raise ValueError(f"Invalid SQL query: {exc}") from exc

    def register_csv_as_table(self, csv_path: str, table_name: str) -> None:
        """Register a CSV file as a DuckDB table."""
        csv_file = Path(csv_path).expanduser()
        if not csv_file.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        table_identifier = self._quote_identifier(table_name)
        connection = self._ensure_connection()

        try:
            connection.execute(
                f"CREATE OR REPLACE TABLE {table_identifier} AS "
                "SELECT * FROM read_csv_auto(?)",
                [str(csv_file)],
            )
        except duckdb.Error as exc:
            raise RuntimeError(
                f"Failed to register CSV as table '{table_name}': {exc}"
            ) from exc

    def get_schema(self, table_name: str) -> str:
        """Return a compact text schema for a table."""
        table_identifier = self._quote_identifier(table_name)
        connection = self._ensure_connection()

        try:
            rows = connection.execute(f"DESCRIBE {table_identifier}").fetchall()
        except duckdb.Error as exc:
            raise ValueError(f"Failed to get schema for table '{table_name}': {exc}") from exc

        columns = "\n".join(
            f"{column_name}: {column_type}" for column_name, column_type, *_ in rows
        )
        return f"Table: {table_name}\nColumns:\n{columns}"

    def get_sample_rows(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Return a small sample from a table."""
        if limit < 1:
            raise ValueError("limit must be greater than 0")

        table_identifier = self._quote_identifier(table_name)
        connection = self._ensure_connection()

        try:
            return connection.execute(
                f"SELECT * FROM {table_identifier} LIMIT {int(limit)}"
            ).fetchdf()
        except duckdb.Error as exc:
            raise ValueError(
                f"Failed to get sample rows for table '{table_name}': {exc}"
            ) from exc

    def close(self) -> None:
        """Close the active DuckDB connection."""
        if self._connection is None:
            return

        try:
            self._connection.close()
        finally:
            self._connection = None

    def _ensure_connection(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            return self.connect()

        return self._connection

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        if not identifier or not identifier.strip():
            raise ValueError("table_name must not be empty")

        return '"' + identifier.replace('"', '""') + '"'
