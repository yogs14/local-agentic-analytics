"""Ingest a small synthetic stock prices dataset into DuckDB."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"
DEFAULT_TABLE_NAME = "stock_prices"

SYNTHETIC_STOCK_ROWS: tuple[tuple[str, float | None, int | None, float | None], ...] = (
    ("2024-01-02", 150.25, 1200000, None),
    ("2024-01-03", 151.10, 1350000, 0.566),
    ("2024-01-04", 149.80, 1285000, -0.860),
    ("2024-01-05", 152.35, 1420000, 1.702),
    ("2024-01-08", 153.40, 1395000, 0.689),
    ("2024-01-09", 155.10, None, 1.108),
    ("2024-01-10", 154.20, 1310000, -0.580),
    ("2024-01-11", None, 1250000, None),
    ("2024-01-12", 156.00, 1480000, 1.167),
    ("2024-01-16", 155.55, 1365000, -0.288),
    ("2024-01-17", 157.20, 1510000, 1.061),
    ("2024-01-18", 158.05, 1460000, 0.541),
)


def quote_identifier(identifier: str) -> str:
    if not identifier or not identifier.strip():
        raise ValueError("Identifier must not be empty")

    return '"' + identifier.replace('"', '""') + '"'


def ingest_finance_dataset(
    db_path: Path,
    table_name: str = DEFAULT_TABLE_NAME,
    rows: tuple[tuple[str, float | None, int | None, float | None], ...] = (
        SYNTHETIC_STOCK_ROWS
    ),
) -> dict[str, Any]:
    """Create or replace the finance table with deterministic synthetic rows."""
    if not rows:
        raise ValueError("rows must not be empty")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    table_identifier = quote_identifier(table_name)

    with duckdb.connect(str(db_path)) as connection:
        connection.execute("PRAGMA threads=1")
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {table_identifier} (
                date DATE,
                close_price DOUBLE,
                volume BIGINT,
                return_pct DOUBLE
            );
            """
        )
        connection.executemany(
            (
                f"INSERT INTO {table_identifier} "
                "(date, close_price, volume, return_pct) VALUES (?, ?, ?, ?)"
            ),
            rows,
        )
        row_count = connection.execute(
            f"SELECT COUNT(*) FROM {table_identifier}"
        ).fetchone()[0]
        schema_rows = connection.execute(f"DESCRIBE {table_identifier}").fetchall()

    return {
        "db_path": str(db_path),
        "table_name": table_name,
        "row_count": int(row_count),
        "schema": [
            {
                "column_name": row[0],
                "column_type": row[1],
                "nullable": row[2],
                "key": row[3],
                "default": row[4],
                "extra": row[5],
            }
            for row in schema_rows
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest synthetic finance stock prices into DuckDB."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="DuckDB database path.",
    )
    parser.add_argument(
        "--table-name",
        default=DEFAULT_TABLE_NAME,
        help="Destination DuckDB table name.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        metadata = ingest_finance_dataset(args.db_path, args.table_name)
    except (duckdb.Error, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"DuckDB database: {metadata['db_path']}")
    print(f"Table: {metadata['table_name']}")
    print(f"Rows: {metadata['row_count']}")
    print("Schema:")
    for column in metadata["schema"]:
        print(
            f"- {column['column_name']}: {column['column_type']} "
            f"(nullable={column['nullable']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
