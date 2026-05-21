"""Ingest household electric power consumption data into DuckDB."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "energy"
DEFAULT_DB_PATH = PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"
DEFAULT_TABLE_NAME = "electric_power"
PREFERRED_FILENAMES = (
    "household_power_consumption.txt",
    "household_power_consumption.csv",
)
NUMERIC_COLUMNS = {
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
}


def quote_identifier(identifier: str) -> str:
    if not identifier or not identifier.strip():
        raise ValueError("Identifier must not be empty")

    return '"' + identifier.replace('"', '""') + '"'


def find_dataset_file(input_dir: Path) -> Path:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for filename in PREFERRED_FILENAMES:
        candidate = input_dir / filename
        if candidate.is_file():
            return candidate

    candidates = sorted(
        file for pattern in ("*.txt", "*.csv") for file in input_dir.glob(pattern)
    )
    if not candidates:
        raise FileNotFoundError(f"No .txt or .csv dataset file found in: {input_dir}")

    if len(candidates) > 1:
        names = ", ".join(file.name for file in candidates)
        raise ValueError(
            "Multiple dataset files found. Use --input-dir with only one file "
            f"or keep the expected filename. Found: {names}"
        )

    return candidates[0]


def read_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8") as file:
        header = file.readline().strip()

    if not header:
        raise ValueError(f"Dataset file is empty: {csv_path}")

    return header.split(";")


def build_ingest_sql(csv_path: Path, table_name: str, columns: list[str]) -> str:
    table_identifier = quote_identifier(table_name)
    csv_literal = str(csv_path).replace("'", "''")
    has_datetime_columns = "Date" in columns and "Time" in columns

    reader_sql = (
        f"read_csv_auto('{csv_literal}', delim=';', header=true, nullstr='?', "
        "all_varchar=true, sample_size=-1)"
    )

    if not has_datetime_columns:
        select_items = [
            build_column_select_expression(column)
            for column in columns
        ]
        select_sql = ", ".join(select_items)
        return f"CREATE OR REPLACE TABLE {table_identifier} AS SELECT {select_sql} FROM {reader_sql}"

    select_items = [
        (
            "try_strptime(CAST(\"Date\" AS VARCHAR) || ' ' || CAST(\"Time\" AS VARCHAR), "
            "'%d/%m/%Y %H:%M:%S') AS datetime"
        )
    ]
    select_items.extend(
        build_column_select_expression(column)
        for column in columns
        if column not in {"Date", "Time"}
    )
    select_sql = ", ".join(select_items)
    return (
        f"CREATE OR REPLACE TABLE {table_identifier} AS "
        f"SELECT {select_sql} FROM {reader_sql}"
    )


def build_column_select_expression(column: str) -> str:
    column_identifier = quote_identifier(column)
    if column in NUMERIC_COLUMNS:
        return f"try_cast({column_identifier} AS DOUBLE) AS {column_identifier}"

    return column_identifier


def ingest_energy_dataset(csv_path: Path, db_path: Path, table_name: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    columns = read_header(csv_path)
    has_datetime_columns = "Date" in columns and "Time" in columns

    if not has_datetime_columns:
        print("Warning: Date and Time columns were not found. Loading raw columns only.")

    with duckdb.connect(str(db_path)) as connection:
        connection.execute("PRAGMA threads=1")
        connection.execute(build_ingest_sql(csv_path, table_name, columns))

        table_identifier = quote_identifier(table_name)
        row_count = connection.execute(
            f"SELECT COUNT(*) FROM {table_identifier}"
        ).fetchone()[0]
        final_columns = [
            row[1]
            for row in connection.execute(
                f"PRAGMA table_info({table_identifier})"
            ).fetchall()
        ]
        schema_rows = connection.execute(f"DESCRIBE {table_identifier}").fetchall()

    print(f"Dataset file: {csv_path}")
    print(f"DuckDB database: {db_path}")
    print(f"Table: {table_name}")
    print(f"Rows: {row_count}")
    print("Columns:")
    for column in final_columns:
        print(f"- {column}")
    print("Schema:")
    for column_name, column_type, nullable, key, default, extra in schema_rows:
        print(
            f"- {column_name}: {column_type} "
            f"(nullable={nullable}, key={key}, default={default}, extra={extra})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest household electric power consumption data into DuckDB."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing household_power_consumption.txt or .csv.",
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
        dataset_file = find_dataset_file(args.input_dir)
        ingest_energy_dataset(dataset_file, args.db_path, args.table_name)
    except (duckdb.Error, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
