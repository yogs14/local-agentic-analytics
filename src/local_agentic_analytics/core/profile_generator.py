"""Generate a :class:`DatasetProfile` by inspecting a DuckDB table schema.

This module lets the API build a profile for an arbitrary CSV that was ingested
into DuckDB, without any hand-written ``profile.yaml``. It only reads the table
schema (``DESCRIBE``) and never scans table data, so it stays cheap for large
tables.
"""

from __future__ import annotations

from typing import Any

from local_agentic_analytics.core.dataset_profile import ColumnProfile, DatasetProfile
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool


# Column-name keywords used to detect a datetime column (case-insensitive,
# substring match) when no native TIMESTAMP/DATE column exists.
_DATETIME_NAME_KEYWORDS: tuple[str, ...] = (
    "date", "time", "datetime", "timestamp", "tanggal", "waktu",
    "periode", "tgl", "dt", "created", "updated", "recorded",
)

# DuckDB types treated as continuous numeric.
_NUMERIC_TYPES: frozenset[str] = frozenset({
    "DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC",
    "FLOAT4", "FLOAT8",
})

# DuckDB types treated as integer.
_INTEGER_TYPES: frozenset[str] = frozenset({
    "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT",
    "UBIGINT", "UINTEGER", "USMALLINT", "UTINYINT", "INT", "INT4",
    "INT8", "INT2", "INT1", "SIGNED",
})

# DuckDB types treated as native datetime.
_NATIVE_DATETIME_TYPES: frozenset[str] = frozenset({
    "TIMESTAMP", "DATE", "TIMESTAMP WITH TIME ZONE",
    "TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMP_NS",
})

# DuckDB types treated as categorical strings.
_STRING_TYPES: frozenset[str] = frozenset({
    "VARCHAR", "TEXT", "STRING", "CHAR", "BPCHAR", "UUID",
})


def _normalize_type(duckdb_type: str) -> tuple[str, str]:
    """Return ``(full, base)`` upper-cased forms of a DuckDB type string.

    ``full`` keeps the whole declaration (e.g. ``TIMESTAMP WITH TIME ZONE``)
    while ``base`` strips any ``(...)`` parameters (e.g. ``DECIMAL(18,3)`` ->
    ``DECIMAL``).
    """
    full = str(duckdb_type).upper().strip()
    base = full.split("(", 1)[0].strip()
    return full, base


def _is_native_datetime(duckdb_type: str) -> bool:
    full, base = _normalize_type(duckdb_type)
    return full in _NATIVE_DATETIME_TYPES or base in _NATIVE_DATETIME_TYPES


def _is_varchar(duckdb_type: str) -> bool:
    _, base = _normalize_type(duckdb_type)
    return base in _STRING_TYPES


def _classify(duckdb_type: str) -> tuple[str, str]:
    """Map a DuckDB type to ``(profile_type, semantic_type)``."""
    full, base = _normalize_type(duckdb_type)

    if full in _NATIVE_DATETIME_TYPES or base in _NATIVE_DATETIME_TYPES:
        return "datetime", "datetime"
    if base == "BOOLEAN" or base == "BOOL":
        return "boolean", "boolean"
    if base in _NUMERIC_TYPES:
        return "numeric", "numeric"
    if base in _INTEGER_TYPES:
        return "integer", "integer"
    if base in _STRING_TYPES:
        return "string", "categorical"
    return "string", "other"


def _name_matches_datetime_keyword(column_name: str) -> bool:
    lowered = column_name.lower()
    return any(keyword in lowered for keyword in _DATETIME_NAME_KEYWORDS)


def _describe_columns(
    duckdb_tool: DuckDBTool,
    table_name: str,
) -> list[tuple[str, str]]:
    """Return ``[(column_name, duckdb_type), ...]`` in table definition order.

    Raises ``ValueError`` if the table does not exist or has no columns.
    """
    escaped = table_name.replace("'", "''")
    exists = duckdb_tool.execute_query(
        "SELECT table_name FROM information_schema.tables "
        f"WHERE table_name = '{escaped}'"
    )
    if exists.empty:
        raise ValueError(f"Table not found in database: {table_name}")

    quoted = '"' + table_name.replace('"', '""') + '"'
    described = duckdb_tool.execute_query(f"DESCRIBE {quoted}")
    if described.empty:
        raise ValueError(f"Table has no columns: {table_name}")

    return [
        (str(row.column_name), str(row.column_type))
        for row in described.itertuples(index=False)
    ]


def _detect_datetime_column(columns: list[tuple[str, str]]) -> tuple[str | None, str]:
    """Return ``(datetime_column, its_duckdb_type)`` following the priority rules.

    P1 native TIMESTAMP/DATE columns, P2 VARCHAR columns whose name matches a
    datetime keyword, otherwise ``(None, "")``.
    """
    for name, duckdb_type in columns:
        if _is_native_datetime(duckdb_type):
            return name, duckdb_type

    for name, duckdb_type in columns:
        if _is_varchar(duckdb_type) and _name_matches_datetime_keyword(name):
            return name, duckdb_type

    return None, ""


def _build_date_filter(column: str, duckdb_type: str) -> str:
    _, base = _normalize_type(duckdb_type)
    if base == "DATE":
        return f"{column} = DATE 'YYYY-MM-DD'"
    return f"CAST({column} AS DATE) = DATE 'YYYY-MM-DD'"


def generate_profile_from_duckdb(
    duckdb_tool: DuckDBTool,
    table_name: str,
    domain: str = "custom",
    dataset_name: str | None = None,
) -> DatasetProfile:
    """Inspect an existing DuckDB table and return a :class:`DatasetProfile`.

    Only reads the schema via ``DESCRIBE`` - never queries data.
    """
    columns = _describe_columns(duckdb_tool, table_name)

    datetime_column, datetime_type = _detect_datetime_column(columns)

    column_profiles: dict[str, ColumnProfile] = {}
    for name, duckdb_type in columns:
        profile_type, semantic_type = _classify(duckdb_type)
        column_profiles[name] = ColumnProfile(
            name=name,
            type=profile_type,
            unit="",
            semantic_type=semantic_type,
        )

    sql_rules: dict[str, str] = {
        "missing_value_count": "COUNT(*) FILTER (WHERE <column> IS NULL)",
    }
    if datetime_column is not None:
        sql_rules["date_filter"] = _build_date_filter(
            datetime_column, datetime_type
        )

    return DatasetProfile(
        name=dataset_name or table_name,
        domain=domain,
        table_name=table_name,
        datetime_column=datetime_column,  # type: ignore[arg-type]
        description="",
        columns=column_profiles,
        sql_rules=sql_rules,
    )


def get_table_summary(
    duckdb_tool: DuckDBTool,
    table_name: str,
) -> dict[str, Any]:
    """Return compact table metadata for API responses."""
    columns = _describe_columns(duckdb_tool, table_name)
    datetime_column, _ = _detect_datetime_column(columns)

    quoted = '"' + table_name.replace('"', '""') + '"'
    count_df = duckdb_tool.execute_query(f"SELECT COUNT(*) AS n FROM {quoted}")
    row_count = int(count_df["n"].iloc[0])

    column_records: list[dict[str, Any]] = []
    numeric_column_names: list[str] = []
    categorical_column_names: list[str] = []

    for name, duckdb_type in columns:
        profile_type, semantic_type = _classify(duckdb_type)
        is_datetime = semantic_type == "datetime" or name == datetime_column
        is_numeric = semantic_type in ("numeric", "integer")
        column_records.append(
            {
                "name": name,
                "duckdb_type": duckdb_type,
                "semantic_type": semantic_type,
                "is_datetime": is_datetime,
                "is_numeric": is_numeric,
            }
        )
        if is_numeric:
            numeric_column_names.append(name)
        if semantic_type == "categorical":
            categorical_column_names.append(name)

    return {
        "table_name": table_name,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": column_records,
        "detected_datetime_column": datetime_column,
        "numeric_column_names": numeric_column_names,
        "categorical_column_names": categorical_column_names,
    }
