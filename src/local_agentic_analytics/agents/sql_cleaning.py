"""Helpers for cleaning generated SQL text."""

from __future__ import annotations

import re


DATE_LITERAL_PATTERN = r"(\d{4}-\d{2}-\d{2})"


def clean_sql_response(response: str) -> str:
    """Remove common wrapping and normalize fragile DuckDB SQL patterns."""
    sql = response.strip()

    if sql.startswith("```"):
        lines = sql.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql = "\n".join(lines).strip()

    if sql.lower().startswith("sql:"):
        sql = sql[4:].strip()

    return normalize_common_duckdb_sql(sql)


def normalize_common_duckdb_sql(sql: str) -> str:
    """Normalize common date-filter mistakes from small local models."""
    normalized = re.sub(
        rf"\bdatetime\s*=\s*CAST\s*\(\s*datetime\s+AS\s+DATE\s*\)\s*=\s*DATE\s+'{DATE_LITERAL_PATTERN}'",
        r"CAST(datetime AS DATE) = DATE '\1'",
        sql,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        rf"\bdatetime\s*=\s*DATE\s+'{DATE_LITERAL_PATTERN}'",
        r"CAST(datetime AS DATE) = DATE '\1'",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        rf"\bdatetime\s*=\s*CAST\s*\(\s*'{DATE_LITERAL_PATTERN}'\s+AS\s+DATE\s*\)",
        r"CAST(datetime AS DATE) = DATE '\1'",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized
