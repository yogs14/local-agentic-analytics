"""Helpers for cleaning generated SQL text."""

from __future__ import annotations

import re
from collections.abc import Iterable


DATE_LITERAL_PATTERN = r"(\d{4}-\d{2}-\d{2})"
# A bare DuckDB identifier: starts with a letter/underscore, then word chars.
# Anything else (spaces, commas, leading digits) must be double-quoted.
_SIMPLE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Matches a whole double-quoted identifier or single-quoted string literal so
# their contents can be shielded from identifier rewriting.
_QUOTED_OR_STRING_PATTERN = re.compile(r'"(?:[^"]|"")*"' + r"|'(?:[^']|'')*'")
ENERGY_COLUMNS = (
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)


def identifier_needs_quoting(name: str) -> bool:
    """Return True when a column/table name is not a bare DuckDB identifier.

    Pure-digit names are excluded: quoting them is rarely intended and a digit
    sequence would collide with numeric literals during the rewrite.
    """
    if not name or name.isdigit():
        return False
    return not bool(_SIMPLE_IDENTIFIER_PATTERN.match(name))


def quote_known_identifiers(sql: str, column_names: Iterable[str]) -> str:
    """Double-quote bare occurrences of known columns that DuckDB requires quoted.

    Small local models routinely emit ``AVG(Hydroelectric Power)`` for columns
    whose names contain spaces or punctuation, which DuckDB rejects with a parser
    error. This pass rewrites such bare references to ``AVG("Hydroelectric
    Power")`` using the exact column names from the dataset schema.

    Columns that are already valid bare identifiers are never touched, so clean
    schemas (energy, finance) pass through unchanged. Occurrences inside string
    literals or that are already double-quoted are left alone. Longer names are
    handled first so a column like ``Conventional Hydroelectric Power`` is not
    corrupted by the shorter ``Hydroelectric Power``.
    """
    if not sql:
        return sql

    targets = sorted(
        {name for name in column_names if identifier_needs_quoting(name)},
        key=len,
        reverse=True,
    )
    if not targets:
        return sql

    placeholders: dict[str, str] = {}

    def _store(value: str) -> str:
        key = f"\x00{len(placeholders)}\x00"
        placeholders[key] = value
        return key

    # Shield existing string literals and quoted identifiers from rewriting.
    masked = _QUOTED_OR_STRING_PATTERN.sub(lambda m: _store(m.group(0)), sql)

    for name in targets:
        quoted = '"' + name.replace('"', '""') + '"'
        pattern = re.compile(
            r'(?<![A-Za-z0-9_."])' + re.escape(name) + r'(?![A-Za-z0-9_"])'
        )
        masked = pattern.sub(lambda m, value=quoted: _store(value), masked)

    for key, value in placeholders.items():
        masked = masked.replace(key, value)
    return masked


def clean_sql_response(
    response: str,
    question: str | None = None,
    apply_domain_normalization: bool = True,
) -> str:
    """Remove common wrapping and normalize fragile DuckDB SQL patterns.

    When ``apply_domain_normalization`` is ``False`` the markdown/code-fence
    stripping, first-SELECT extraction, and ``normalize_common_duckdb_sql`` are
    still applied, but ``normalize_energy_domain_sql`` is skipped so the raw
    model SQL is preserved.
    """
    sql = _extract_first_select_statement(response)

    sql = normalize_common_duckdb_sql(sql)
    if question and apply_domain_normalization:
        sql = normalize_energy_domain_sql(sql, question)

    return _strip_trailing_markdown_artifacts(sql)


def extract_raw_model_sql(response: str) -> str:
    """Return model SQL after fence-strip and common normalization only.

    This is the SQL before any domain-specific rewriting, used to measure raw
    model quality independently of the deterministic normalizers.
    """
    return clean_sql_response(response, question=None, apply_domain_normalization=False)


def _extract_first_select_statement(response: str) -> str:
    text = _remove_markdown_fence_lines(response)
    text = _strip_leading_language_or_label(text)

    match = re.search(r"\bSELECT\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :]

    semicolon_index = text.find(";")
    if semicolon_index != -1:
        text = text[: semicolon_index + 1]

    return _strip_trailing_markdown_artifacts(text)


def _remove_markdown_fence_lines(text: str) -> str:
    cleaned_lines = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered in {"```", "```sql", "```duckdb"}:
            continue
        if lowered in {"sql", "duckdb"}:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _strip_leading_language_or_label(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^(sql|duckdb)\s*\n", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"^sql\s*:\s*", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _strip_trailing_markdown_artifacts(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"```(?:sql|duckdb)?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip("` \t\r\n")
    return cleaned.strip()


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


def normalize_energy_domain_sql(sql: str, question: str) -> str:
    """Normalize proven energy-domain mistakes from small local models."""
    normalized = sql
    normalized = _normalize_total_energy_sql(normalized, question)
    normalized = _normalize_record_count_sql(normalized, question)
    normalized = _normalize_missing_value_sql(normalized, question)
    normalized = _normalize_max_active_power_sql(normalized, question)
    normalized = _normalize_energy_aliases(normalized, question)
    return normalized


def _normalize_total_energy_sql(sql: str, question: str) -> str:
    if not _asks_total_energy(question):
        return sql
    if "/ 60" in sql or "/60" in sql:
        return sql

    return re.sub(
        r"SUM\s*\(\s*Global_active_power\s*\)(?:\s+AS\s+[A-Za-z_][A-Za-z0-9_]*)?",
        "SUM(Global_active_power) / 60.0 AS total_energy_kwh",
        sql,
        count=1,
        flags=re.IGNORECASE,
    )


def _normalize_missing_value_sql(sql: str, question: str) -> str:
    if not _asks_missing_value_count(question):
        return sql

    column = _extract_energy_column(question)
    if column is None:
        return sql

    table_name = _extract_table_name(sql) or "electric_power"
    alias = f"missing_{_to_snake_case(column)}_count"
    return (
        f"SELECT COUNT(*) FILTER (WHERE {column} IS NULL) AS {alias} "
        f"FROM {table_name};"
    )


def _normalize_record_count_sql(sql: str, question: str) -> str:
    lowered_question = question.lower()
    if "jumlah record" not in lowered_question and "jumlah data" not in lowered_question:
        return sql

    table_name = _extract_table_name(sql) or "electric_power"
    date_condition = _extract_date_condition(sql) or _date_condition_from_question(question)
    if date_condition:
        return f"SELECT COUNT(*) AS record_count FROM {table_name} WHERE {date_condition};"

    return f"SELECT COUNT(*) AS record_count FROM {table_name};"


def _normalize_energy_aliases(sql: str, question: str) -> str:
    normalized = sql
    lowered_question = question.lower()

    if _asks_average_active_power(lowered_question):
        normalized = _replace_aggregate_alias(
            normalized,
            expression="AVG(Global_active_power)",
            alias="avg_global_active_power_kw",
        )
    if "tegangan rata-rata" in lowered_question or "rata-rata tegangan" in lowered_question:
        normalized = _replace_aggregate_alias(
            normalized,
            expression="AVG(Voltage)",
            alias="avg_voltage_v",
        )
    if "arus rata-rata" in lowered_question or "rata-rata arus" in lowered_question:
        normalized = _replace_aggregate_alias(
            normalized,
            expression="AVG(Global_intensity)",
            alias="avg_global_intensity_a",
        )

    return normalized


def _normalize_max_active_power_sql(sql: str, question: str) -> str:
    lowered_question = question.lower()
    if not _asks_max_active_power_value(lowered_question):
        return sql

    table_name = _extract_table_name(sql) or "electric_power"
    where_clause = _extract_where_clause(sql)
    if where_clause:
        return (
            "SELECT MAX(Global_active_power) AS max_global_active_power_kw "
            f"FROM {table_name} {where_clause};"
        )

    return (
        "SELECT MAX(Global_active_power) AS max_global_active_power_kw "
        f"FROM {table_name};"
    )


def _replace_aggregate_alias(sql: str, expression: str, alias: str) -> str:
    escaped_expression = re.escape(expression)
    return re.sub(
        rf"{escaped_expression}(?:\s+AS\s+[A-Za-z_][A-Za-z0-9_]*)?",
        f"{expression} AS {alias}",
        sql,
        count=1,
        flags=re.IGNORECASE,
    )


def _asks_total_energy(question: str) -> bool:
    lowered = question.lower()
    return any(
        phrase in lowered
        for phrase in (
            "total energi",
            "energi kwh",
            "total kwh",
            "konsumsi energi",
        )
    )


def _asks_missing_value_count(question: str) -> bool:
    lowered = question.lower()
    return (
        "missing value" in lowered
        or "nilai hilang" in lowered
        or "null" in lowered
    )


def _asks_average_active_power(lowered_question: str) -> bool:
    return (
        "rata-rata daya aktif" in lowered_question
        or "rata rata daya aktif" in lowered_question
        or "rata-rata konsumsi daya aktif" in lowered_question
        or "rata rata konsumsi daya aktif" in lowered_question
    )


def _asks_max_active_power_value(lowered_question: str) -> bool:
    asks_maximum = (
        "daya aktif maksimum" in lowered_question
        or "maksimum daya aktif" in lowered_question
        or "daya aktif tertinggi" in lowered_question
    )
    asks_time = (
        "jam berapa" in lowered_question
        or "kapan" in lowered_question
        or "waktu" in lowered_question
    )
    return asks_maximum and not asks_time


def _extract_energy_column(question: str) -> str | None:
    lowered = question.lower()
    for column in ENERGY_COLUMNS:
        if column.lower() in lowered:
            return column

    return None


def _extract_table_name(sql: str) -> str | None:
    match = re.search(r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.IGNORECASE)
    if not match:
        return None

    return match.group(1)


def _extract_where_clause(sql: str) -> str:
    match = re.search(
        r"\bWHERE\b\s+(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|;|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""

    condition = " ".join(match.group(1).strip().split())
    return f"WHERE {condition}" if condition else ""


def _extract_date_condition(sql: str) -> str:
    match = re.search(
        rf"CAST\s*\(\s*datetime\s+AS\s+DATE\s*\)\s*=\s*DATE\s+'{DATE_LITERAL_PATTERN}'",
        sql,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""

    return f"CAST(datetime AS DATE) = DATE '{match.group(1)}'"


def _date_condition_from_question(question: str) -> str:
    match = re.search(
        r"(\d{1,2})\s+Desember\s+(\d{4})",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""

    day = int(match.group(1))
    year = int(match.group(2))
    return f"CAST(datetime AS DATE) = DATE '{year:04d}-12-{day:02d}'"


def _to_snake_case(column: str) -> str:
    return column.lower()
