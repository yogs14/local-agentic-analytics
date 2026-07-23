"""Deterministic SQL resolver for common low-risk analytics questions."""

from __future__ import annotations

from datetime import date
import re

from local_agentic_analytics.core.dataset_profile import DatasetProfile


INDONESIAN_MONTHS = {
    "januari": 1,
    "februari": 2,
    "maret": 3,
    "april": 4,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "agustus": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "desember": 12,
}


class RuleBasedSQLResolver:
    """Resolve frequent low-risk questions into deterministic DuckDB SQL."""

    def resolve(self, question: str, dataset_profile: DatasetProfile) -> str | None:
        if not question or not question.strip():
            return None

        normalized_question = _normalize_text(question)
        table_name = dataset_profile.table_name

        if not table_name:
            return None

        # Domain-agnostic intent that does not need a datetime column: averaging
        # every numeric column. Resolved from the profile so text columns are
        # never aggregated (which would raise avg(VARCHAR)) and wide tables are
        # never truncated by the LLM.
        if _is_numeric_summary_question(normalized_question):
            summary_sql = _build_numeric_summary_sql(dataset_profile, table_name)
            if summary_sql is not None:
                return summary_sql

        # Null/missing-value counts per column also do not need a datetime
        # column, so resolve them before the datetime guard below.
        if _is_missing_value_question(normalized_question):
            missing_sql = _resolve_missing_value_question(
                question=question,
                normalized_question=normalized_question,
                dataset_profile=dataset_profile,
                table_name=table_name,
            )
            if missing_sql is not None:
                return missing_sql

        datetime_column = dataset_profile.datetime_column
        if not datetime_column:
            return None

        parsed_date = _parse_date(question)

        domain = dataset_profile.domain.lower()
        if domain == "finance":
            return _resolve_finance_question(
                normalized_question=normalized_question,
                dataset_profile=dataset_profile,
                table_name=table_name,
                datetime_column=datetime_column,
                parsed_date=parsed_date,
            )

        if domain != "energy":
            return None

        if _is_average_active_power_question(normalized_question):
            if not parsed_date or "Global_active_power" not in dataset_profile.columns:
                return None
            return (
                "SELECT AVG(Global_active_power) AS avg_global_active_power_kw\n"
                f"FROM {table_name}\n"
                f"WHERE CAST({datetime_column} AS DATE) = DATE '{parsed_date}';"
            )

        if _is_total_energy_question(normalized_question):
            if not parsed_date or "Global_active_power" not in dataset_profile.columns:
                return None
            return (
                "SELECT SUM(Global_active_power) / 60.0 AS total_energy_kwh\n"
                f"FROM {table_name}\n"
                f"WHERE CAST({datetime_column} AS DATE) = DATE '{parsed_date}';"
            )

        return None


FINANCE_TICKERS = ("NVDA", "NFLX", "TSLA", "GOOGL")


def _resolve_finance_question(
    normalized_question: str,
    dataset_profile: DatasetProfile,
    table_name: str,
    datetime_column: str,
    parsed_date: str | None,
) -> str | None:
    ticker = _find_ticker(normalized_question)

    if _is_average_close_price_question(normalized_question):
        if "close" not in dataset_profile.columns:
            return None
        return _build_finance_aggregate_sql(
            expression="AVG(close)",
            alias="avg_close_usd",
            table_name=table_name,
            datetime_column=datetime_column,
            parsed_date=parsed_date,
            ticker=ticker,
        )

    if _is_max_close_price_question(normalized_question):
        if "close" not in dataset_profile.columns:
            return None
        return _build_finance_aggregate_sql(
            expression="MAX(close)",
            alias="max_close_usd",
            table_name=table_name,
            datetime_column=datetime_column,
            parsed_date=parsed_date,
            ticker=ticker,
        )

    if _is_average_volume_question(normalized_question):
        if "volume" not in dataset_profile.columns:
            return None
        return _build_finance_aggregate_sql(
            expression="AVG(volume)",
            alias="avg_volume_shares",
            table_name=table_name,
            datetime_column=datetime_column,
            parsed_date=parsed_date,
            ticker=ticker,
        )

    return None


def _resolve_missing_value_question(
    question: str,
    normalized_question: str,
    dataset_profile: DatasetProfile,
    table_name: str,
) -> str | None:
    # "every/all column" phrasings always mean a per-column null report.
    if _mentions_all_columns(normalized_question):
        return _build_all_missing_values_sql(dataset_profile, table_name)

    column_name = _find_mentioned_column(question, dataset_profile)
    if column_name is None:
        # A named-but-unknown column ("kolom <x>") stays ambiguous -> defer to
        # the LLM; otherwise default to a per-column null report.
        if "kolom" in normalized_question or "column" in normalized_question:
            return None
        return _build_all_missing_values_sql(dataset_profile, table_name)

    alias = f"missing_{_alias_column_name(column_name)}_count"
    return (
        f"SELECT COUNT(*) FILTER (WHERE {column_name} IS NULL) AS {alias}\n"
        f"FROM {table_name};"
    )


def _build_finance_aggregate_sql(
    expression: str,
    alias: str,
    table_name: str,
    datetime_column: str,
    parsed_date: str | None,
    ticker: str | None,
) -> str:
    conditions: list[str] = []
    if ticker:
        conditions.append(f"ticker = '{ticker}'")
    if parsed_date:
        conditions.append(f"CAST({datetime_column} AS DATE) = DATE '{parsed_date}'")

    sql = f"SELECT {expression} AS {alias}\nFROM {table_name}"
    if conditions:
        sql += "\nWHERE " + " AND ".join(conditions)
    return f"{sql};"


def _find_ticker(text: str) -> str | None:
    for ticker in FINANCE_TICKERS:
        if re.search(rf"\b{ticker.lower()}\b", text):
            return ticker
    return None


def _build_all_missing_values_sql(
    dataset_profile: DatasetProfile,
    table_name: str,
) -> str | None:
    columns = [
        column_name
        for column_name, column in dataset_profile.columns.items()
        if column.type.lower() != "date"
    ]
    if not columns:
        return None

    select_items = [
        (
            f"COUNT(*) FILTER (WHERE {column_name} IS NULL) "
            f"AS missing_{_alias_column_name(column_name)}_count"
        )
        for column_name in columns
    ]
    return "SELECT " + ", ".join(select_items) + f"\nFROM {table_name};"


# DuckDB type names (and profile types) treated as aggregatable numbers.
_NUMERIC_COLUMN_TYPES = frozenset({
    "numeric", "integer", "double", "float", "real", "decimal", "bigint",
    "int", "smallint", "tinyint", "hugeint", "ubigint", "uinteger",
    "usmallint", "utinyint", "float4", "float8", "int1", "int2", "int4", "int8",
})


def _is_numeric_summary_question(text: str) -> bool:
    """Detect "average of every numeric column" style questions.

    Kept conservative so it never shadows the domain-specific aggregate rules:
    it requires an average word plus an explicit "numeric"/"all columns" hint.
    """
    if not _has_average(text):
        return False
    return any(
        hint in text
        for hint in (
            "numerik",
            "numeric",
            "setiap kolom",
            "semua kolom",
            "tiap kolom",
            "seluruh kolom",
            "setiap kolom numerik",
            "each column",
            "every column",
            "all columns",
            "all numeric",
        )
    )


def _is_numeric_column(column) -> bool:
    return (
        column.semantic_type.lower() in {"numeric", "integer"}
        or column.type.lower() in _NUMERIC_COLUMN_TYPES
    )


def _build_numeric_summary_sql(
    dataset_profile: DatasetProfile,
    table_name: str,
) -> str | None:
    numeric_columns = [
        column_name
        for column_name, column in dataset_profile.columns.items()
        if _is_numeric_column(column)
    ]
    if not numeric_columns:
        return None

    select_items = [
        f'AVG({_quote_identifier(column_name)}) '
        f"AS avg_{_alias_column_name(column_name)}"
        for column_name in numeric_columns
    ]
    return "SELECT " + ", ".join(select_items) + f"\nFROM {table_name};"


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _is_average_active_power_question(text: str) -> bool:
    has_average = "rata rata" in text or "rata-rata" in text
    has_active_power = "daya aktif" in text or "konsumsi daya aktif" in text
    return has_average and has_active_power


def _is_total_energy_question(text: str) -> bool:
    return any(
        phrase in text
        for phrase in ("total energi", "energi kwh", "total kwh")
    )


def _is_missing_value_question(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "missing value",
            "missing values",
            "nilai hilang",
            "nilai kosong",
            "null",
        )
    )


def _mentions_all_columns(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "tiap kolom",
            "setiap kolom",
            "semua kolom",
            "seluruh kolom",
            "masing-masing kolom",
            "masing masing kolom",
            "per kolom",
            "each column",
            "every column",
            "all columns",
            "all column",
        )
    )


def _is_average_close_price_question(text: str) -> bool:
    return _has_average(text) and (
        "harga penutupan" in text or "close price" in text or "close_price" in text
    )


def _is_max_close_price_question(text: str) -> bool:
    return _has_maximum(text) and (
        "harga penutupan" in text or "close price" in text or "close_price" in text
    )


def _is_average_volume_question(text: str) -> bool:
    return _has_average(text) and "volume" in text


def _has_average(text: str) -> bool:
    return "rata rata" in text or "rata-rata" in text or "average" in text


def _has_maximum(text: str) -> bool:
    return any(
        phrase in text
        for phrase in ("maksimum", "maximum", "tertinggi", "paling tinggi", "max")
    )


def _parse_date(text: str) -> str | None:
    iso_match = re.search(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", text)
    if iso_match:
        return _safe_iso_date(
            int(iso_match.group(1)),
            int(iso_match.group(2)),
            int(iso_match.group(3)),
        )

    month_names = "|".join(INDONESIAN_MONTHS)
    date_match = re.search(
        rf"\b(\d{{1,2}})\s+({month_names})\s+((?:19|20)\d{{2}})\b",
        text.lower(),
    )
    if not date_match:
        return None

    day = int(date_match.group(1))
    month = INDONESIAN_MONTHS[date_match.group(2)]
    year = int(date_match.group(3))
    return _safe_iso_date(year, month, day)


def _safe_iso_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _find_mentioned_column(
    question: str,
    dataset_profile: DatasetProfile,
) -> str | None:
    question_lower = question.lower()
    normalized_question = _normalize_column_text(question)

    for column_name in dataset_profile.columns:
        column_lower = column_name.lower()
        normalized_column = _normalize_column_text(column_name)
        if column_lower in question_lower or normalized_column in normalized_question:
            return column_name

    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _normalize_column_text(text: str) -> str:
    lowered = text.lower().replace("_", " ")
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _alias_column_name(column_name: str) -> str:
    alias = re.sub(r"[^a-z0-9]+", "_", column_name.lower())
    return alias.strip("_")
