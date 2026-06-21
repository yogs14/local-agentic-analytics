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
        datetime_column = dataset_profile.datetime_column

        if not table_name or not datetime_column:
            return None

        parsed_date = _parse_date(question)
        if _is_missing_value_question(normalized_question):
            return _resolve_missing_value_question(
                question=question,
                normalized_question=normalized_question,
                dataset_profile=dataset_profile,
                table_name=table_name,
            )

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
    column_name = _find_mentioned_column(question, dataset_profile)
    if column_name is None:
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
    return "missing value" in text or "nilai hilang" in text


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
