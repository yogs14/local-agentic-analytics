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
    """Resolve frequent energy questions into deterministic DuckDB SQL."""

    def resolve(self, question: str, dataset_profile: DatasetProfile) -> str | None:
        if not question or not question.strip():
            return None
        if dataset_profile.domain.lower() != "energy":
            return None

        normalized_question = _normalize_text(question)
        table_name = dataset_profile.table_name
        datetime_column = dataset_profile.datetime_column

        if not table_name or not datetime_column:
            return None

        parsed_date = _parse_date(question)
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

        if _is_missing_value_question(normalized_question):
            column_name = _find_mentioned_column(question, dataset_profile)
            if column_name is None:
                return None
            alias = f"missing_{_alias_column_name(column_name)}_count"
            return (
                f"SELECT COUNT(*) FILTER (WHERE {column_name} IS NULL) AS {alias}\n"
                f"FROM {table_name};"
            )

        return None


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
