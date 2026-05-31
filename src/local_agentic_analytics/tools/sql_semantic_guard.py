"""Small semantic guardrails for generated SQL."""

from __future__ import annotations

import re


def validate_energy_sql_semantics(question: str, sql: str) -> tuple[bool, str]:
    """Validate common energy-domain SQL semantics.

    This guard is intentionally narrow. It only protects against historical
    regressions for total energy conversion and missing value counts.
    """
    if not sql or not sql.strip():
        return False, "SQL must not be empty."

    normalized_question = _normalize(question)
    normalized_sql = _normalize(sql)

    if re.search(r"count\s*\(\s*distinct\s+case\s+when", normalized_sql):
        return (
            False,
            "Missing value counts must not use COUNT(DISTINCT CASE WHEN ...). "
            "Use COUNT(*) FILTER (WHERE <column> IS NULL).",
        )

    if _asks_total_energy(normalized_question) and not re.search(
        r"/\s*60(?:\.0+)?\b",
        normalized_sql,
    ):
        return (
            False,
            "Total energy kWh queries must divide Global_active_power by 60.0.",
        )

    if _asks_missing_value(normalized_question) and not (
        re.search(r"count\s*\(\s*\*\s*\)\s*filter", normalized_sql)
        or "count_if" in normalized_sql
    ):
        return (
            False,
            "Missing value queries must use COUNT(*) FILTER or COUNT_IF.",
        )

    return True, ""


def _asks_total_energy(question: str) -> bool:
    return any(
        phrase in question
        for phrase in ("total energi", "energi kwh", "total kwh")
    )


def _asks_missing_value(question: str) -> bool:
    return any(
        phrase in question
        for phrase in ("missing value", "nilai hilang", "null")
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()
