"""Row-set comparison for SQL query results (``result_match_full``).

The legacy numeric compare in ``sql_gold_eval`` only handles 1x1 scalar
results, which is why only a subset of gold questions is numerically
comparable. This module compares full result sets instead:

- shape must match (row count and column count),
- rows are compared as an unordered set (sorted canonically first), because
  agent and gold SQL may emit rows in different orders,
- numeric cells match within a float tolerance,
- column NAMES are ignored (aliases legitimately differ); column ORDER is
  significant — a different column order is a real output-shape difference.

This is an additional metric; it never replaces or alters the legacy
``numeric_match`` metric.
"""

from __future__ import annotations

import json
from typing import Any


ABS_TOLERANCE = 1e-6
REL_TOLERANCE = 1e-6
# Floats are rounded to this many decimals to build the canonical sort key;
# it is aligned with the tolerances so near-equal values sort identically.
SORT_KEY_DECIMALS = 6

RESULT_COMPARISON_FIELDS = ("result_match_full", "result_match_reason")

_ZERO_TIME_SUFFIXES = ("T00:00:00.000", "T00:00:00", " 00:00:00")


def parse_result_records(result_text: str | None) -> list[list[Any]] | None:
    """Parse the eval's stored result text (pandas ``orient='records'`` JSON).

    Returns rows as lists of cell values (insertion order preserved), or
    ``None`` when the text is empty or not a records-style JSON list.
    """
    if result_text is None or not str(result_text).strip():
        return None
    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, list):
        return None

    rows: list[list[Any]] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        rows.append(list(item.values()))
    return rows


def compare_result_sets(
    agent_result_text: str | None,
    gold_result_text: str | None,
    abs_tolerance: float = ABS_TOLERANCE,
    rel_tolerance: float = REL_TOLERANCE,
) -> dict[str, Any]:
    """Compare two stored result sets as unordered rows with float tolerance.

    Returns ``{"result_match_full": bool | "", "result_match_reason": str}``.
    ``result_match_full`` is ``""`` (not comparable) when either side has no
    parseable result, mirroring how the legacy metric reports blanks.
    """
    agent_rows = parse_result_records(agent_result_text)
    gold_rows = parse_result_records(gold_result_text)

    if agent_rows is None or gold_rows is None:
        return {
            "result_match_full": "",
            "result_match_reason": "missing_result",
        }

    if len(agent_rows) != len(gold_rows):
        return {
            "result_match_full": False,
            "result_match_reason": (
                f"row_count_mismatch (agent={len(agent_rows)}, "
                f"gold={len(gold_rows)})"
            ),
        }

    if not gold_rows:
        return {"result_match_full": True, "result_match_reason": "both_empty"}

    agent_width = _row_width(agent_rows)
    gold_width = _row_width(gold_rows)
    if agent_width != gold_width:
        return {
            "result_match_full": False,
            "result_match_reason": (
                f"column_count_mismatch (agent={agent_width}, "
                f"gold={gold_width})"
            ),
        }

    agent_sorted = sorted(agent_rows, key=_row_sort_key)
    gold_sorted = sorted(gold_rows, key=_row_sort_key)

    for row_index, (agent_row, gold_row) in enumerate(
        zip(agent_sorted, gold_sorted)
    ):
        for col_index, (agent_cell, gold_cell) in enumerate(
            zip(agent_row, gold_row)
        ):
            if not values_match(
                agent_cell, gold_cell, abs_tolerance, rel_tolerance
            ):
                return {
                    "result_match_full": False,
                    "result_match_reason": (
                        f"value_mismatch (sorted row {row_index}, column "
                        f"{col_index}: agent={agent_cell!r}, gold={gold_cell!r})"
                    ),
                }

    return {"result_match_full": True, "result_match_reason": "rows_match"}


def values_match(
    agent_value: Any,
    gold_value: Any,
    abs_tolerance: float = ABS_TOLERANCE,
    rel_tolerance: float = REL_TOLERANCE,
) -> bool:
    """Compare two cells: floats within tolerance, strings normalized."""
    if agent_value is None and gold_value is None:
        return True
    if agent_value is None or gold_value is None:
        return False

    # bool is an int subclass, so it must be checked before numerics.
    if isinstance(agent_value, bool) or isinstance(gold_value, bool):
        return agent_value is gold_value

    agent_number = _coerce_number(agent_value)
    gold_number = _coerce_number(gold_value)
    if agent_number is not None and gold_number is not None:
        return _numbers_match(agent_number, gold_number, abs_tolerance, rel_tolerance)

    return _normalize_text(str(agent_value)) == _normalize_text(str(gold_value))


def _numbers_match(
    agent_number: float,
    gold_number: float,
    abs_tolerance: float,
    rel_tolerance: float,
) -> bool:
    difference = abs(agent_number - gold_number)
    if difference <= abs_tolerance:
        return True
    if gold_number != 0 and difference / abs(gold_number) <= rel_tolerance:
        return True
    return False


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _normalize_text(text: str) -> str:
    """Strip whitespace and zero-time suffixes so ``2007-01-15T00:00:00.000``
    equals ``2007-01-15`` (agent and gold may serialize dates differently)."""
    normalized = text.strip()
    for suffix in _ZERO_TIME_SUFFIXES:
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _row_width(rows: list[list[Any]]) -> int:
    return max(len(row) for row in rows)


def _row_sort_key(row: list[Any]) -> tuple:
    return tuple(_cell_sort_key(cell) for cell in row)


def _cell_sort_key(cell: Any) -> tuple[int, str]:
    """Type-ranked canonical key so heterogeneous cells sort deterministically.

    Floats are rounded (aligned with the match tolerance) so two values that
    would match also sort to the same position on both sides.
    """
    if cell is None:
        return (0, "")
    if isinstance(cell, bool):
        return (1, str(cell))
    number = _coerce_number(cell)
    if number is not None:
        return (2, f"{round(number, SORT_KEY_DECIMALS):.{SORT_KEY_DECIMALS}f}")
    return (3, _normalize_text(str(cell)))
