import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.result_comparison import (
    compare_result_sets,
    parse_result_records,
    values_match,
)


def _records(rows):
    return json.dumps(rows)


def test_parse_result_records():
    assert parse_result_records('[{"a": 1, "b": "x"}]') == [[1, "x"]]
    assert parse_result_records("") is None
    assert parse_result_records(None) is None
    assert parse_result_records("not json") is None
    assert parse_result_records('{"a": 1}') is None
    assert parse_result_records("[]") == []


def test_scalar_match_within_tolerance():
    result = compare_result_sets(
        _records([{"v": 1.0000000001}]), _records([{"v": 1.0}])
    )
    assert result["result_match_full"] is True


def test_scalar_mismatch():
    result = compare_result_sets(_records([{"v": 2.0}]), _records([{"v": 1.0}]))
    assert result["result_match_full"] is False
    assert "value_mismatch" in result["result_match_reason"]


def test_multi_row_order_insensitive():
    agent = _records([{"m": 2, "v": 20.0}, {"m": 1, "v": 10.0}])
    gold = _records([{"month": 1, "value": 10.0}, {"month": 2, "value": 20.0}])

    result = compare_result_sets(agent, gold)

    assert result["result_match_full"] is True


def test_row_count_mismatch():
    agent = _records([{"v": 1.0}])
    gold = _records([{"v": 1.0}, {"v": 2.0}])

    result = compare_result_sets(agent, gold)

    assert result["result_match_full"] is False
    assert "row_count_mismatch" in result["result_match_reason"]


def test_column_count_mismatch():
    agent = _records([{"a": 1, "b": 2}])
    gold = _records([{"a": 1}])

    result = compare_result_sets(agent, gold)

    assert result["result_match_full"] is False
    assert "column_count_mismatch" in result["result_match_reason"]


def test_missing_result_not_comparable():
    result = compare_result_sets("", _records([{"v": 1.0}]))
    assert result["result_match_full"] == ""
    assert result["result_match_reason"] == "missing_result"


def test_both_empty_match():
    result = compare_result_sets("[]", "[]")
    assert result["result_match_full"] is True


def test_date_normalization():
    agent = _records([{"d": "2007-01-15", "v": 5.0}])
    gold = _records([{"d": "2007-01-15T00:00:00.000", "v": 5.0}])

    result = compare_result_sets(agent, gold)

    assert result["result_match_full"] is True


def test_numeric_string_coercion():
    assert values_match("1.5", 1.5) is True
    assert values_match("abc", 1.5) is False
    assert values_match(None, None) is True
    assert values_match(None, 1.0) is False
    assert values_match(True, True) is True
    assert values_match(True, 1) is False


def test_value_mismatch_beyond_tolerance():
    result = compare_result_sets(
        _records([{"v": 1.01}]), _records([{"v": 1.0}])
    )
    assert result["result_match_full"] is False
