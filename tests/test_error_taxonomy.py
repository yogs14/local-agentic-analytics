import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.error_taxonomy import (
    CATEGORY_AGGREGATION,
    CATEGORY_DATE,
    CATEGORY_GROUPING,
    CATEGORY_NL_UNDERSTANDING,
    CATEGORY_OTHER,
    CATEGORY_OUTPUT_SHAPE,
    CATEGORY_SCHEMA,
    CATEGORY_SYNTAX,
    CATEGORY_UNIT,
    GOLD_ERROR_LABEL,
    LATE_MATCH_LABEL,
    build_schema_vocabulary,
    classify_benchmark_rows,
    classify_failure,
    distribution_by_group,
    extract_sql_features,
    is_failed_row,
)


TABLE = "energy_consumption"
GOLD_AVG = (
    f"SELECT AVG(global_active_power) FROM {TABLE} "
    "WHERE CAST(ts AS DATE) = '2007-01-15'"
)


def _vocabulary():
    return build_schema_vocabulary(
        [
            GOLD_AVG,
            f"SELECT SUM(global_active_power) / 60.0 FROM {TABLE}",
            f"SELECT AVG(voltage) FROM {TABLE}",
        ]
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def test_extract_sql_features_basics():
    features = extract_sql_features(
        f"SELECT DATE_TRUNC('month', ts) AS m, SUM(global_active_power) / 60.0 "
        f"FROM {TABLE} WHERE ts >= '2007-01-01' AND EXTRACT(year FROM ts) = 2007 "
        "GROUP BY DATE_TRUNC('month', ts) HAVING SUM(global_active_power) > 10"
    )

    assert features.parse_ok is True
    assert TABLE in features.tables
    assert "global_active_power" in features.columns
    assert "sum" in features.aggregates
    assert "2007-01-01" in features.date_literals
    assert "2007" in features.year_literals
    assert "60" in features.arithmetic_constants
    assert features.has_group_by is True
    assert features.has_having is True


def test_extract_sql_features_invalid_sql():
    features = extract_sql_features("SELECT FROM WHERE !!!")
    assert features.parse_ok is False

    empty = extract_sql_features("")
    assert empty.parse_ok is False


# ---------------------------------------------------------------------------
# Classification order (first match wins)
# ---------------------------------------------------------------------------


def test_no_sql_is_syntax_error():
    result = classify_failure("", GOLD_AVG)
    assert result.category == CATEGORY_SYNTAX


def test_parser_error_message_is_syntax_error():
    result = classify_failure(
        f"SELECT AVG(global_active_power) FROM {TABLE}",
        GOLD_AVG,
        error_message='Parser Error: syntax error at or near "FROM"',
    )
    assert result.category == CATEGORY_SYNTAX


def test_binder_error_is_schema_linking():
    result = classify_failure(
        f"SELECT AVG(active_power) FROM {TABLE}",
        GOLD_AVG,
        error_message='Binder Error: column "active_power" does not exist',
    )
    assert result.category == CATEGORY_SCHEMA


def test_hallucinated_column_is_schema_linking():
    result = classify_failure(
        f"SELECT AVG(power_usage) FROM {TABLE} "
        "WHERE CAST(ts AS DATE) = '2007-01-15'",
        GOLD_AVG,
        vocabulary=_vocabulary(),
    )
    assert result.category == CATEGORY_SCHEMA
    assert "power_usage" in result.evidence


def test_missing_conversion_is_unit_conversion():
    result = classify_failure(
        f"SELECT SUM(global_active_power) FROM {TABLE}",
        f"SELECT SUM(global_active_power) / 60.0 FROM {TABLE}",
        vocabulary=_vocabulary(),
    )
    assert result.category == CATEGORY_UNIT


def test_wrong_date_is_date_filter():
    result = classify_failure(
        f"SELECT AVG(global_active_power) FROM {TABLE} "
        "WHERE CAST(ts AS DATE) = '2007-01-16'",
        GOLD_AVG,
        vocabulary=_vocabulary(),
    )
    assert result.category == CATEGORY_DATE


def test_wrong_aggregate_is_aggregation_choice():
    result = classify_failure(
        f"SELECT SUM(global_active_power) FROM {TABLE} "
        "WHERE CAST(ts AS DATE) = '2007-01-15'",
        GOLD_AVG,
        vocabulary=_vocabulary(),
    )
    assert result.category == CATEGORY_AGGREGATION


def test_group_by_difference_is_grouping_logic():
    result = classify_failure(
        f"SELECT AVG(global_active_power) FROM {TABLE} "
        "WHERE CAST(ts AS DATE) = '2007-01-15' GROUP BY voltage",
        GOLD_AVG,
        vocabulary=_vocabulary(),
    )
    assert result.category == CATEGORY_GROUPING


def test_wrong_existing_column_is_nl_understanding():
    result = classify_failure(
        f"SELECT AVG(voltage) FROM {TABLE} "
        "WHERE CAST(ts AS DATE) = '2007-01-15'",
        GOLD_AVG,
        vocabulary=_vocabulary(),
    )
    assert result.category == CATEGORY_NL_UNDERSTANDING
    assert result.needs_manual_review is True


def test_value_overlap_is_output_shape():
    agent_result = json.dumps([{"d": "2007-01-15", "v": 1.23}])
    gold_result = json.dumps([{"v": 1.23}])

    result = classify_failure(
        GOLD_AVG,
        GOLD_AVG,
        agent_result_text=agent_result,
        gold_result_text=gold_result,
        vocabulary=_vocabulary(),
    )
    assert result.category == CATEGORY_OUTPUT_SHAPE


def test_fallback_is_other_with_manual_review():
    agent_result = json.dumps([{"v": 9.99}])
    gold_result = json.dumps([{"v": 1.23}])

    result = classify_failure(
        GOLD_AVG,
        GOLD_AVG,
        agent_result_text=agent_result,
        gold_result_text=gold_result,
        vocabulary=_vocabulary(),
    )
    assert result.category == CATEGORY_OTHER
    assert result.needs_manual_review is True


# ---------------------------------------------------------------------------
# Failure detection on CSV-style rows
# ---------------------------------------------------------------------------


def test_is_failed_row():
    assert is_failed_row(
        {"gold_success": "True", "execution_success": "False"}
    ) is True
    assert is_failed_row(
        {
            "gold_success": "True",
            "execution_success": "True",
            "numeric_match": "True",
            "result_match_full": "",
        }
    ) is False
    assert is_failed_row(
        {
            "gold_success": "True",
            "execution_success": "True",
            "numeric_match": "",
            "result_match_full": "True",
        }
    ) is False
    assert is_failed_row(
        {
            "gold_success": "True",
            "execution_success": "True",
            "numeric_match": "False",
            "result_match_full": "False",
        }
    ) is True
    # Executed but nothing comparable -> still counts as a failure to review.
    assert is_failed_row(
        {
            "gold_success": "True",
            "execution_success": "True",
            "numeric_match": "",
            "result_match_full": "",
        }
    ) is True
    # Gold itself failed -> not an agent failure.
    assert is_failed_row(
        {"gold_success": "False", "execution_success": "False"}
    ) is False
    # sql_gold format uses agent_success instead of execution_success.
    assert is_failed_row(
        {"gold_success": "True", "agent_success": "False"}
    ) is True


def test_classify_benchmark_rows_end_to_end():
    questions_by_id = {
        "E101": {
            "id": "E101",
            "question": "Q1",
            "gold_sql_file": "references/sql_gold/E101.sql",
            "expected_unit": "kW",
        }
    }
    rows = [
        {  # success row: skipped
            "question_id": "E101",
            "gold_success": "True",
            "execution_success": "True",
            "numeric_match": "True",
            "result_match_full": "True",
            "config": "D_full",
        },
        {  # gold failure: separated out
            "question_id": "E101",
            "gold_success": "False",
            "execution_success": "False",
            "config": "D_full",
        },
        {  # agent failure: classified (no SQL -> syntax)
            "question_id": "E101",
            "gold_success": "True",
            "execution_success": "False",
            "numeric_match": "",
            "agent_sql": "",
            "config": "A_raw_llm",
            "error_message": "Agent did not produce SQL",
        },
    ]

    classified, gold_errors, late_matches = classify_benchmark_rows(
        rows,
        questions_by_id,
        vocabulary=_vocabulary(),
        gold_sql_loader=lambda path: GOLD_AVG,
        metadata={"source_file": "x.csv", "suite": "ablation", "model": "m1"},
    )

    assert len(classified) == 1
    assert classified[0]["category"] == CATEGORY_SYNTAX
    assert classified[0]["gold_sql"] == GOLD_AVG
    assert classified[0]["model"] == "m1"
    assert len(gold_errors) == 1
    assert gold_errors[0]["category"] == GOLD_ERROR_LABEL
    assert late_matches == []

    distribution = distribution_by_group(classified)
    assert distribution == {"m1/A_raw_llm": {CATEGORY_SYNTAX: 1}}


def test_classify_benchmark_rows_detects_late_row_set_match():
    """Legacy rows whose results fully match under row-set comparison are
    successes the scalar metric skipped, not errors."""
    matching_result = json.dumps([{"m": 1, "v": 10.0}, {"m": 2, "v": 20.0}])

    def executor(sql):
        return True, matching_result, ""

    rows = [
        {  # legacy ablation format: no result_match_full column at all
            "question_id": "E109",
            "gold_success": "True",
            "execution_success": "True",
            "numeric_match": "",
            "agent_sql": GOLD_AVG,
            "config": "D_full",
        }
    ]

    classified, gold_errors, late_matches = classify_benchmark_rows(
        rows,
        {"E109": {"id": "E109", "gold_sql_file": "E109.sql"}},
        vocabulary=_vocabulary(),
        sql_executor=executor,
        gold_sql_loader=lambda path: GOLD_AVG,
    )

    assert classified == []
    assert gold_errors == []
    assert len(late_matches) == 1
    assert late_matches[0]["category"] == LATE_MATCH_LABEL


# ---------------------------------------------------------------------------
# EN gold set parity
# ---------------------------------------------------------------------------


def test_en_gold_set_mirrors_v2_except_question_text():
    v2_path = (
        PROJECT_ROOT / "references" / "sql_gold" / "energy_gold_questions_v2.json"
    )
    en_path = (
        PROJECT_ROOT
        / "references"
        / "sql_gold"
        / "energy_gold_questions_v2_en.json"
    )
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    en = json.loads(en_path.read_text(encoding="utf-8"))

    assert len(en) == len(v2) == 36
    for item_v2, item_en in zip(v2, en):
        assert item_en["id"] == item_v2["id"]
        assert item_en["gold_sql_file"] == item_v2["gold_sql_file"]
        assert item_en["expected_unit"] == item_v2["expected_unit"]
        assert item_en["category"] == item_v2["category"]
        assert item_en["difficulty"] == item_v2["difficulty"]
        assert item_en["question"] != item_v2["question"]
