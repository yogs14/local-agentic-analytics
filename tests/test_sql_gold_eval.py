from pathlib import Path
import csv
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.sql_gold_eval import (
    SQL_GOLD_EVAL_COLUMNS,
    compare_numeric_results,
    extract_numeric_scalar,
    load_gold_questions,
    run_sql_gold_evaluation,
    write_sql_gold_results,
)


class FakeWorkflow:
    def __init__(self, sql: str = "SELECT 1 AS value"):
        self.sql = sql

    def run(self, user_query: str) -> AnalyticsState:
        return AnalyticsState(
            user_query=user_query,
            generated_sql=self.sql,
            sql_result={"rows": [{"value": 1.0}]},
            success=True,
        )


class FakeDuckDBTool:
    def __init__(self, results_by_sql):
        self.results_by_sql = results_by_sql

    def execute_query(self, sql: str) -> pd.DataFrame:
        result = self.results_by_sql[sql]
        if isinstance(result, Exception):
            raise result
        return result


def test_load_gold_questions_reads_valid_json(tmp_path):
    questions_path = tmp_path / "gold_questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "E001",
                    "question": "Berapa rata-rata?",
                    "gold_sql_file": str(tmp_path / "E001.sql"),
                    "expected_unit": "kW",
                }
            ]
        ),
        encoding="utf-8",
    )

    questions = load_gold_questions(questions_path)

    assert questions[0]["id"] == "E001"
    assert questions[0]["gold_sql_file"].endswith("E001.sql")


def test_run_sql_gold_evaluation_compares_numeric_scalar(tmp_path):
    gold_sql_path = tmp_path / "E001.sql"
    gold_sql = "SELECT 1.0 AS value"
    gold_sql_path.write_text(gold_sql, encoding="utf-8")
    questions = [
        {
            "id": "E001",
            "question": "Berapa nilai?",
            "gold_sql_file": str(gold_sql_path),
            "expected_unit": "count",
        }
    ]
    duckdb_tool = FakeDuckDBTool(
        {
            "SELECT 1 AS value": pd.DataFrame({"value": [1.0]}),
            gold_sql: pd.DataFrame({"value": [1.0]}),
        }
    )

    rows = run_sql_gold_evaluation(
        questions,
        workflow=FakeWorkflow(sql="SELECT 1 AS value"),
        duckdb_tool=duckdb_tool,
    )

    assert rows[0]["agent_success"] is True
    assert rows[0]["gold_success"] is True
    assert rows[0]["numeric_match"] is True
    assert rows[0]["absolute_error"] == 0.0


def test_run_sql_gold_evaluation_continues_when_gold_sql_missing(tmp_path):
    questions = [
        {
            "id": "E001",
            "question": "Berapa nilai?",
            "gold_sql_file": str(tmp_path / "missing.sql"),
            "expected_unit": "count",
        }
    ]
    duckdb_tool = FakeDuckDBTool(
        {"SELECT 1 AS value": pd.DataFrame({"value": [1.0]})}
    )

    rows = run_sql_gold_evaluation(
        questions,
        workflow=FakeWorkflow(sql="SELECT 1 AS value"),
        duckdb_tool=duckdb_tool,
    )

    assert len(rows) == 1
    assert rows[0]["agent_success"] is True
    assert rows[0]["gold_success"] is False
    assert "gold_sql_load" in rows[0]["error_message"]


def test_non_scalar_result_is_not_numeric_compared():
    dataframe = pd.DataFrame({"a": [1.0], "b": [2.0]})

    assert extract_numeric_scalar(dataframe) is None
    assert compare_numeric_results(None, 1.0)["numeric_match"] == ""


def test_write_sql_gold_results_writes_expected_columns(tmp_path):
    output_path = tmp_path / "sql_gold_eval.csv"
    rows = [
        {
            "question_id": "E001",
            "question": "test",
            "agent_sql": "SELECT 1",
            "gold_sql": "SELECT 1",
            "agent_success": True,
            "gold_success": True,
            "numeric_match": True,
            "absolute_error": 0.0,
            "relative_error": 0.0,
            "agent_result": "[{\"value\":1}]",
            "gold_result": "[{\"value\":1}]",
            "error_message": "",
        }
    ]

    write_sql_gold_results(rows, output_path)

    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        loaded_rows = list(reader)

    assert reader.fieldnames == list(SQL_GOLD_EVAL_COLUMNS)
    assert loaded_rows[0]["question_id"] == "E001"
