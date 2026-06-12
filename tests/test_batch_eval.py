from pathlib import Path
import csv
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.core.state import AnalyticsState
from local_agentic_analytics.evaluation.batch_eval import (
    BATCH_EVAL_COLUMNS,
    load_questions,
    run_batch_evaluation,
    state_to_batch_row,
    write_batch_results,
)


class FakeWorkflow:
    def run(self, user_query: str) -> AnalyticsState:
        if "gagal" in user_query:
            raise RuntimeError("workflow error")

        return AnalyticsState(
            user_query=user_query,
            generated_sql="SELECT 1",
            repaired_sql=None,
            sql_result={"rows": [{"value": 1}]},
            final_answer="Jawaban ringkas.",
            latency={"total": 2.0, "sql_generation": 1.0},
            success=True,
        )


def test_load_questions_reads_valid_json(tmp_path):
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "E001",
                    "question": "Berapa rata-rata konsumsi?",
                    "expected_unit": "kW",
                    "category": "aggregation",
                }
            ]
        ),
        encoding="utf-8",
    )

    questions = load_questions(questions_path)

    assert questions[0]["id"] == "E001"
    assert questions[0]["expected_unit"] == "kW"


def test_run_batch_evaluation_continues_after_question_failure():
    questions = [
        {
            "id": "E001",
            "question": "pertanyaan sukses",
            "expected_unit": "kW",
            "category": "aggregation",
        },
        {
            "id": "E002",
            "question": "pertanyaan gagal",
            "expected_unit": "kW",
            "category": "aggregation",
        },
    ]

    rows, summary = run_batch_evaluation(questions, workflow=FakeWorkflow())

    assert len(rows) == 2
    assert rows[0]["success"] is True
    assert rows[1]["success"] is False
    assert rows[1]["error_message"] == "workflow error"
    assert summary["total_questions"] == 2
    assert summary["success_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["success_rate"] == 0.5


def test_state_to_batch_row_sets_sql_execution_success_from_sql_result():
    state = AnalyticsState(
        user_query="test",
        generated_sql="SELECT 1",
        sql_result={"rows": []},
        latency={"total": 1.25},
        success=True,
    )

    row = state_to_batch_row(
        state=state,
        question_id="E001",
        expected_unit="count",
        timestamp="2026-05-22T00:00:00+00:00",
    )

    assert row["question_id"] == "E001"
    assert row["sql_execution_success"] is True
    assert row["latency_total"] == 1.25
    assert row["timestamp"] == "2026-05-22T00:00:00+00:00"


def test_write_batch_results_writes_expected_columns(tmp_path):
    output_path = tmp_path / "batch_eval_energy.csv"
    rows = [
        {
            "question_id": "E001",
            "question": "test",
            "generated_sql": "SELECT 1",
            "repaired_sql": "",
            "success": True,
            "error_message": "",
            "final_answer": "ok",
            "expected_unit": "count",
            "latency_total": 1.0,
            "sql_execution_success": True,
            "timestamp": "2026-05-22T00:00:00+00:00",
        }
    ]

    write_batch_results(rows, output_path)

    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        loaded_rows = list(reader)

    assert reader.fieldnames == list(BATCH_EVAL_COLUMNS)
    assert loaded_rows[0]["question_id"] == "E001"
    assert loaded_rows[0]["generated_sql"] == "SELECT 1"
