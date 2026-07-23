import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.model_benchmark import (
    SUITE_ABLATION,
    SUITE_SQL_GOLD_V2,
    TELEMETRY_COLUMNS,
    TelemetryRecorder,
    attach_telemetry,
    hash_gold_dataset,
    is_tag_available,
    list_model_keys,
    load_model_benchmark_config,
    mean_std,
    percentile,
    render_summary_markdown,
    suite_columns,
    summarize_model_benchmark,
    summarize_run_rows,
    unit_mentioned,
    validate_locked_variables,
    write_benchmark_rows,
    write_manifest,
    write_summary_csv,
)
from local_agentic_analytics.evaluation.ablation_eval import ABLATION_EVAL_COLUMNS
from local_agentic_analytics.evaluation.sql_gold_eval import SQL_GOLD_EVAL_COLUMNS


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def test_registered_model_configs_load_and_are_apple_to_apple():
    keys = list_model_keys()
    assert "gemma2_2b" in keys
    assert "qwen2.5_1.5b" in keys
    assert "qwen2.5_3b" in keys

    for key in keys:
        config = load_model_benchmark_config(key)
        assert config.key == key
        assert config.ollama_tag
        assert validate_locked_variables(config) == []


def test_load_model_config_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_model_benchmark_config("does_not_exist")


def test_load_model_config_rejects_missing_field(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text(
        "model:\n  key: broken\n  ollama_tag: x:y\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required field"):
        load_model_benchmark_config("broken", config_dir=tmp_path)


def test_load_model_config_rejects_key_mismatch(tmp_path):
    path = tmp_path / "alpha.yaml"
    path.write_text(
        (
            "model:\n"
            "  key: beta\n"
            "  ollama_tag: x:y\n"
            "  param_count: 1B\n"
            "  quantization: Q4_K_M\n"
            "  context_window: 2048\n"
            "  temperature: 0.0\n"
            "  prompt_template: default\n"
            "  source: base\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match file name"):
        load_model_benchmark_config("alpha", config_dir=tmp_path)


def test_validate_locked_variables_flags_deviations():
    config = load_model_benchmark_config("gemma2_2b")

    wrong_context = dataclasses.replace(config, context_window=4096)
    assert any(
        "context_window" in problem
        for problem in validate_locked_variables(wrong_context)
    )

    wrong_temperature = dataclasses.replace(config, temperature=0.7)
    assert any(
        "temperature" in problem
        for problem in validate_locked_variables(wrong_temperature)
    )

    wrong_prompt = dataclasses.replace(config, prompt_template="custom")
    assert any(
        "prompt_template" in problem
        for problem in validate_locked_variables(wrong_prompt)
    )

    finetuned_without_hash = dataclasses.replace(
        config, source="finetuned", modelfile_sha256=None
    )
    assert any(
        "modelfile_sha256" in problem
        for problem in validate_locked_variables(finetuned_without_hash)
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class _FakeWorkflow:
    def __init__(self, state=None, error=None):
        self.state = state
        self.error = error

    def run(self, user_query):
        if self.error is not None:
            raise self.error
        return self.state


def _fake_state(latency_total=2.5, final_answer="Rata-rata 1.2 kW", success=True):
    return SimpleNamespace(
        latency={"total": latency_total, "sql_generation": 1.0},
        tool_calls=[
            {
                "component": "ollama",
                "metadata": {"eval_count": 50, "eval_duration": 5.0},
            }
        ],
        final_answer=final_answer,
        success=success,
    )


def test_telemetry_recorder_captures_latency_and_throughput():
    recorder = TelemetryRecorder()
    wrapped = recorder.wrap(_FakeWorkflow(state=_fake_state()))

    state = wrapped.run("Berapa rata-rata daya?")

    assert state.success is True
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record["question"] == "Berapa rata-rata daya?"
    assert record["latency_total"] == 2.5
    assert record["tokens_per_second"] == pytest.approx(10.0)
    assert record["final_answer"] == "Rata-rata 1.2 kW"


def test_telemetry_recorder_records_failure_and_reraises():
    recorder = TelemetryRecorder()
    wrapped = recorder.wrap(_FakeWorkflow(error=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        wrapped.run("Q1")

    assert len(recorder.records) == 1
    assert recorder.records[0]["success"] is False
    assert recorder.records[0]["latency_total"] is None


def test_unit_mentioned_token_boundaries():
    assert unit_mentioned("kW", "Konsumsi rata-rata 1.2 kW hari itu") is True
    assert unit_mentioned("kW", "Total energi 5 kWh") is False
    assert unit_mentioned("kwh", "Total energi 5 kWh") is True
    assert unit_mentioned("", "jawaban") is None
    assert unit_mentioned("kW", "") is None


def test_attach_telemetry_aligns_by_question_order():
    questions = [
        {"question": "Q1", "expected_unit": "kW"},
        {"question": "Q2", "expected_unit": "V"},
    ]
    rows = [
        {"question": "Q1", "agent_success": True},
        {"question": "Q2", "agent_success": False},
    ]
    records = [
        {
            "question": "Q1",
            "latency_total": 3.0,
            "tokens_per_second": 8.0,
            "final_answer": "1.2 kW",
            "success": True,
        },
        {
            "question": "Q2",
            "latency_total": 4.0,
            "tokens_per_second": None,
            "final_answer": "",
            "success": False,
        },
    ]

    merged = attach_telemetry(rows, records, questions)

    assert merged[0]["latency_total"] == 3.0
    assert merged[0]["unit_correct"] is True
    assert merged[1]["latency_total"] == 4.0
    assert merged[1]["tokens_per_second"] == ""
    assert merged[1]["unit_correct"] == ""


def test_attach_telemetry_skips_rows_without_matching_record():
    questions = [{"question": "Q1", "expected_unit": "kW"}]
    rows = [
        {"question": "Q0-load-error"},
        {"question": "Q1"},
    ]
    records = [
        {
            "question": "Q1",
            "latency_total": 1.5,
            "tokens_per_second": 5.0,
            "final_answer": "2 kW",
            "success": True,
        }
    ]

    merged = attach_telemetry(rows, records, questions)

    assert merged[0]["latency_total"] == ""
    assert merged[1]["latency_total"] == 1.5
    assert merged[1]["unit_correct"] is True


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def test_percentile_interpolates():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile(values, 95) == pytest.approx(3.85)
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError):
        percentile(values, 101)


def test_mean_std():
    assert mean_std([]) == (None, None)
    mean, std = mean_std([2.0])
    assert mean == 2.0
    assert std == 0.0
    mean, std = mean_std([1.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.4142, abs=1e-3)


def test_is_tag_available_normalizes_latest():
    assert is_tag_available("gemma2:2b", ["gemma2:2b", "qwen2.5:3b"]) is True
    assert is_tag_available("sqlcoder", ["sqlcoder:latest"]) is True
    assert is_tag_available("gemma2:2b", ["gemma2:9b"]) is False


# ---------------------------------------------------------------------------
# Dataset hashing
# ---------------------------------------------------------------------------


def test_hash_gold_dataset(tmp_path):
    sql_path = tmp_path / "G1.sql"
    sql_path.write_text("SELECT 1;", encoding="utf-8")
    questions = [
        {"id": "G1", "question": "Q", "gold_sql_file": str(sql_path)},
        {"id": "G2", "question": "Q2", "gold_sql_file": str(tmp_path / "nope.sql")},
    ]
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps(questions), encoding="utf-8")

    info = hash_gold_dataset(questions_path, questions)

    assert info["n_questions"] == 2
    assert info["n_gold_sql_files"] == 2
    assert len(info["questions_sha256"]) == 64
    assert len(info["gold_sql_combined_sha256"]) == 64
    assert info["missing_gold_sql_files"] == [str(tmp_path / "nope.sql")]


# ---------------------------------------------------------------------------
# Suite columns, run summarization, manifest, cross-model summary
# ---------------------------------------------------------------------------


def test_suite_columns_extend_original_columns():
    assert suite_columns(SUITE_SQL_GOLD_V2) == (
        tuple(SQL_GOLD_EVAL_COLUMNS) + TELEMETRY_COLUMNS
    )
    assert suite_columns(SUITE_ABLATION) == (
        tuple(ABLATION_EVAL_COLUMNS) + TELEMETRY_COLUMNS
    )


def test_summarize_run_rows_sql_gold():
    rows = [
        {
            "agent_success": "True",
            "numeric_match": "True",
            "unit_correct": "True",
            "latency_total": "2.0",
            "tokens_per_second": "10.0",
        },
        {
            "agent_success": "True",
            "numeric_match": "False",
            "unit_correct": "",
            "latency_total": "4.0",
            "tokens_per_second": "8.0",
        },
        {
            "agent_success": "False",
            "numeric_match": "",
            "unit_correct": "False",
            "latency_total": "",
            "tokens_per_second": "",
        },
    ]

    metrics = summarize_run_rows(rows, SUITE_SQL_GOLD_V2)["-"]

    assert metrics["n_questions"] == 3
    assert metrics["execution_success_rate"] == pytest.approx(2 / 3)
    assert metrics["numeric_match_compared_rate"] == pytest.approx(1 / 2)
    assert metrics["numeric_match_total_rate"] == pytest.approx(1 / 3)
    assert metrics["unit_correct_rate"] == pytest.approx(1 / 2)
    assert metrics["latency_p50"] == pytest.approx(3.0)
    assert metrics["tokens_per_second_mean"] == pytest.approx(9.0)


def test_summarize_run_rows_ablation_groups_by_config():
    rows = [
        {
            "config": "A_raw_llm",
            "execution_success": "False",
            "numeric_match": "",
            "unit_correct": "",
            "latency_total": "1.0",
            "tokens_per_second": "",
        },
        {
            "config": "D_full",
            "execution_success": "True",
            "numeric_match": "True",
            "unit_correct": "",
            "latency_total": "2.0",
            "tokens_per_second": "",
        },
    ]

    metrics = summarize_run_rows(rows, SUITE_ABLATION)

    assert set(metrics) == {"A_raw_llm", "D_full"}
    assert metrics["A_raw_llm"]["execution_success_rate"] == 0.0
    assert metrics["D_full"]["execution_success_rate"] == 1.0


def test_write_manifest_merges_suites(tmp_path):
    config = load_model_benchmark_config("gemma2_2b")
    environment = {"commit": "abc", "updated_at": "t"}
    dataset = {"questions_sha256": "x"}

    write_manifest(
        tmp_path, config, "sql_gold_v2", {"runs": [1]}, dataset, environment
    )
    path = write_manifest(
        tmp_path, config, "finance", {"runs": [2]}, dataset, environment
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["model_key"] == "gemma2_2b"
    assert set(manifest["suites"]) == {"sql_gold_v2", "finance"}
    assert manifest["suites"]["sql_gold_v2"]["runs"] == [1]
    assert manifest["model_config"]["ollama_tag"] == "gemma2:2b"


def _write_fake_run(model_dir: Path, run_index: int, latency: float):
    columns = suite_columns(SUITE_SQL_GOLD_V2)
    rows = [
        {
            "question_id": "E101",
            "question": "Q1",
            "agent_success": True,
            "gold_success": True,
            "numeric_match": True,
            "latency_total": latency,
            "tokens_per_second": 10.0,
            "unit_correct": True,
        },
        {
            "question_id": "E102",
            "question": "Q2",
            "agent_success": False,
            "gold_success": True,
            "numeric_match": "",
            "latency_total": latency + 1.0,
            "tokens_per_second": 9.0,
            "unit_correct": "",
        },
    ]
    write_benchmark_rows(
        rows, columns, model_dir / f"sql_gold_v2_run{run_index}.csv"
    )
    return {
        "run_index": run_index,
        "csv": f"sql_gold_v2_run{run_index}.csv",
        "resources": {"peak_rss_mb": 900.0 + run_index, "model_vram_mb": 1500.0},
    }


def test_summarize_model_benchmark_end_to_end(tmp_path):
    config = load_model_benchmark_config("gemma2_2b")
    model_dir = tmp_path / "gemma2_2b"
    model_dir.mkdir()

    runs = [
        _write_fake_run(model_dir, 1, latency=2.0),
        _write_fake_run(model_dir, 2, latency=4.0),
    ]
    write_manifest(
        model_dir,
        config,
        "sql_gold_v2",
        {"repeats": 2, "runs": runs},
        {"questions_sha256": "x"},
        {"commit": "abc"},
    )

    summary_rows = summarize_model_benchmark(tmp_path)

    assert len(summary_rows) == 1
    row = summary_rows[0]
    assert row["model"] == "gemma2_2b"
    assert row["suite"] == "sql_gold_v2"
    assert row["runs"] == 2
    assert row["n_questions"] == 2
    assert row["execution_success_mean"] == pytest.approx(0.5)
    assert row["execution_success_std"] == pytest.approx(0.0)
    assert row["numeric_match_compared_mean"] == pytest.approx(1.0)
    assert row["peak_rss_mb_max"] == pytest.approx(902.0)
    assert row["model_vram_mb_max"] == pytest.approx(1500.0)

    csv_path = write_summary_csv(summary_rows, tmp_path / "summary.csv")
    assert csv_path.is_file()
    markdown = render_summary_markdown(summary_rows)
    assert "gemma2_2b" in markdown
    assert "50.0%" in markdown
