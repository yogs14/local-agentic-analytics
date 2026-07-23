from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.insight_model_benchmark import (
    aggregate_benchmark_samples,
    run_insight_model_benchmark,
    tokens_per_second,
    write_benchmark_result,
)


class _FakeOllamaTool:
    model = "gemma2-energy-insight:v3"

    def __init__(self):
        self._metrics = {"eval_count": 120, "eval_duration": 3.0, "total_duration": 4.0}

    def get_last_metrics(self):
        return dict(self._metrics)


class _FakeAgent:
    def __init__(self, narrative: str):
        self.narrative = narrative
        self.ollama_tool = _FakeOllamaTool()

    def generate_insight(self, context: dict) -> str:
        return self.narrative


def _voltage_context() -> dict:
    return {
        "chart_id": "voltage_distribution",
        "chart_title": "Distribusi Voltage",
        "chart_path": "p.png",
        "stats": {
            "min_voltage_v": 223.2,
            "max_voltage_v": 254.15,
            "avg_voltage_v": 240.8399,
            "stddev_voltage_v": 3.24,
            "unit": "Volt",
        },
    }


def test_tokens_per_second_handles_missing_and_zero():
    assert tokens_per_second({"eval_count": 100, "eval_duration": 2.0}) == 50.0
    assert tokens_per_second({"eval_count": 100, "eval_duration": 0}) is None
    assert tokens_per_second({}) is None


def test_run_benchmark_aggregates_quality_and_efficiency():
    agent = _FakeAgent(
        "Rata-rata tegangan 240,8399 V dengan simpangan baku 3,24 V pada rentang "
        "223,2 V hingga 254,15 V, konsisten dengan profil beban residensial."
    )

    result = run_insight_model_benchmark(agent, [_voltage_context()], repeats=3)

    assert result["repeats"] == 3
    assert result["quality"]["samples"] == 3
    assert result["quality"]["acceptance_rate"] == 1.0
    assert result["quality"]["number_groundedness"] == 1.0
    assert result["quality"]["markdown_clean_rate"] == 1.0
    assert result["efficiency"]["tokens_per_second"]["mean"] == 40.0
    assert result["efficiency"]["latency_seconds"]["mean"] == 4.0
    assert result["model"] == "gemma2-energy-insight:v3"


def test_run_benchmark_rejects_invalid_repeats():
    with pytest.raises(ValueError):
        run_insight_model_benchmark(_FakeAgent("x"), [_voltage_context()], repeats=0)


def test_aggregate_marks_ungrounded_sample(tmp_path):
    samples = [
        {
            "chart_id": "voltage_distribution",
            "repeat_index": 0,
            "latency_seconds": 4.0,
            "tokens_per_second": 40.0,
            "eval_count": 120,
            "quality": {
                "ok": False,
                "require_unit": True,
                "total_numbers": 5,
                "ungrounded_numbers": 1,
                "has_unit": True,
                "enough_concept_terms": True,
                "concept_terms": 4,
                "markdown_clean": True,
                "sentence_count": 4,
            },
        }
    ]

    result = aggregate_benchmark_samples(samples, repeats=1)

    assert result["quality"]["number_groundedness"] == 0.8
    assert result["quality"]["fully_grounded_rate"] == 0.0

    paths = write_benchmark_result(
        result,
        output_path=tmp_path / "bench.json",
        csv_path=tmp_path / "bench.csv",
    )
    assert paths["json"].is_file()
    assert "chart_id" in paths["csv"].read_text(encoding="utf-8")
