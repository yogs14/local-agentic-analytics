from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.insight_quality_eval import (
    evaluate_insight_quality,
    evaluate_insight_record,
    write_insight_quality_result,
    _extract_latency_metrics,
)


def _voltage_record(insight: str) -> dict:
    return {
        "chart_id": "voltage_distribution",
        "stats": {
            "record_count": 2_049_280,
            "min_voltage_v": 223.2,
            "max_voltage_v": 254.15,
            "avg_voltage_v": 240.8399,
            "stddev_voltage_v": 3.24,
            "unit": "Volt",
        },
        "insight": insight,
        "success": True,
    }


def test_grounded_voltage_narrative_passes_all_checks():
    record = _voltage_record(
        "Distribusi tegangan listrik rumah tangga menunjukkan rata-rata sebesar "
        "240,8399 V dengan simpangan baku 3,24 V. Rentang tegangan dari 223,2 V "
        "hingga 254,15 V konsisten dengan profil beban residensial yang stabil."
    )

    result = evaluate_insight_record(record)

    assert result["evaluated"] is True
    assert result["ungrounded_numbers"] == 0
    assert result["number_groundedness"] == 1.0
    assert result["has_unit"] is True
    assert result["markdown_clean"] is True
    assert result["ok"] is True


def test_hallucinated_number_is_flagged_as_ungrounded():
    record = _voltage_record(
        "Tegangan rata-rata 240,8399 V, namun terjadi lonjakan hingga 999,9 V "
        "yang tidak wajar pada sistem distribusi rumah tangga."
    )

    result = evaluate_insight_record(record)

    assert result["ungrounded_numbers"] >= 1
    assert result["number_groundedness"] < 1.0
    assert result["ok"] is False


def test_markdown_leakage_is_detected():
    record = _voltage_record(
        "## Distribusi Tegangan\n\nRata-rata **240,8399 V** dengan simpangan baku "
        "3,24 V pada rentang 223,2 V hingga 254,15 V."
    )

    result = evaluate_insight_record(record)

    assert result["markdown_clean"] is False
    assert "heading" in result["markdown_artifacts"]
    assert "bold_or_italic" in result["markdown_artifacts"]


def test_correlation_chart_does_not_require_a_unit():
    record = {
        "chart_id": "correlation_heatmap",
        "stats": {
            "pair_count": 21,
            "strongest_positive": {
                "pair": "Global_active_power vs Global_intensity",
                "correlation": 0.9989,
            },
            "strongest_negative": {
                "pair": "Voltage vs Global_intensity",
                "correlation": -0.4114,
            },
            "unit_note": "tanpa satuan",
        },
        "insight": (
            "Korelasi Pearson terkuat sebesar 0,9989 antara daya aktif dan "
            "intensitas arus, sedangkan hubungan negatif -0,4114 muncul pada "
            "tegangan, konsisten dengan profil beban residensial."
        ),
        "success": True,
    }

    result = evaluate_insight_record(record)

    assert result["require_unit"] is False
    assert result["ungrounded_numbers"] == 0
    assert result["ok"] is True


def test_failed_generation_is_marked_not_evaluated():
    record = {
        "chart_id": "voltage_distribution",
        "stats": {"avg_voltage_v": 240.8, "unit": "Volt"},
        "insight": "Insight otomatis tidak dapat dibuat untuk grafik ini.",
        "success": False,
    }

    result = evaluate_insight_record(record)

    assert result["evaluated"] is False
    assert result["ok"] is False


def test_aggregate_and_write(tmp_path):
    metadata = {
        "model": "gemma2-energy-insight:v3",
        "engine": "custom",
        "insights": [
            _voltage_record(
                "Rata-rata tegangan 240,8399 V dengan simpangan baku 3,24 V pada "
                "rentang 223,2 V hingga 254,15 V, konsisten dengan beban residensial."
            ),
        ],
    }

    result = evaluate_insight_quality(metadata)

    assert result["insights_evaluated"] == 1
    assert result["number_groundedness"] == 1.0
    assert result["unit_presence_rate"] == 1.0

    paths = write_insight_quality_result(
        result,
        output_path=tmp_path / "insight_quality_eval.json",
        csv_path=tmp_path / "insight_quality_eval.csv",
    )

    assert paths["json"].is_file()
    assert paths["csv"].is_file()
    assert "chart_id" in paths["csv"].read_text(encoding="utf-8")


def test_latency_metrics_extracted_from_timestamps():
    metadata = {
        "model": "gemma2-energy-insight:v3",
        "engine": "custom",
        "timestamp_start": "2026-06-26T13:25:54.378279+00:00",
        "timestamp_end": "2026-06-26T13:26:46.179265+00:00",
        "latency": {},
        "tool_calls": [],
        "insights": [
            _voltage_record(
                "Rata-rata tegangan 240,8399 V dengan simpangan baku 3,24 V pada "
                "rentang 223,2 V hingga 254,15 V, konsisten dengan beban residensial."
            ),
        ],
    }

    result = evaluate_insight_quality(metadata)

    assert result["total_latency_seconds"] > 0.0
    assert result["total_latency_label"].endswith("s")
    assert "total_latency_seconds" in result
    assert "step_latencies" in result
    assert result["composite_insight_score"] is not None


def test_latency_handles_missing_timestamps():
    metadata = {
        "model": "gemma2-energy-insight:v3",
        "engine": "custom",
        "timestamp_start": "",
        "timestamp_end": "",
        "latency": {},
        "tool_calls": [],
        "insights": [],
    }

    result = evaluate_insight_quality(metadata)

    assert result["total_latency_seconds"] == 0.0
    assert result["total_latency_label"] == ""


def test_latency_extracts_step_latencies():
    metadata = {
        "timestamp_start": "",
        "timestamp_end": "",
        "latency": {
            "generate_charts": 1.234,
            "generate_insights": 45.678,
            "render_latex": 0.567,
        },
        "tool_calls": [],
    }

    lat_info = _extract_latency_metrics(metadata)

    assert "generate_charts" in lat_info["step_latencies"]
    assert lat_info["step_latencies"]["generate_charts"] == 1.234
    assert lat_info["step_latencies"]["generate_insights"] == 45.678
    assert lat_info["step_latencies"]["render_latex"] == 0.567


def test_latency_sums_tool_call_durations():
    metadata = {
        "timestamp_start": "",
        "timestamp_end": "",
        "latency": {},
        "tool_calls": [
            {"tool": "ollama.generate", "latency_seconds": 3.5, "status": "success"},
            {"tool": "ollama.generate", "latency_seconds": 4.2, "status": "success"},
            {"tool": "report.render_latex", "latency_seconds": 0.3, "status": "success"},
        ],
    }

    lat_info = _extract_latency_metrics(metadata)

    assert lat_info["tool_call_latency_total"] == 8.0
    assert lat_info["tool_call_count"] == 3


def test_composite_insight_score_is_average_of_four_components():
    metadata = {
        "engine": "custom",
        "insights": [
            _voltage_record(
                "Rata-rata tegangan 240,8399 V dengan simpangan baku 3,24 V pada "
                "rentang 223,2 V hingga 254,15 V, konsisten dengan beban residensial."
            ),
        ],
    }

    result = evaluate_insight_quality(metadata)

    assert 0.0 <= result["composite_insight_score"] <= 1.0
    assert result["composite_insight_score"] == 1.0


def test_composite_score_penalizes_ungrounded():
    metadata = {
        "engine": "custom",
        "insights": [
            _voltage_record(
                "Tegangan rata-rata 999,9 V yang tidak wajar pada sistem distribusi rumah tangga."
            ),
        ],
    }

    result = evaluate_insight_quality(metadata)

    assert result["composite_insight_score"] < 1.0
