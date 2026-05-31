from pathlib import Path
import json
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.report_eval import (
    evaluate_report_artifacts,
    evaluate_report_metadata,
    load_report_ground_truth,
    load_report_metadata,
    write_report_eval_result,
)


def test_evaluate_report_metadata_passes_when_requirements_are_met():
    metadata = {
        "chart_count": 2,
        "insight_success_count": 1,
        "tex_success": True,
        "pdf_success": False,
        "charts": [
            {"chart_id": "daily_active_power_trend"},
            {"chart_id": "power_distribution"},
        ],
    }
    ground_truth = {
        "report_id": "test",
        "required_chart_ids": [
            "daily_active_power_trend",
            "power_distribution",
        ],
        "min_chart_count": 2,
        "min_successful_insights": 1,
        "require_tex_success": True,
        "pdf_optional": True,
    }

    result = evaluate_report_metadata(metadata, ground_truth)

    assert result["passed"] is True
    assert result["chart_coverage"] == 1.0
    assert result["pdf_pass"] is True


def test_evaluate_report_metadata_reports_missing_charts():
    result = evaluate_report_metadata(
        metadata={
            "chart_count": 1,
            "insight_success_count": 0,
            "tex_success": True,
            "pdf_success": False,
            "charts": [{"chart_id": "daily_active_power_trend"}],
        },
        ground_truth={
            "report_id": "test",
            "required_chart_ids": [
                "daily_active_power_trend",
                "power_distribution",
            ],
            "min_chart_count": 2,
            "min_successful_insights": 1,
            "require_tex_success": True,
            "pdf_optional": False,
        },
    )

    assert result["passed"] is False
    assert result["missing_chart_ids"] == ["power_distribution"]
    assert result["insight_pass"] is False
    assert result["pdf_pass"] is False


def test_report_eval_loaders_and_writer_use_json_files(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    ground_truth_path = tmp_path / "ground_truth.json"
    output_path = tmp_path / "report_eval.json"
    metadata_path.write_text(json.dumps({"charts": []}), encoding="utf-8")
    ground_truth_path.write_text(json.dumps({"required_chart_ids": []}), encoding="utf-8")

    metadata = load_report_metadata(metadata_path)
    ground_truth = load_report_ground_truth(ground_truth_path)
    result_path = write_report_eval_result(
        {"passed": True},
        output_path=output_path,
    )

    assert metadata == {"charts": []}
    assert ground_truth == {"required_chart_ids": []}
    assert result_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"passed": True}


def test_report_eval_loader_rejects_non_object_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain an object"):
        load_report_metadata(path)


def test_evaluate_report_artifacts_scores_sections_charts_and_pdf(tmp_path):
    chart_dir = tmp_path / "figures"
    chart_dir.mkdir()
    required_charts = [
        "daily_active_power_trend",
        "hourly_consumption_pattern",
    ]
    for chart_id in required_charts:
        (chart_dir / f"{chart_id}.png").write_bytes(b"fake png")

    ground_truth_path = tmp_path / "ground_truth.json"
    metadata_path = tmp_path / "metadata.json"
    latex_path = tmp_path / "report.tex"
    ground_truth_path.write_text(
        json.dumps(
            {
                "report_id": "energy_analysis_report",
                "required_sections": [
                    "Abstract",
                    "Introduction",
                    "Methodology",
                    "Detailed Analysis",
                    "Synthesis and Implications",
                    "Conclusion",
                ],
                "required_charts": required_charts,
                "unit_rules": {"Voltage": "V"},
                "numeric_facts": [
                    {
                        "id": "N001",
                        "description": "average active power on 2006-12-16",
                        "value": 3.0534747475,
                        "unit": "kW",
                        "tolerance": 0.00001,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "pdf_success": True,
                "charts": [
                    {
                        "chart_id": chart_id,
                        "path": str(chart_dir / f"{chart_id}.png"),
                    }
                    for chart_id in required_charts
                ],
            }
        ),
        encoding="utf-8",
    )
    latex_path.write_text(
        r"""
\begin{abstract}
Ringkasan.
\end{abstract}
\section{Introduction}
\section{Methodology}
\section{Detailed Analysis}
\section{Synthesis and Implications}
\section{Conclusion}
""",
        encoding="utf-8",
    )

    result = evaluate_report_artifacts(
        ground_truth_path=ground_truth_path,
        metadata_path=metadata_path,
        latex_path=latex_path,
    )

    assert result["section_completeness"] == 1.0
    assert result["chart_validity"] == 1.0
    assert result["pdf_compile_success"] is True
    assert result["latex_exists"] is True
    assert result["required_chart_count"] == 2
    assert result["existing_chart_count"] == 2
    assert result["final_score"] == 1.0
    assert result["numeric_accuracy"] is None
    assert "not_implemented" in result["numeric_accuracy_note"]


def test_evaluate_report_artifacts_reports_missing_sections_and_chart_files(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.json"
    metadata_path = tmp_path / "metadata.json"
    latex_path = tmp_path / "report.tex"
    ground_truth_path.write_text(
        json.dumps(
            {
                "required_sections": ["Abstract", "Introduction", "Conclusion"],
                "required_charts": ["daily_active_power_trend", "power_distribution"],
            }
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "pdf_success": False,
                "charts": [
                    {
                        "chart_id": "daily_active_power_trend",
                        "path": str(tmp_path / "missing.png"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    latex_path.write_text(r"\section{Introduction}", encoding="utf-8")

    result = evaluate_report_artifacts(
        ground_truth_path=ground_truth_path,
        metadata_path=metadata_path,
        latex_path=latex_path,
    )

    assert result["section_completeness"] == pytest.approx(1 / 3)
    assert result["missing_sections"] == ["Abstract", "Conclusion"]
    assert result["chart_validity"] == 0.0
    assert result["missing_charts"] == ["power_distribution"]
    assert result["missing_chart_files"] == ["daily_active_power_trend"]
    assert result["pdf_compile_success"] is False
    assert result["final_score"] < 1.0
