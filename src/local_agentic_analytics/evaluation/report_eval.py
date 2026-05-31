"""Ground-truth evaluation for generated reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from local_agentic_analytics.core.config import PROJECT_ROOT


DEFAULT_REPORT_LOG_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "report_generation_log.json"
)
DEFAULT_REPORT_GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "references" / "gold_reports" / "energy_report_ground_truth.json"
)
DEFAULT_REPORT_EVAL_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "report_eval.json"
)
DEFAULT_LATEX_PATH = PROJECT_ROOT / "reports" / "latex" / "energy_analysis_report.tex"


def load_report_metadata(path: str | Path = DEFAULT_REPORT_LOG_PATH) -> dict[str, Any]:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Report metadata file not found: {metadata_path}")
    return _load_json_object(metadata_path)


def load_report_ground_truth(
    path: str | Path = DEFAULT_REPORT_GROUND_TRUTH_PATH,
) -> dict[str, Any]:
    ground_truth_path = Path(path)
    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"Report ground truth file not found: {ground_truth_path}")
    return _load_json_object(ground_truth_path)


def evaluate_report_artifacts(
    ground_truth_path: str | Path = DEFAULT_REPORT_GROUND_TRUTH_PATH,
    metadata_path: str | Path = DEFAULT_REPORT_LOG_PATH,
    latex_path: str | Path = DEFAULT_LATEX_PATH,
) -> dict[str, Any]:
    ground_truth = load_report_ground_truth(ground_truth_path)
    metadata = load_report_metadata(metadata_path)
    latex_file = Path(latex_path)
    latex_exists = latex_file.is_file()
    latex_text = latex_file.read_text(encoding="utf-8") if latex_exists else ""

    section_result = _evaluate_sections(
        latex_text=latex_text,
        required_sections=_as_string_list(ground_truth.get("required_sections", [])),
    )
    chart_result = _evaluate_charts(
        metadata=metadata,
        required_charts=_as_string_list(ground_truth.get("required_charts", [])),
    )
    pdf_compile_success = bool(metadata.get("pdf_success"))
    latex_exists_score = 1.0 if latex_exists else 0.0

    final_score = round(
        (
            section_result["section_completeness"]
            + chart_result["chart_validity"]
            + (1.0 if pdf_compile_success else 0.0)
            + latex_exists_score
        )
        / 4.0,
        6,
    )

    return {
        "report_id": str(ground_truth.get("report_id", "")),
        "section_completeness": section_result["section_completeness"],
        "required_sections": section_result["required_sections"],
        "found_sections": section_result["found_sections"],
        "missing_sections": section_result["missing_sections"],
        "chart_validity": chart_result["chart_validity"],
        "required_charts": chart_result["required_charts"],
        "missing_charts": chart_result["missing_charts"],
        "missing_chart_files": chart_result["missing_chart_files"],
        "pdf_compile_success": pdf_compile_success,
        "latex_exists": latex_exists,
        "required_chart_count": chart_result["required_chart_count"],
        "existing_chart_count": chart_result["existing_chart_count"],
        "final_score": final_score,
        "numeric_accuracy": None,
        "numeric_accuracy_note": (
            "not_implemented: numeric facts are defined in ground truth, but "
            "automatic LaTeX numeric parsing is not implemented in this stage."
        ),
        "unit_rules": ground_truth.get("unit_rules", {}),
        "numeric_facts": ground_truth.get("numeric_facts", []),
        "metadata_path": str(Path(metadata_path)),
        "latex_path": str(latex_file),
        "ground_truth_path": str(Path(ground_truth_path)),
    }


def evaluate_report_metadata(
    metadata: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Backward-compatible metadata-only chart/report evaluation."""
    required_chart_ids = _as_string_list(
        ground_truth.get("required_chart_ids", ground_truth.get("required_charts", []))
    )
    generated_chart_ids = [
        str(chart.get("chart_id", ""))
        for chart in metadata.get("charts", [])
        if isinstance(chart, dict)
    ]
    generated_chart_set = set(generated_chart_ids)
    missing_chart_ids = [
        chart_id for chart_id in required_chart_ids if chart_id not in generated_chart_set
    ]

    required_count = len(required_chart_ids)
    chart_coverage = (
        (required_count - len(missing_chart_ids)) / required_count
        if required_count
        else 1.0
    )

    min_chart_count = int(ground_truth.get("min_chart_count", 0))
    min_successful_insights = int(ground_truth.get("min_successful_insights", 0))
    require_tex_success = bool(ground_truth.get("require_tex_success", True))
    pdf_optional = bool(ground_truth.get("pdf_optional", True))

    chart_count = int(metadata.get("chart_count", 0) or 0)
    insight_success_count = int(metadata.get("insight_success_count", 0) or 0)
    tex_success = bool(metadata.get("tex_success"))
    pdf_success = bool(metadata.get("pdf_success"))

    chart_count_pass = chart_count >= min_chart_count
    insight_pass = insight_success_count >= min_successful_insights
    tex_pass = tex_success if require_tex_success else True
    pdf_pass = pdf_success or pdf_optional

    passed = (
        chart_coverage == 1.0
        and chart_count_pass
        and insight_pass
        and tex_pass
        and pdf_pass
    )

    return {
        "report_id": str(ground_truth.get("report_id", "")),
        "passed": passed,
        "chart_coverage": chart_coverage,
        "missing_chart_ids": missing_chart_ids,
        "chart_count": chart_count,
        "min_chart_count": min_chart_count,
        "chart_count_pass": chart_count_pass,
        "insight_success_count": insight_success_count,
        "min_successful_insights": min_successful_insights,
        "insight_pass": insight_pass,
        "tex_success": tex_success,
        "tex_pass": tex_pass,
        "pdf_success": pdf_success,
        "pdf_optional": pdf_optional,
        "pdf_pass": pdf_pass,
    }


def write_report_eval_result(
    result: dict[str, Any],
    output_path: str | Path = DEFAULT_REPORT_EVAL_OUTPUT_PATH,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _evaluate_sections(
    latex_text: str,
    required_sections: list[str],
) -> dict[str, Any]:
    if not required_sections:
        return {
            "section_completeness": 1.0,
            "required_sections": [],
            "found_sections": [],
            "missing_sections": [],
        }

    found_sections = [
        section for section in required_sections if _latex_contains_section(latex_text, section)
    ]
    missing_sections = [
        section for section in required_sections if section not in found_sections
    ]
    return {
        "section_completeness": round(len(found_sections) / len(required_sections), 6),
        "required_sections": required_sections,
        "found_sections": found_sections,
        "missing_sections": missing_sections,
    }


def _evaluate_charts(
    metadata: dict[str, Any],
    required_charts: list[str],
) -> dict[str, Any]:
    chart_by_id = {
        str(chart.get("chart_id", "")): chart
        for chart in metadata.get("charts", [])
        if isinstance(chart, dict)
    }
    missing_charts = [
        chart_id for chart_id in required_charts if chart_id not in chart_by_id
    ]
    missing_chart_files = []
    existing_chart_count = 0

    for chart_id in required_charts:
        chart = chart_by_id.get(chart_id)
        if chart is None:
            continue

        chart_path = Path(str(chart.get("path", "")))
        if chart_path.is_file():
            existing_chart_count += 1
        else:
            missing_chart_files.append(chart_id)

    required_chart_count = len(required_charts)
    chart_validity = (
        existing_chart_count / required_chart_count
        if required_chart_count
        else 1.0
    )

    return {
        "chart_validity": round(chart_validity, 6),
        "required_charts": required_charts,
        "missing_charts": missing_charts,
        "missing_chart_files": missing_chart_files,
        "required_chart_count": required_chart_count,
        "existing_chart_count": existing_chart_count,
    }


def _latex_contains_section(latex_text: str, section: str) -> bool:
    if not latex_text:
        return False
    if section.lower() == "abstract":
        return bool(re.search(r"\\begin\{abstract\}", latex_text, flags=re.IGNORECASE))

    escaped_section = re.escape(section)
    return bool(
        re.search(
            rf"\\(?:section|subsection)\*?\{{\s*{escaped_section}\s*\}}",
            latex_text,
            flags=re.IGNORECASE,
        )
    )


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data
