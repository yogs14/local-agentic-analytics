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
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_PDF_DIR = PROJECT_ROOT / "reports" / "pdf"


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
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    pdf_dir: str | Path = DEFAULT_PDF_DIR,
) -> dict[str, Any]:
    ground_truth = load_report_ground_truth(ground_truth_path)
    metadata = load_report_metadata(metadata_path)
    latex_file = Path(latex_path)
    figures_path = Path(figures_dir)
    pdf_path = Path(pdf_dir)
    latex_exists = latex_file.is_file()
    latex_text = latex_file.read_text(encoding="utf-8") if latex_exists else ""

    section_result = _evaluate_sections(
        latex_text=latex_text,
        required_sections=_as_string_list(ground_truth.get("required_sections", [])),
    )
    chart_result = _evaluate_charts(
        metadata=metadata,
        required_charts=_as_string_list(ground_truth.get("required_charts", [])),
        figures_dir=figures_path,
    )
    pdf_result = _evaluate_pdf(metadata=metadata, pdf_dir=pdf_path)
    unit_result = _evaluate_unit_rules(
        latex_text=latex_text,
        unit_rules=ground_truth.get("unit_rules", {}),
    )
    numeric_result = _evaluate_numeric_facts(
        latex_text=latex_text,
        numeric_facts=ground_truth.get("numeric_facts", []),
    )
    pdf_compile_success = pdf_result["pdf_compile_success"]
    latex_exists_score = 1.0 if latex_exists else 0.0

    final_score = round(
        (
            section_result["section_completeness"]
            + chart_result["chart_validity"]
            + (1.0 if pdf_compile_success else 0.0)
            + latex_exists_score
            + unit_result["unit_rule_compliance"]
            + numeric_result["numeric_fact_coverage"]
        )
        / 6.0,
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
        "pdf_exists": pdf_result["pdf_exists"],
        "pdf_files": pdf_result["pdf_files"],
        "latex_exists": latex_exists,
        "unit_rule_compliance": unit_result["unit_rule_compliance"],
        "unit_rule_results": unit_result["unit_rule_results"],
        "missing_unit_rules": unit_result["missing_unit_rules"],
        "numeric_fact_coverage": numeric_result["numeric_fact_coverage"],
        "numeric_fact_results": numeric_result["numeric_fact_results"],
        "missing_numeric_facts": numeric_result["missing_numeric_facts"],
        "required_chart_count": chart_result["required_chart_count"],
        "existing_chart_count": chart_result["existing_chart_count"],
        "final_report_score": final_score,
        "final_score": final_score,
        "numeric_accuracy": numeric_result["numeric_fact_coverage"],
        "numeric_accuracy_note": "computed_from_latex_numeric_fact_coverage",
        "unit_rules": ground_truth.get("unit_rules", {}),
        "numeric_facts": ground_truth.get("numeric_facts", []),
        "metadata_path": str(Path(metadata_path)),
        "latex_path": str(latex_file),
        "figures_dir": str(figures_path),
        "pdf_dir": str(pdf_path),
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
    figures_dir: Path = DEFAULT_FIGURES_DIR,
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

        chart_path = _resolve_chart_path(chart=chart, chart_id=chart_id, figures_dir=figures_dir)
        if chart_path is not None and chart_path.is_file():
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


def _evaluate_pdf(metadata: dict[str, Any], pdf_dir: Path) -> dict[str, Any]:
    pdf_files = sorted(str(path) for path in pdf_dir.glob("*.pdf") if path.is_file())
    pdf_path = _resolve_path(str(metadata.get("pdf_path", "")), base_dir=PROJECT_ROOT)
    pdf_exists = bool(pdf_path and pdf_path.is_file()) or bool(pdf_files)
    pdf_compile_success = bool(metadata.get("pdf_success")) and pdf_exists

    return {
        "pdf_compile_success": pdf_compile_success,
        "pdf_exists": pdf_exists,
        "pdf_files": pdf_files,
    }


def _evaluate_unit_rules(
    latex_text: str,
    unit_rules: Any,
) -> dict[str, Any]:
    specs = _build_unit_rule_specs(unit_rules)
    if not specs:
        return {
            "unit_rule_compliance": 1.0,
            "unit_rule_results": [],
            "missing_unit_rules": [],
        }

    plain_text = _latex_to_plain_text(latex_text)
    results = []
    for spec in specs:
        matched = _unit_rule_matches(plain_text, spec)
        results.append(
            {
                "id": spec["id"],
                "description": spec["description"],
                "expected_unit": spec["unit"],
                "matched": matched,
            }
        )

    matched_count = sum(1 for result in results if result["matched"])
    return {
        "unit_rule_compliance": round(matched_count / len(results), 6),
        "unit_rule_results": results,
        "missing_unit_rules": [
            result["id"] for result in results if not result["matched"]
        ],
    }


def _evaluate_numeric_facts(
    latex_text: str,
    numeric_facts: Any,
) -> dict[str, Any]:
    facts = numeric_facts if isinstance(numeric_facts, list) else []
    if not facts:
        return {
            "numeric_fact_coverage": 1.0,
            "numeric_fact_results": [],
            "missing_numeric_facts": [],
        }

    plain_text = _latex_to_plain_text(latex_text)
    results = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue

        value = _float_value(fact.get("value"))
        variants = _numeric_value_variants(value) if value is not None else []
        matched_variant = _find_numeric_variant(plain_text, variants)
        fact_id = str(fact.get("id", ""))
        results.append(
            {
                "id": fact_id,
                "description": str(fact.get("description", "")),
                "expected_value": value,
                "expected_unit": str(fact.get("unit", "")),
                "accepted_values": variants,
                "matched": matched_variant is not None,
                "matched_value": matched_variant or "",
            }
        )

    if not results:
        return {
            "numeric_fact_coverage": 1.0,
            "numeric_fact_results": [],
            "missing_numeric_facts": [],
        }

    matched_count = sum(1 for result in results if result["matched"])
    return {
        "numeric_fact_coverage": round(matched_count / len(results), 6),
        "numeric_fact_results": results,
        "missing_numeric_facts": [
            result["id"] for result in results if not result["matched"]
        ],
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


def _resolve_chart_path(
    chart: dict[str, Any],
    chart_id: str,
    figures_dir: Path,
) -> Path | None:
    candidate_values = [
        str(chart.get("path", "")),
        str(chart.get("filename", "")),
        f"{chart_id}.png",
    ]
    candidates: list[Path] = []
    for value in candidate_values:
        path = _resolve_path(value, base_dir=PROJECT_ROOT)
        if path is not None:
            candidates.append(path)
        if value:
            candidates.append(figures_dir / value)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return candidates[0] if candidates else None


def _resolve_path(value: str, base_dir: Path) -> Path | None:
    if not value.strip():
        return None

    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _build_unit_rule_specs(unit_rules: Any) -> list[dict[str, Any]]:
    configured_rules = unit_rules if isinstance(unit_rules, dict) else {}
    default_specs = [
        {
            "id": "global_active_power_average",
            "source_key": "Global_active_power average",
            "description": "Global_active_power average must use kW.",
            "terms": ["global_active_power", "global active power", "daya aktif"],
            "context_terms": ["average", "avg", "rata-rata"],
            "unit": "kW",
        },
        {
            "id": "global_active_power_total_energy",
            "source_key": "total energy from Global_active_power",
            "description": "Total energy from Global_active_power must use kWh.",
            "terms": ["global_active_power", "global active power", "daya aktif"],
            "context_terms": ["total", "energy", "energi", "konsumsi"],
            "unit": "kWh",
        },
        {
            "id": "voltage",
            "source_key": "Voltage",
            "description": "Voltage must use V.",
            "terms": ["voltage", "tegangan"],
            "context_terms": [],
            "unit": "V",
        },
        {
            "id": "global_intensity",
            "source_key": "Global_intensity",
            "description": "Global_intensity must use A.",
            "terms": ["global_intensity", "global intensity", "arus", "intensity"],
            "context_terms": [],
            "unit": "A",
        },
    ]

    specs = []
    for spec in default_specs:
        source_key = spec["source_key"]
        if configured_rules and source_key not in configured_rules:
            continue
        expected_value = configured_rules.get(source_key, spec["unit"])
        specs.append({**spec, "expected_rule": str(expected_value)})

    return specs or default_specs


def _unit_rule_matches(text: str, spec: dict[str, Any]) -> bool:
    if not text.strip():
        return False

    lowered_text = text.casefold()
    context_terms = [str(term).casefold() for term in spec.get("context_terms", [])]
    terms = [str(term).casefold() for term in spec.get("terms", [])]
    unit = str(spec.get("unit", ""))

    for term in terms:
        start = 0
        while True:
            index = lowered_text.find(term, start)
            if index < 0:
                break
            window = text[max(0, index - 140) : index + 180]
            lowered_window = window.casefold()
            context_matches = (
                not context_terms
                or any(context_term in lowered_window for context_term in context_terms)
            )
            if context_matches and _contains_unit(window, unit):
                return True
            start = index + len(term)

    return False


def _contains_unit(text: str, unit: str) -> bool:
    if unit == "kW":
        return bool(re.search(r"(?<![A-Za-z])k\s*W(?![A-Za-z])", text))
    if unit == "kWh":
        return bool(re.search(r"(?<![A-Za-z])k\s*Wh(?![A-Za-z])", text))
    if unit == "V":
        return bool(re.search(r"(?<![A-Za-z])V(?![A-Za-z])", text))
    if unit == "A":
        return bool(re.search(r"(?<![A-Za-z])A(?![A-Za-z])", text))

    return bool(re.search(rf"(?<![A-Za-z]){re.escape(unit)}(?![A-Za-z])", text))


def _latex_to_plain_text(latex_text: str) -> str:
    text = latex_text.replace(r"\_", "_")
    text = text.replace(r"\%", "%")
    text = re.sub(r"\\(?:section|subsection|caption|title|author)\*?\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z]+(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"[{}$]", " ", text)
    return " ".join(text.split())


def _numeric_value_variants(value: float) -> list[str]:
    variants: list[str] = []
    for decimals in range(2, 7):
        formatted = f"{value:.{decimals}f}"
        trimmed = formatted.rstrip("0").rstrip(".")
        for candidate in (formatted, trimmed):
            if candidate and candidate not in variants:
                variants.append(candidate)

    if float(value).is_integer():
        integer_text = str(int(value))
        if integer_text not in variants:
            variants.append(integer_text)

    return variants


def _find_numeric_variant(text: str, variants: list[str]) -> str | None:
    for variant in variants:
        decimal_pattern = re.escape(variant).replace(r"\.", r"[\.,]")
        if re.search(rf"(?<![\d\.,]){decimal_pattern}(?![\d\.,])", text):
            return variant
    return None


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
