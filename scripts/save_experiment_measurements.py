from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.report_eval import evaluate_report_artifacts
from local_agentic_analytics.evaluation.insight_quality_eval import (
    evaluate_insight_quality,
    load_report_metadata,
)


EXPERIMENTS_DIR = PROJECT_ROOT / "reports" / "experiments"
BASELINE_DIR = EXPERIMENTS_DIR / "baseline_gemma2_2b"
V3_REPORT_LOG = EXPERIMENTS_DIR / "report_generation_log.json"
V3_REPORT_EVAL = EXPERIMENTS_DIR / "report_eval.json"
V3_INSIGHT_EVAL = EXPERIMENTS_DIR / "insight_quality_eval.json"
BASELINE_REPORT_LOG = BASELINE_DIR / "report_generation_log.json"
BASELINE_REPORT_EVAL = BASELINE_DIR / "report_eval.json"
COMPARISON_JSON = EXPERIMENTS_DIR / "report_comparison_vs_baseline.json"
COMPARISON_CSV = EXPERIMENTS_DIR / "report_comparison_vs_baseline.csv"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_number(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    baseline_eval = load_json(BASELINE_REPORT_EVAL)
    v3_eval = load_json(V3_REPORT_EVAL)
    baseline_gen = load_json(BASELINE_REPORT_LOG)
    v3_gen = load_json(V3_REPORT_LOG)

    v3_insight_eval = None
    if V3_INSIGHT_EVAL.is_file():
        v3_insight_eval = load_json(V3_INSIGHT_EVAL)

    comparison = {
        "comparison_type": "model_upgrade_gemma2_2b_to_v3",
        "baseline_model": "gemma2:2b",
        "v3_model": "gemma2-energy-insight:v3",
        "report_eval_metrics": {
            "metric": [
                "section_completeness",
                "chart_validity",
                "pdf_compile_success",
                "latex_exists",
                "unit_rule_compliance",
                "numeric_fact_coverage",
                "final_report_score",
            ],
            "baseline_gemma2_2b": [
                baseline_eval.get("section_completeness", 0.0),
                baseline_eval.get("chart_validity", 0.0),
                1.0 if baseline_eval.get("pdf_compile_success") else 0.0,
                1.0 if baseline_eval.get("latex_exists") else 0.0,
                baseline_eval.get("unit_rule_compliance", 0.0),
                baseline_eval.get("numeric_fact_coverage", 0.0),
                baseline_eval.get("final_report_score", 0.0),
            ],
            "v3_gemma2_energy_insight": [
                v3_eval.get("section_completeness", 0.0),
                v3_eval.get("chart_validity", 0.0),
                1.0 if v3_eval.get("pdf_compile_success") else 0.0,
                1.0 if v3_eval.get("latex_exists") else 0.0,
                v3_eval.get("unit_rule_compliance", 0.0),
                v3_eval.get("numeric_fact_coverage", 0.0),
                v3_eval.get("final_report_score", 0.0),
            ],
            "delta_v3_minus_baseline": [
                round(coerce_number(v3_eval.get("section_completeness")) - coerce_number(baseline_eval.get("section_completeness")), 4),
                round(coerce_number(v3_eval.get("chart_validity")) - coerce_number(baseline_eval.get("chart_validity")), 4),
                round((1.0 if v3_eval.get("pdf_compile_success") else 0.0) - (1.0 if baseline_eval.get("pdf_compile_success") else 0.0), 4),
                round((1.0 if v3_eval.get("latex_exists") else 0.0) - (1.0 if baseline_eval.get("latex_exists") else 0.0), 4),
                round(coerce_number(v3_eval.get("unit_rule_compliance")) - coerce_number(baseline_eval.get("unit_rule_compliance")), 4),
                round(coerce_number(v3_eval.get("numeric_fact_coverage")) - coerce_number(baseline_eval.get("numeric_fact_coverage")), 4),
                round(coerce_number(v3_eval.get("final_report_score")) - coerce_number(baseline_eval.get("final_report_score")), 4),
            ],
        },
        "generation_summary": {
            "baseline_gemma2_2b": {
                "success": baseline_gen.get("success", False),
                "chart_count": baseline_gen.get("chart_count", 0),
                "insight_success_count": baseline_gen.get("insight_success_count", 0),
                "insight_failed_count": baseline_gen.get("insight_failed_count", 0),
                "error_message": baseline_gen.get("error_message", ""),
            },
            "v3_gemma2_energy_insight": {
                "success": v3_gen.get("success", False),
                "chart_count": v3_gen.get("chart_count", 0),
                "insight_success_count": v3_gen.get("insight_success_count", 0),
                "insight_failed_count": v3_gen.get("insight_failed_count", 0),
                "error_message": v3_gen.get("error_message", ""),
            },
        },
    }

    if v3_insight_eval:
        comparison["insight_quality_metrics"] = {
            "metric": [
                "acceptance_rate",
                "number_groundedness",
                "fully_grounded_rate",
                "unit_presence_rate",
                "concept_coverage_rate",
                "mean_concept_terms",
                "markdown_clean_rate",
                "total_latency_seconds",
                "composite_insight_score",
            ],
            "v3_gemma2_energy_insight": [
                v3_insight_eval.get("acceptance_rate", 0.0),
                v3_insight_eval.get("number_groundedness", 0.0),
                v3_insight_eval.get("fully_grounded_rate", 0.0),
                v3_insight_eval.get("unit_presence_rate", 0.0),
                v3_insight_eval.get("concept_coverage_rate", 0.0),
                v3_insight_eval.get("mean_concept_terms", 0.0),
                v3_insight_eval.get("markdown_clean_rate", 0.0),
                v3_insight_eval.get("total_latency_seconds", 0.0),
                v3_insight_eval.get("composite_insight_score", 0.0),
            ],
            "note": "Baseline (gemma2:2b) did not produce insights, so no insight-quality eval exists.",
        }

    comparison["unit_rule_detail"] = {
        "baseline_gemma2_2b": baseline_eval.get("unit_rule_results", []),
        "v3_gemma2_energy_insight": v3_eval.get("unit_rule_results", []),
    }
    comparison["numeric_fact_detail"] = {
        "baseline_gemma2_2b": baseline_eval.get("numeric_fact_results", []),
        "v3_gemma2_energy_insight": v3_eval.get("numeric_fact_results", []),
    }

    COMPARISON_JSON.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_JSON.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_comparison_csv(comparison, COMPARISON_CSV)

    print(f"Comparison saved to: {COMPARISON_JSON}")
    print(f"Comparison CSV saved to: {COMPARISON_CSV}")

    print("\n--- Report Eval Comparison ---")
    metrics = comparison["report_eval_metrics"]["metric"]
    baseline_vals = comparison["report_eval_metrics"]["baseline_gemma2_2b"]
    v3_vals = comparison["report_eval_metrics"]["v3_gemma2_energy_insight"]
    deltas = comparison["report_eval_metrics"]["delta_v3_minus_baseline"]
    for i, metric in enumerate(metrics):
        print(f"  {metric}: {baseline_vals[i]:.4f} -> {v3_vals[i]:.4f} (delta={deltas[i]:+.4f})")

    if v3_insight_eval:
        print("\n--- Insight Quality (v3 only) ---")
        iq_metrics = comparison["insight_quality_metrics"]["metric"]
        iq_vals = comparison["insight_quality_metrics"]["v3_gemma2_energy_insight"]
        for i, metric in enumerate(iq_metrics):
            print(f"  {metric}: {iq_vals[i]}")

    return 0


def _write_comparison_csv(comparison: dict, csv_path: Path) -> None:
    rows: list[dict] = []
    report = comparison["report_eval_metrics"]
    metrics = report["metric"]
    baseline = report["baseline_gemma2_2b"]
    v3 = report["v3_gemma2_energy_insight"]
    deltas = report["delta_v3_minus_baseline"]

    for i, metric in enumerate(metrics):
        rows.append({
            "evaluation_type": "report_eval",
            "metric": metric,
            "baseline_gemma2_2b": baseline[i],
            "v3_gemma2_energy_insight": v3[i],
            "delta": deltas[i],
        })

    gen = comparison["generation_summary"]
    for key in ["success", "chart_count", "insight_success_count", "insight_failed_count"]:
        rows.append({
            "evaluation_type": "generation",
            "metric": key,
            "baseline_gemma2_2b": gen["baseline_gemma2_2b"].get(key, ""),
            "v3_gemma2_energy_insight": gen["v3_gemma2_energy_insight"].get(key, ""),
            "delta": "",
        })

    iq = comparison.get("insight_quality_metrics")
    if iq:
        for i, metric in enumerate(iq["metric"]):
            rows.append({
                "evaluation_type": "insight_quality",
                "metric": metric,
                "baseline_gemma2_2b": "",
                "v3_gemma2_energy_insight": iq["v3_gemma2_energy_insight"][i],
                "delta": "",
            })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["evaluation_type", "metric", "baseline_gemma2_2b", "v3_gemma2_energy_insight", "delta"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
