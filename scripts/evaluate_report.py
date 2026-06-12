"""Evaluate generated energy report artifacts against ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.report_eval import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_LATEX_PATH,
    DEFAULT_PDF_DIR,
    DEFAULT_REPORT_EVAL_OUTPUT_PATH,
    DEFAULT_REPORT_GROUND_TRUTH_PATH,
    DEFAULT_REPORT_LOG_PATH,
    evaluate_report_artifacts,
    write_report_eval_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated report artifacts against ground truth."
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=DEFAULT_REPORT_GROUND_TRUTH_PATH,
        help="Path to report ground truth JSON.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_REPORT_LOG_PATH,
        help="Path to report generation metadata JSON.",
    )
    parser.add_argument(
        "--latex-path",
        type=Path,
        default=DEFAULT_LATEX_PATH,
        help="Path to generated LaTeX report.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Path to generated report figures directory.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help="Path to generated report PDF directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_REPORT_EVAL_OUTPUT_PATH,
        help="Path to output report evaluation JSON.",
    )
    return parser.parse_args()


def print_summary(result: dict, output_path: Path) -> None:
    print("Report evaluation summary:")
    print(f"- section_completeness: {result['section_completeness']:.3f}")
    print(f"- chart_validity: {result['chart_validity']:.3f}")
    print(f"- pdf_compile_success: {result['pdf_compile_success']}")
    print(f"- latex_exists: {result['latex_exists']}")
    print(f"- unit_rule_compliance: {result['unit_rule_compliance']:.3f}")
    print(f"- numeric_fact_coverage: {result['numeric_fact_coverage']:.3f}")
    print(f"- final_report_score: {result['final_report_score']:.3f}")
    print(f"- output_path: {output_path}")

    if result.get("missing_sections"):
        print(f"- missing_sections: {', '.join(result['missing_sections'])}")
    if result.get("missing_charts"):
        print(f"- missing_charts: {', '.join(result['missing_charts'])}")
    if result.get("missing_chart_files"):
        print(f"- missing_chart_files: {', '.join(result['missing_chart_files'])}")
    if result.get("missing_unit_rules"):
        print(f"- missing_unit_rules: {', '.join(result['missing_unit_rules'])}")
    if result.get("missing_numeric_facts"):
        print(f"- missing_numeric_facts: {', '.join(result['missing_numeric_facts'])}")


def main() -> int:
    args = parse_args()
    result = evaluate_report_artifacts(
        ground_truth_path=args.ground_truth_path,
        metadata_path=args.metadata_path,
        latex_path=args.latex_path,
        figures_dir=args.figures_dir,
        pdf_dir=args.pdf_dir,
    )
    output_path = write_report_eval_result(result, args.output_path)
    print_summary(result, output_path)
    return 0 if result["latex_exists"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
