from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.report_eval import (
    DEFAULT_LATEX_PATH,
    DEFAULT_REPORT_EVAL_OUTPUT_PATH,
    DEFAULT_REPORT_GROUND_TRUTH_PATH,
    DEFAULT_REPORT_LOG_PATH,
    evaluate_report_artifacts,
    write_report_eval_result,
)


def main() -> int:
    result = evaluate_report_artifacts(
        ground_truth_path=DEFAULT_REPORT_GROUND_TRUTH_PATH,
        metadata_path=DEFAULT_REPORT_LOG_PATH,
        latex_path=DEFAULT_LATEX_PATH,
    )
    output_path = write_report_eval_result(
        result,
        output_path=DEFAULT_REPORT_EVAL_OUTPUT_PATH,
    )

    print(f"Report evaluation saved to: {output_path}")
    print(f"section_completeness: {result['section_completeness']:.2%}")
    print(f"chart_validity: {result['chart_validity']:.2%}")
    print(f"pdf_compile_success: {result['pdf_compile_success']}")
    print(f"latex_exists: {result['latex_exists']}")
    print(f"final_score: {result['final_score']:.2%}")

    return 0 if result["final_score"] >= 0.75 else 1


if __name__ == "__main__":
    raise SystemExit(main())
