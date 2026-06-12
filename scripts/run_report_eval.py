from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.report_eval import (
    evaluate_report_artifacts,
    write_report_eval_result,
)


def main() -> int:
    result = evaluate_report_artifacts()
    output_path = write_report_eval_result(result)

    print(f"Report evaluation saved to: {output_path}")
    print(f"section_completeness: {result['section_completeness']:.2%}")
    print(f"chart_validity: {result['chart_validity']:.2%}")
    print(f"pdf_compile_success: {result['pdf_compile_success']}")
    print(f"latex_exists: {result['latex_exists']}")
    print(f"unit_rule_compliance: {result['unit_rule_compliance']:.2%}")
    print(f"numeric_fact_coverage: {result['numeric_fact_coverage']:.2%}")
    print(f"final_report_score: {result['final_report_score']:.2%}")

    return 0 if result["latex_exists"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
