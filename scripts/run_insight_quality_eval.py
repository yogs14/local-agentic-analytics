from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.insight_quality_eval import (
    evaluate_insight_quality,
    load_report_metadata,
    write_insight_quality_result,
)


def main() -> int:
    metadata = load_report_metadata()
    result = evaluate_insight_quality(metadata)
    paths = write_insight_quality_result(result)

    print(f"Insight quality evaluation saved to: {paths['json']}")
    print(f"per-insight CSV: {paths['csv']}")
    print(f"insights_evaluated: {result['insights_evaluated']}/{result['insights_total']}")
    print(f"acceptance_rate: {result['acceptance_rate']:.2%}")
    print(f"number_groundedness: {result['number_groundedness']:.2%}")
    print(f"fully_grounded_rate: {result['fully_grounded_rate']:.2%}")
    print(f"unit_presence_rate: {result['unit_presence_rate']:.2%}")
    print(f"concept_coverage_rate: {result['concept_coverage_rate']:.2%}")
    print(f"mean_concept_terms: {result['mean_concept_terms']}")
    print(f"markdown_clean_rate: {result['markdown_clean_rate']:.2%}")
    print(f"mean_sentence_count: {result['mean_sentence_count']}")
    print(f"total_latency: {result.get('total_latency_label', '-')}")
    print(f"composite_insight_score: {result.get('composite_insight_score', 0.0):.4f}")

    if result.get("step_latencies"):
        print("Step latencies:")
        for step, seconds in result["step_latencies"].items():
            print(f"  - {step}: {seconds:.4f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
