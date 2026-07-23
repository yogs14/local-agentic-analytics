import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.insight_agent import InsightAgent
from local_agentic_analytics.evaluation.insight_model_benchmark import (
    run_insight_model_benchmark,
    write_benchmark_result,
)
from local_agentic_analytics.evaluation.insight_quality_eval import (
    load_report_metadata,
)


def _chart_contexts_from_log() -> list[dict]:
    metadata = load_report_metadata()
    contexts = []
    for record in metadata.get("insights", []):
        if not isinstance(record, dict):
            continue
        stats = record.get("stats")
        if isinstance(stats, dict) and stats:
            contexts.append(
                {
                    "chart_id": record["chart_id"],
                    "chart_title": record.get("chart_title", record["chart_id"]),
                    "chart_path": record.get("chart_path", ""),
                    "stats": stats,
                }
            )
    return contexts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeated-run quality + efficiency benchmark for the v3 narrator."
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    contexts = _chart_contexts_from_log()
    if not contexts:
        print("No chart stats found in report_generation_log.json. Run the report first.")
        return 1

    agent = InsightAgent.for_energy_finetune()
    result = run_insight_model_benchmark(agent, contexts, repeats=args.repeats)
    paths = write_benchmark_result(result)

    quality = result["quality"]
    efficiency = result["efficiency"]
    print(f"Benchmark saved to: {paths['json']}")
    print(f"per-sample CSV: {paths['csv']}")
    print(f"model: {result.get('model')} | repeats: {result['repeats']} | samples: {quality['samples']}")
    print(f"acceptance_rate: {quality['acceptance_rate']:.2%}")
    print(f"number_groundedness: {quality['number_groundedness']:.2%}")
    print(f"fully_grounded_rate: {quality['fully_grounded_rate']:.2%}")
    print(f"unit_presence_rate: {quality['unit_presence_rate']:.2%}")
    print(f"markdown_clean_rate: {quality['markdown_clean_rate']:.2%}")
    print(f"mean_concept_terms: {quality['mean_concept_terms']}")
    print(
        "latency_seconds: mean=%.2f sd=%.2f min=%.2f max=%.2f"
        % (
            efficiency["latency_seconds"]["mean"],
            efficiency["latency_seconds"]["sd"],
            efficiency["latency_seconds"]["min"],
            efficiency["latency_seconds"]["max"],
        )
    )
    print(
        "tokens_per_second: mean=%.2f sd=%.2f"
        % (
            efficiency["tokens_per_second"]["mean"],
            efficiency["tokens_per_second"]["sd"],
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
