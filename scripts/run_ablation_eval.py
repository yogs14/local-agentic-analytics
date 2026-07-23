"""Run the model-vs-scaffolding ablation evaluation on the energy gold set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.ablation_eval import (
    DEFAULT_CSV_OUTPUT_PATH,
    DEFAULT_SUMMARY_OUTPUT_PATH,
    load_default_gold_questions,
    run_ablation_evaluation,
    summarize_ablation_results,
    write_ablation_results,
    write_ablation_summary,
)
from local_agentic_analytics.evaluation.sql_gold_eval import (
    DEFAULT_GOLD_QUESTIONS_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how much text-to-SQL accuracy comes from the model versus "
            "the deterministic scaffolding."
        )
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DEFAULT_GOLD_QUESTIONS_PATH,
        help="Path to gold SQL questions JSON.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_OUTPUT_PATH,
        help="Path to the per-question per-config CSV output.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_PATH,
        help="Path to the summary JSON output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of questions to run per config.",
    )
    return parser.parse_args()


def print_summary(summary: dict[str, Any]) -> None:
    configs = summary.get("configs", {})
    print("\nAblation summary (model vs scaffolding):")
    header = (
        f"{'config':<26} {'total':>5} {'exec_success':>13} "
        f"{'numeric_match':>14} {'result_full':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, stats in configs.items():
        print(
            f"{name:<26} {stats['total']:>5} "
            f"{stats['execution_success_rate']:>12.1%} "
            f"{stats['numeric_match_rate']:>13.1%} "
            f"{stats.get('result_match_full_rate', 0.0):>11.1%}"
        )

    route = summary.get("d_full_route_distribution", {})
    if route:
        print("\nD_full route distribution:")
        print(
            f"- rule_based: {route['rule_based_count']}/{route['total']} "
            f"({route['rule_based_pct']:.1%})"
        )
        print(
            f"- llm:        {route['llm_count']}/{route['total']} "
            f"({route['llm_pct']:.1%})"
        )
        if route.get("other_count"):
            print(f"- other:      {route['other_count']}/{route['total']}")


def main() -> int:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be greater than 0")
        return 1

    try:
        questions = load_default_gold_questions(args.questions_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if args.limit is not None:
        questions = questions[: args.limit]

    rows = run_ablation_evaluation(questions)
    summary = summarize_ablation_results(rows)

    write_ablation_results(rows, args.csv_path)
    write_ablation_summary(summary, args.summary_path)

    print_summary(summary)
    print(f"\nCSV:     {args.csv_path}")
    print(f"Summary: {args.summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
