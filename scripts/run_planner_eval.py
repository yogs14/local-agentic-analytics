"""Run the planner routing evaluation on the finance gold question set.

Compares rule-based-only routing against rule-based+LLM routing so the LLM
planner's contribution to routing accuracy is measurable.

Run:
    python scripts/run_planner_eval.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.planner_eval import (
    DEFAULT_CSV_OUTPUT_PATH,
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_SUMMARY_OUTPUT_PATH,
    load_planner_questions,
    run_planner_evaluation,
    summarize_planner_results,
    write_planner_results,
    write_planner_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure planner routing accuracy (rule-based-only vs "
            "rule-based+LLM) on the finance gold question set."
        )
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to finance questions JSON.",
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
    print("\nPlanner routing summary:")
    header = f"{'config':<22} {'total':>5} {'correct':>8} {'accuracy':>9}"
    print(header)
    print("-" * len(header))
    for name, stats in configs.items():
        print(
            f"{name:<22} {stats['total']:>5} {stats['correct']:>8} "
            f"{stats['accuracy']:>8.1%}"
        )

    for name, stats in configs.items():
        print(f"\n[{name}] route_source: {stats['route_source_breakdown']}")
        per_route = stats.get("per_route_accuracy", {})
        for route, route_stats in per_route.items():
            print(
                f"  {route:<16} {route_stats['correct']}/{route_stats['total']} "
                f"({route_stats['accuracy']:.1%})"
            )


def main() -> int:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be greater than 0")
        return 1

    try:
        questions = load_planner_questions(args.questions_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if args.limit is not None:
        questions = questions[: args.limit]

    rows = run_planner_evaluation(questions)
    summary = summarize_planner_results(rows)

    write_planner_results(rows, args.csv_path)
    write_planner_summary(summary, args.summary_path)

    print_summary(summary)
    print(f"\nCSV:     {args.csv_path}")
    print(f"Summary: {args.summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
