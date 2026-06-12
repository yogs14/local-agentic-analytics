"""Run gold SQL benchmark evaluation for energy questions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.sql_gold_eval import (
    DEFAULT_GOLD_QUESTIONS_PATH,
    DEFAULT_OUTPUT_PATH,
    load_gold_questions,
    run_sql_gold_evaluation,
    summarize_sql_gold_results,
    write_sql_gold_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gold SQL evaluation for energy text-to-SQL accuracy."
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DEFAULT_GOLD_QUESTIONS_PATH,
        help="Path to gold SQL questions JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of questions to run.",
    )
    return parser.parse_args()


def print_summary(summary: dict[str, float | int], output_path: Path) -> None:
    print("SQL gold evaluation summary:")
    print(f"- total_questions: {summary['total_questions']}")
    print(f"- agent_success_count: {summary['agent_success_count']}")
    print(f"- gold_success_count: {summary['gold_success_count']}")
    print(f"- numeric_compared_count: {summary['numeric_compared_count']}")
    print(f"- numeric_match_count: {summary['numeric_match_count']}")
    print(f"- numeric_match_rate: {summary['numeric_match_rate']:.2%}")
    print(f"- output_path: {output_path}")


def main() -> int:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be greater than 0")
        return 1

    try:
        questions = load_gold_questions(args.questions_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if args.limit is not None:
        questions = questions[: args.limit]

    rows = run_sql_gold_evaluation(questions)
    write_sql_gold_results(rows, args.output_path)
    print_summary(summarize_sql_gold_results(rows), args.output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
