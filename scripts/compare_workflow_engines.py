"""Compare custom and LangGraph workflow engines on energy questions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.engine_comparison import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_QUESTIONS_PATH,
    load_questions,
    run_engine_comparison,
    write_engine_comparison_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare custom and LangGraph workflow engines."
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to energy evaluation questions JSON.",
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
    print("Engine comparison summary:")
    print(f"- total_questions: {summary['total_questions']}")
    print(f"- custom_success_count: {summary['custom_success_count']}")
    print(f"- langgraph_success_count: {summary['langgraph_success_count']}")
    print(f"- both_success_count: {summary['both_success_count']}")
    print(f"- same_sql_count: {summary['same_sql_count']}")
    print(f"- avg_custom_latency: {summary['avg_custom_latency']:.3f}s")
    print(f"- avg_langgraph_latency: {summary['avg_langgraph_latency']:.3f}s")
    print(f"- output_path: {output_path}")


def main() -> int:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be greater than 0")
        return 1

    try:
        questions = load_questions(args.questions_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if args.limit is not None:
        questions = questions[: args.limit]

    rows, summary = run_engine_comparison(questions)
    write_engine_comparison_results(rows, args.output_path)
    print_summary(summary, args.output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
