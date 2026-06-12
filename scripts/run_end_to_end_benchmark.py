"""Run end-to-end benchmark for custom and LangGraph workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.batch_eval import DEFAULT_QUESTIONS_PATH
from local_agentic_analytics.evaluation.end_to_end_benchmark import (
    DEFAULT_BENCHMARK_OUTPUT_PATH,
    DEFAULT_BENCHMARK_SUMMARY_PATH,
    load_questions,
    run_end_to_end_benchmark,
    write_end_to_end_benchmark_rows,
    write_end_to_end_benchmark_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end benchmark for custom and LangGraph workflows."
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to energy Q&A evaluation questions JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_BENCHMARK_OUTPUT_PATH,
        help="Path to output CSV benchmark file.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_BENCHMARK_SUMMARY_PATH,
        help="Path to output summary JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of Q&A questions to run.",
    )
    return parser.parse_args()


def print_summary(summary: dict, output_path: Path, summary_path: Path) -> None:
    print("End-to-end benchmark summary:")
    print(f"- custom_success_rate: {summary['custom_success_rate']:.3f}")
    print(f"- langgraph_success_rate: {summary['langgraph_success_rate']:.3f}")
    print(f"- custom_avg_latency: {summary['custom_avg_latency']:.3f}s")
    print(f"- langgraph_avg_latency: {summary['langgraph_avg_latency']:.3f}s")
    print(f"- avg_tool_call_count: {summary['avg_tool_call_count']:.3f}")
    print(f"- report_pdf_success: {summary['report_pdf_success']}")
    print(f"- report_eval_score: {summary['report_eval_score']}")
    print(f"- gold_numeric_match_rate: {summary['gold_numeric_match_rate']:.3f}")
    print(f"- output_path: {output_path}")
    print(f"- summary_path: {summary_path}")


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

    rows, summary = run_end_to_end_benchmark(questions=questions)
    output_path = write_end_to_end_benchmark_rows(rows, args.output_path)
    summary_path = write_end_to_end_benchmark_summary(summary, args.summary_path)
    print_summary(summary, output_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
