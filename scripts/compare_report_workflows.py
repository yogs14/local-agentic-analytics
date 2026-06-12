"""Compare custom and LangGraph report workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.report_engine_comparison import (
    DEFAULT_OUTPUT_PATH,
    run_report_engine_comparison,
    write_report_engine_comparison_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare custom and LangGraph report workflow outputs."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output JSON comparison file.",
    )
    return parser.parse_args()


def print_summary(comparison: dict, output_path: Path) -> None:
    print("Report engine comparison summary:")
    print(f"- custom_success: {comparison['custom_success']}")
    print(f"- langgraph_success: {comparison['langgraph_success']}")
    print(f"- custom_tex_success: {comparison['custom_tex_success']}")
    print(f"- langgraph_tex_success: {comparison['langgraph_tex_success']}")
    print(f"- custom_pdf_success: {comparison['custom_pdf_success']}")
    print(f"- langgraph_pdf_success: {comparison['langgraph_pdf_success']}")
    print(f"- same_chart_count: {comparison['same_chart_count']}")
    print(f"- same_tex_success: {comparison['same_tex_success']}")
    print(f"- same_pdf_success: {comparison['same_pdf_success']}")
    print(f"- custom_tool_call_count: {comparison['custom_tool_call_count']}")
    print(f"- langgraph_tool_call_count: {comparison['langgraph_tool_call_count']}")
    print(f"- output_path: {output_path}")

    if comparison.get("custom_error"):
        print(f"- custom_error: {comparison['custom_error']}")
    if comparison.get("langgraph_error"):
        print(f"- langgraph_error: {comparison['langgraph_error']}")


def main() -> int:
    args = parse_args()
    comparison = run_report_engine_comparison()
    output_path = write_report_engine_comparison_result(comparison, args.output_path)
    print_summary(comparison, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
