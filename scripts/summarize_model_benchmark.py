"""Aggregate model benchmark runs into one cross-model comparison table.

Reads every ``<model>/manifest.json`` + run CSVs produced by
``scripts/run_model_benchmark.py`` and writes a summary CSV plus a markdown
table (mean ± std across repeats).

Example:
    python scripts/summarize_model_benchmark.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.model_benchmark import (
    DEFAULT_OUTPUT_DIR,
    render_summary_markdown,
    summarize_model_benchmark,
    write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize model benchmark results across models and suites."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing <model>/manifest.json + run CSVs.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Summary CSV output (default: <input-dir>/model_benchmark_summary.csv).",
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=None,
        help="Markdown output (default: <input-dir>/model_benchmark_summary.md).",
    )
    parser.add_argument(
        "--baseline",
        default="gemma2_2b",
        help=(
            "Baseline model key for paired McNemar p-values "
            "(default: gemma2_2b)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    csv_path = args.csv_path or input_dir / "model_benchmark_summary.csv"
    markdown_path = args.markdown_path or input_dir / "model_benchmark_summary.md"

    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        return 1

    summary_rows = summarize_model_benchmark(input_dir, baseline_model=args.baseline)
    if not summary_rows:
        print(
            f"No benchmark results found under {input_dir}. Run "
            "scripts/run_model_benchmark.py first."
        )
        return 1

    write_summary_csv(summary_rows, csv_path)
    markdown = render_summary_markdown(summary_rows)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"CSV:      {csv_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
