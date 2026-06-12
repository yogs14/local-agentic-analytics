"""Analyze numeric mismatches from SQL gold evaluation output."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "reports" / "experiments" / "sql_gold_eval.csv"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "sql_gold_mismatch_report.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze numeric mismatches from sql_gold_eval.csv."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to sql_gold_eval.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to markdown mismatch report.",
    )
    return parser.parse_args()


def load_rows(input_path: Path) -> list[dict[str, str]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"SQL gold eval CSV not found: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def is_numeric_mismatch(row: dict[str, str]) -> bool:
    return str(row.get("numeric_match", "")).strip().lower() == "false"


def diagnose(row: dict[str, str]) -> str:
    question = row.get("question", "")
    agent_sql = row.get("agent_sql", "")
    gold_sql = row.get("gold_sql", "")
    agent_sql_upper = agent_sql.upper()
    gold_sql_upper = gold_sql.upper()

    if "/ 60" in gold_sql and "/ 60" not in agent_sql:
        return "possible_unit_conversion_issue"
    if "AVG" in gold_sql_upper and "AVG" not in agent_sql_upper:
        return "possible_aggregation_issue"
    if "GROUP BY" in gold_sql_upper and "GROUP BY" not in agent_sql_upper:
        return "possible_grouping_issue"
    if _question_mentions_date(question) and "CAST(datetime AS DATE)" not in agent_sql:
        return "possible_date_filter_issue"

    return "unknown"


def build_markdown_report(mismatches: list[dict[str, str]]) -> str:
    lines = [
        "# SQL Gold Mismatch Report",
        "",
        f"Total numeric mismatches: {len(mismatches)}",
        "",
    ]

    if not mismatches:
        lines.append("No numeric mismatches found.")
        lines.append("")
        return "\n".join(lines)

    for row in mismatches:
        diagnosis = row["diagnosis"]
        lines.extend(
            [
                f"## {row.get('question_id', '-')}",
                "",
                f"- **Diagnosis:** `{diagnosis}`",
                f"- **Question:** {row.get('question', '')}",
                f"- **Absolute Error:** {row.get('absolute_error', '')}",
                f"- **Relative Error:** {row.get('relative_error', '')}",
                "",
                "**Agent SQL**",
                "",
                "```sql",
                row.get("agent_sql", ""),
                "```",
                "",
                "**Gold SQL**",
                "",
                "```sql",
                row.get("gold_sql", ""),
                "```",
                "",
                "**Agent Result**",
                "",
                "```json",
                row.get("agent_result", ""),
                "```",
                "",
                "**Gold Result**",
                "",
                "```json",
                row.get("gold_result", ""),
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def print_mismatches(mismatches: list[dict[str, str]]) -> None:
    if not mismatches:
        print("Tidak ada numeric mismatch ditemukan.")
        return

    for row in mismatches:
        print(f"question_id: {row.get('question_id', '')}")
        print(f"diagnosis: {row.get('diagnosis', '')}")
        print(f"question: {row.get('question', '')}")
        print("agent_sql:")
        print(row.get("agent_sql", ""))
        print("gold_sql:")
        print(row.get("gold_sql", ""))
        print(f"agent_result: {row.get('agent_result', '')}")
        print(f"gold_result: {row.get('gold_result', '')}")
        print(f"absolute_error: {row.get('absolute_error', '')}")
        print(f"relative_error: {row.get('relative_error', '')}")
        print()


def write_markdown_report(report: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def _question_mentions_date(question: str) -> bool:
    lowered = question.lower()
    return "tanggal" in lowered or "desember" in lowered or "date" in lowered


def main() -> int:
    args = parse_args()

    try:
        rows = load_rows(args.input_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    mismatches = [row for row in rows if is_numeric_mismatch(row)]
    for row in mismatches:
        row["diagnosis"] = diagnose(row)

    print_mismatches(mismatches)
    write_markdown_report(build_markdown_report(mismatches), args.output_path)
    print(f"Markdown report saved to: {args.output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
