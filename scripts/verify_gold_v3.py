"""Verify the expanded gold sets (energy v3, finance v2) against the REAL DB.

Extends scripts/verify_gold_v2.py in two ways:

1. Executes EVERY gold SQL (inherited + generated) and flags degenerate
   results (errors, empty, NULL scalars, zero counts, shape mismatches) —
   same checks as v2.
2. For generated candidates (which carry ``template`` + ``params``), the
   expected result is ALSO recomputed with an independent pandas
   implementation (never by re-running the SQL) and cross-checked against
   the executed result with float tolerance. Mismatches are flagged
   ``PANDAS_MISMATCH``.

All recorded values come from actual execution — nothing is fabricated.
A human reviews the generated review files and flips ``verified: true`` in
the manifests. Exit code is non-zero if anything errored or was flagged.

Examples:
    python scripts/verify_gold_v3.py
    python scripts/verify_gold_v3.py --manifest references/sql_gold/finance_gold_questions_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from local_agentic_analytics.evaluation.gold_candidates import (
    ENERGY_TABLE,
    FINANCE_TABLE,
    recompute_expected,
)
from local_agentic_analytics.evaluation.result_comparison import (
    compare_result_sets,
)
from local_agentic_analytics.evaluation.sql_gold_eval import (
    dataframe_to_result_text,
)
from local_agentic_analytics.tools.duckdb_tool import DuckDBTool


DB_PATH = PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"
DEFAULT_MANIFESTS = (
    PROJECT_ROOT / "references" / "sql_gold" / "energy_gold_questions_v3.json",
    PROJECT_ROOT / "references" / "sql_gold" / "finance_gold_questions_v2.json",
)
REVIEW_NAMES = {
    "energy_gold_questions_v3": "gold_review_v3.md",
    "finance_gold_questions_v2": "finance_gold_review_v2.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute every gold SQL in the expanded manifests and cross-check "
            "generated candidates against an independent pandas computation."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
        help=(
            "Manifest JSON to verify (repeatable). Default: energy v3 and "
            "finance v2."
        ),
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="DuckDB database path.",
    )
    return parser.parse_args()


def _is_nullish(value) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _format_value(value) -> str:
    if _is_nullish(value):
        return "NULL"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _raw_table_for(manifest_stem: str, tool: DuckDBTool) -> pd.DataFrame:
    table = ENERGY_TABLE if manifest_stem.startswith("energy") else FINANCE_TABLE
    return tool.execute_query(f"SELECT * FROM {table}")


def _evaluate(question: dict, tool: DuckDBTool, raw_df: pd.DataFrame) -> dict:
    result = {
        "id": question["id"],
        "question": question["question"],
        "source": question.get("source", ""),
        "hardness": question.get("hardness", ""),
        "template": question.get("template", ""),
        "verified": question.get("verified", False),
        "sql": "",
        "row_count": None,
        "computed_value": "",
        "pandas_check": "",
        "error": "",
        "flags": [],
    }

    sql_path = PROJECT_ROOT / question["gold_sql_file"]
    if not sql_path.is_file():
        result["error"] = f"gold SQL file not found: {sql_path}"
        result["flags"].append("MISSING_FILE")
        return result
    sql = sql_path.read_text(encoding="utf-8").strip()
    result["sql"] = sql

    try:
        df = tool.execute_query(sql)
    except Exception as exc:  # noqa: BLE001 - record any failure for review
        result["error"] = str(exc)
        result["flags"].append("EXEC_ERROR")
        return result

    row_count = int(df.shape[0])
    result["row_count"] = row_count
    if row_count == 0:
        result["flags"].append("EMPTY_RESULT")

    if df.shape == (1, 1):
        scalar = df.iat[0, 0]
        result["computed_value"] = _format_value(scalar)
        if _is_nullish(scalar):
            result["flags"].append("NULL_SCALAR")
        elif (
            question.get("expected_unit") in ("records", "count")
            and isinstance(scalar, (int, float))
            and float(scalar) == 0.0
        ):
            result["flags"].append("ZERO_COUNT")
    elif row_count > 0:
        first_row = {
            key: _format_value(value)
            for key, value in df.iloc[0].to_dict().items()
        }
        result["computed_value"] = json.dumps(first_row, ensure_ascii=False)

    declared_shape = question.get("result_shape", "")
    if declared_shape == "scalar" and df.shape != (1, 1):
        result["flags"].append(
            f"SHAPE_MISMATCH(declared=scalar,got={df.shape[0]}x{df.shape[1]})"
        )
    if declared_shape == "table" and row_count <= 1:
        result["flags"].append(
            f"SHAPE_MISMATCH(declared=table,got_rows={row_count})"
        )

    template = question.get("template", "")
    if template:
        try:
            expected = recompute_expected(
                template, question.get("params", {}), raw_df
            )
        except Exception as exc:  # noqa: BLE001
            result["pandas_check"] = f"recompute failed: {exc}"
            result["flags"].append("PANDAS_RECOMPUTE_ERROR")
            return result

        comparison = compare_result_sets(
            dataframe_to_result_text(df),
            dataframe_to_result_text(expected),
        )
        if comparison["result_match_full"] is True:
            result["pandas_check"] = "match"
        else:
            result["pandas_check"] = str(comparison["result_match_reason"])
            result["flags"].append("PANDAS_MISMATCH")

    return result


def _write_review(manifest_path: Path, results: list[dict]) -> Path:
    review_name = REVIEW_NAMES.get(
        manifest_path.stem, f"gold_review_{manifest_path.stem}.md"
    )
    review_path = manifest_path.parent / review_name

    flagged = [r for r in results if r["flags"]]
    errored = [r for r in results if r["error"]]
    checked = [r for r in results if r["template"]]
    matched = [r for r in checked if r["pandas_check"] == "match"]

    lines: list[str] = []
    lines.append(f"# Gold SQL Review — {manifest_path.stem}\n")
    lines.append("- Source DB: `databases/duckdb/analytics.duckdb`")
    lines.append(f"- Manifest: `{manifest_path.name}`")
    lines.append("- Generated by: `scripts/verify_gold_v3.py` (read-only)")
    lines.append(
        "- Every `computed value` comes from executing the gold SQL; every "
        "`pandas_check` comes from an independent pandas recomputation of "
        "the template. Nothing is hand-entered.\n"
    )
    lines.append(
        f"**Summary:** {len(results)} questions | {len(errored)} execution "
        f"errors | {len(flagged)} flagged | pandas cross-check "
        f"{len(matched)}/{len(checked)} match.\n"
    )

    if flagged:
        lines.append("## Flagged questions (need attention)\n")
        for r in flagged:
            lines.append(f"- **{r['id']}**: {', '.join(r['flags'])}")
        lines.append("")

    lines.append("## Per-question results\n")
    for r in results:
        lines.append(f"### {r['id']}")
        lines.append("")
        lines.append(f"- **Question:** {r['question']}")
        lines.append(f"- **source / hardness:** {r['source']} / {r['hardness']}")
        lines.append(f"- **verified:** {r['verified']}")
        lines.append(f"- **row_count:** {r['row_count']}")
        lines.append(f"- **computed value:** {r['computed_value']}")
        if r["template"]:
            lines.append(f"- **pandas_check:** {r['pandas_check']}")
        if r["error"]:
            lines.append(f"- **ERROR:** {r['error']}")
        lines.append(
            f"- **flags:** {', '.join(r['flags']) if r['flags'] else 'none'}"
        )
        lines.append("")
        lines.append("```sql")
        lines.append(r["sql"])
        lines.append("```")
        lines.append("")

    review_path.write_text("\n".join(lines), encoding="utf-8")
    return review_path


def main() -> int:
    args = parse_args()
    manifests = args.manifest or list(DEFAULT_MANIFESTS)

    if not args.db_path.is_file():
        print(f"Error: database not found: {args.db_path}")
        return 1

    tool = DuckDBTool(str(args.db_path))
    exit_code = 0
    try:
        for manifest_path in manifests:
            if not manifest_path.is_file():
                print(f"Error: manifest not found: {manifest_path}")
                exit_code = 1
                continue

            questions = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(
                f"\n=== {manifest_path.stem}: {len(questions)} questions ==="
            )
            raw_df = _raw_table_for(manifest_path.stem, tool)

            results = [_evaluate(q, tool, raw_df) for q in questions]
            review_path = _write_review(manifest_path, results)

            flagged = [r for r in results if r["flags"]]
            errored = [r for r in results if r["error"]]
            checked = [r for r in results if r["template"]]
            matched = [r for r in checked if r["pandas_check"] == "match"]

            print(f"  execution errors    : {len(errored)}")
            print(f"  flagged             : {len(flagged)}")
            print(
                f"  pandas cross-check  : {len(matched)}/{len(checked)} match"
            )
            for r in flagged:
                print(f"    [FLAG] {r['id']}: {', '.join(r['flags'])}")
            print(f"  review file         : {review_path}")

            if flagged or errored:
                exit_code = 1
    finally:
        tool.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
