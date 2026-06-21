"""Verify v2 gold SQL against the REAL energy DuckDB database.

For every question in references/sql_gold/energy_gold_questions_v2.json:
  - load and execute its gold .sql against databases/duckdb/analytics.duckdb
  - assert no execution error
  - record row_count and (for scalar) the computed value
  - FLAG empty / NULL / degenerate results for human review

Writes references/sql_gold/gold_review_v2.md and prints a summary.
All recorded values come from ACTUAL execution -- nothing is fabricated.

Re-runnable: python scripts/verify_gold_v2.py
Exit code is non-zero if any query errored or any result is flagged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from local_agentic_analytics.tools.duckdb_tool import DuckDBTool

DB_PATH = PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"
MANIFEST_PATH = (
    PROJECT_ROOT / "references" / "sql_gold" / "energy_gold_questions_v2.json"
)
REVIEW_PATH = PROJECT_ROOT / "references" / "sql_gold" / "gold_review_v2.md"


def _is_nullish(value) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _scalar_value(df: pd.DataFrame):
    """Return the single value if df is 1x1, else None."""
    if df.shape == (1, 1):
        return df.iat[0, 0]
    return None


def _format_value(value) -> str:
    if _is_nullish(value):
        return "NULL"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _evaluate(question: dict, tool: DuckDBTool) -> dict:
    qid = question["id"]
    sql_path = PROJECT_ROOT / question["gold_sql_file"]
    result = {
        "id": qid,
        "question": question["question"],
        "result_shape": question.get("result_shape", ""),
        "resolver_eligible": question.get("resolver_eligible", ""),
        "expected_unit": question.get("expected_unit", ""),
        "sql": "",
        "row_count": None,
        "computed_value": "",
        "error": "",
        "flags": [],
    }

    if not sql_path.is_file():
        result["error"] = f"gold SQL file not found: {sql_path}"
        result["flags"].append("MISSING_FILE")
        return result

    sql = sql_path.read_text(encoding="utf-8").strip()
    result["sql"] = sql

    try:
        df = tool.execute_query(sql)
    except Exception as exc:  # noqa: BLE001 - we want to record any failure
        result["error"] = str(exc)
        result["flags"].append("EXEC_ERROR")
        return result

    row_count = int(df.shape[0])
    result["row_count"] = row_count

    declared_shape = question.get("result_shape", "")

    if row_count == 0:
        result["flags"].append("EMPTY_RESULT")

    scalar = _scalar_value(df)
    if scalar is not None or df.shape == (1, 1):
        result["computed_value"] = _format_value(scalar)
        if _is_nullish(scalar):
            result["flags"].append("NULL_SCALAR")
        elif (
            question.get("expected_unit") in ("records", "count")
            and isinstance(scalar, (int, float))
            and float(scalar) == 0.0
        ):
            # A "how many records" question that answers zero is degenerate:
            # a wrong query could also return 0, so it cannot discriminate.
            result["flags"].append("ZERO_COUNT")
    else:
        # Non-scalar: record a compact preview and check for all-null first row.
        if row_count > 0:
            first_row = df.iloc[0].to_dict()
            result["computed_value"] = json.dumps(
                {k: _format_value(v) for k, v in first_row.items()},
                ensure_ascii=False,
            )
            if all(_is_nullish(v) for v in df.iloc[0].tolist()):
                result["flags"].append("NULL_FIRST_ROW")

    # Shape consistency hints (informational, also flagged for review).
    if declared_shape == "scalar" and df.shape != (1, 1):
        result["flags"].append(
            f"SHAPE_MISMATCH(declared=scalar,got={df.shape[0]}x{df.shape[1]})"
        )
    if declared_shape == "table" and row_count <= 1:
        result["flags"].append(
            f"SHAPE_MISMATCH(declared=table,got_rows={row_count})"
        )

    return result


def main() -> int:
    if not DB_PATH.is_file():
        raise SystemExit(f"Database not found: {DB_PATH}")

    questions = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tool = DuckDBTool(str(DB_PATH))

    results = [_evaluate(q, tool) for q in questions]
    tool.close()

    flagged = [r for r in results if r["flags"]]
    errored = [r for r in results if r["error"]]

    lines: list[str] = []
    lines.append("# Gold SQL Review v2 (executed against REAL database)\n")
    lines.append(f"- Source DB: `databases/duckdb/analytics.duckdb`")
    lines.append(f"- Manifest: `references/sql_gold/energy_gold_questions_v2.json`")
    lines.append(f"- Generated by: `scripts/verify_gold_v2.py` (read-only)")
    lines.append(
        "- All `computed value` / `row_count` entries come from actually "
        "executing each gold SQL. Nothing here is hand-entered.\n"
    )
    lines.append(
        f"**Summary:** {len(results)} questions | "
        f"{len(errored)} execution errors | "
        f"{len(flagged)} flagged for review.\n"
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
        lines.append(f"- **result_shape:** {r['result_shape']}")
        lines.append(f"- **resolver_eligible:** {r['resolver_eligible']}")
        lines.append(f"- **expected_unit:** {r['expected_unit']}")
        lines.append(f"- **row_count:** {r['row_count']}")
        lines.append(f"- **computed value:** {r['computed_value']}")
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

    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")

    # Console summary.
    print(f"Verified {len(results)} gold questions against {DB_PATH.name}")
    print(f"  execution errors : {len(errored)}")
    print(f"  flagged          : {len(flagged)}")
    for r in results:
        status = "FLAG" if r["flags"] else "ok  "
        val = r["computed_value"][:40] if r["computed_value"] else ""
        print(
            f"  [{status}] {r['id']} rows={r['row_count']} "
            f"value={val} {('<' + ','.join(r['flags']) + '>') if r['flags'] else ''}"
        )
    print(f"\nWrote {REVIEW_PATH}")

    return 1 if (errored or flagged) else 0


if __name__ == "__main__":
    sys.exit(main())
