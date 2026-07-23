"""Generate the expanded gold SQL candidate sets (energy v3, finance v2).

Deterministic (seeded) template-based generation:
- parameter pools (clean days, full months, thresholds) are read from the
  REAL DuckDB database (read-only), so every generated query has a
  well-defined answer;
- each candidate carries its ``template`` + ``params`` so
  ``scripts/verify_gold_v3.py`` can recompute the expected result with an
  independent pandas implementation;
- every generated item is marked ``verified: false`` until a human reviews it.

Existing gold sets are inherited untouched: energy v2 (36) and finance v1 (8)
items keep their original SQL files and get ``verified: true`` (they were
already human-verified) plus a computed ``hardness``.

Example:
    python scripts/generate_gold_candidates.py
    python scripts/generate_gold_candidates.py --seed 42 --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import duckdb

from local_agentic_analytics.evaluation.gold_candidates import (
    ENERGY_POOLS_SQL,
    FINANCE_TABLE,
    FinancePools,
    build_energy_pools,
    compute_hardness,
    generate_energy_candidates,
    generate_finance_candidates,
)


DB_PATH = PROJECT_ROOT / "databases" / "duckdb" / "analytics.duckdb"
SQL_DIR = PROJECT_ROOT / "references" / "sql_gold"
ENERGY_V2_PATH = SQL_DIR / "energy_gold_questions_v2.json"
FINANCE_V1_PATH = SQL_DIR / "finance_gold_questions.json"
ENERGY_V3_PATH = SQL_DIR / "energy_gold_questions_v3.json"
FINANCE_V2_PATH = SQL_DIR / "finance_gold_questions_v2.json"

MIN_ENERGY_TOTAL = 100
MIN_FINANCE_TOTAL = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate energy v3 (100+) and finance v2 (30+) gold SQL "
            "candidate sets from deterministic templates."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for parameter sampling (default: 42).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="DuckDB database to read parameter pools from (read-only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and hardness distribution without writing files.",
    )
    return parser.parse_args()


def collect_energy_pools(conn: "duckdb.DuckDBPyConnection"):
    clean_days_raw = conn.execute(ENERGY_POOLS_SQL["clean_days"]).fetchall()
    clean_days = [row[0] for row in clean_days_raw]
    clean_days = [
        day.date() if isinstance(day, dt.datetime) else day for day in clean_days
    ]
    gap_median, gap_p90 = conn.execute(
        ENERGY_POOLS_SQL["gap_quantiles"]
    ).fetchone()
    voltage_median = conn.execute(ENERGY_POOLS_SQL["voltage_median"]).fetchone()[0]
    intensity_p90 = conn.execute(ENERGY_POOLS_SQL["intensity_p90"]).fetchone()[0]
    data_min, data_max = conn.execute(
        "SELECT MIN(CAST(datetime AS DATE)), MAX(CAST(datetime AS DATE)) "
        "FROM electric_power"
    ).fetchone()
    return build_energy_pools(
        clean_days=clean_days,
        gap_median=float(gap_median),
        gap_p90=float(gap_p90),
        voltage_median=float(voltage_median),
        intensity_p90=float(intensity_p90),
        data_min_day=data_min,
        data_max_day=data_max,
    )


def collect_finance_pools(conn: "duckdb.DuckDBPyConnection") -> FinancePools:
    tickers = [
        row[0]
        for row in conn.execute(
            f"SELECT DISTINCT ticker FROM {FINANCE_TABLE} ORDER BY ticker"
        ).fetchall()
    ]
    min_date, max_date = conn.execute(
        f"SELECT MIN(date), MAX(date) FROM {FINANCE_TABLE}"
    ).fetchone()

    months: list[tuple[int, int]] = []
    cursor = dt.date(min_date.year, min_date.month, 1)
    while cursor <= max_date:
        if cursor.month == 12:
            month_end = dt.date(cursor.year, 12, 31)
            following = dt.date(cursor.year + 1, 1, 1)
        else:
            following = dt.date(cursor.year, cursor.month + 1, 1)
            month_end = following - dt.timedelta(days=1)
        # Trading data starts on the first trading day (not calendar day 1),
        # so a month counts as complete when it ends on/before the last
        # trading day; only the trailing partial month is excluded.
        if month_end <= max_date:
            months.append((cursor.year, cursor.month))
        cursor = following

    close_median = {
        ticker: float(
            conn.execute(
                f"SELECT quantile_cont(close, 0.5) FROM {FINANCE_TABLE} "
                "WHERE ticker = ?",
                [ticker],
            ).fetchone()[0]
        )
        for ticker in tickers
    }
    return FinancePools(
        tickers=tickers,
        months=months,
        min_date=min_date,
        max_date=max_date,
        close_median=close_median,
    )


def inherit_items(path: Path, source: str) -> list[dict[str, Any]]:
    """Inherit an existing verified gold set into the new manifest untouched."""
    items = json.loads(path.read_text(encoding="utf-8"))
    inherited: list[dict[str, Any]] = []
    for item in items:
        sql_path = PROJECT_ROOT / item["gold_sql_file"]
        sql = sql_path.read_text(encoding="utf-8") if sql_path.is_file() else ""
        merged = dict(item)
        merged["hardness"] = compute_hardness(sql) if sql else "hard"
        merged["verified"] = True
        merged["source"] = source
        merged.setdefault("category", f"legacy_{source}")
        inherited.append(merged)
    return inherited


def hardness_distribution(items: list[dict[str, Any]]) -> Counter:
    return Counter(item.get("hardness", "?") for item in items)


def print_summary(name: str, items: list[dict[str, Any]]) -> None:
    generated = [item for item in items if not item.get("verified", False)]
    print(f"\n{name}: {len(items)} questions "
          f"({len(items) - len(generated)} inherited verified, "
          f"{len(generated)} generated candidates)")
    distribution = hardness_distribution(items)
    for level in ("easy", "medium", "hard", "extra"):
        count = distribution.get(level, 0)
        print(f"  {level:<7} {count:>3}  ({count / len(items):.1%})")
    categories = Counter(item.get("category", "?") for item in items)
    print("  categories:", ", ".join(
        f"{category}={count}" for category, count in sorted(categories.items())
    ))


def main() -> int:
    args = parse_args()

    if not args.db_path.is_file():
        print(f"Error: database not found: {args.db_path}")
        return 1

    try:
        conn = duckdb.connect(str(args.db_path), read_only=True)
    except duckdb.Error as exc:
        print(f"Error: cannot open database read-only: {exc}")
        return 1

    try:
        energy_pools = collect_energy_pools(conn)
        finance_pools = collect_finance_pools(conn)
    finally:
        conn.close()

    print(
        f"Energy pools: {len(energy_pools.clean_days)} clean days, "
        f"{len(energy_pools.full_months)} full months, "
        f"full years {energy_pools.full_years}, "
        f"thresholds gap_median={energy_pools.gap_median}, "
        f"gap_p90={energy_pools.gap_p90}, "
        f"voltage_median={energy_pools.voltage_median}, "
        f"intensity_p90={energy_pools.intensity_p90}"
    )
    print(
        f"Finance pools: tickers={finance_pools.tickers}, "
        f"{len(finance_pools.months)} full months "
        f"({finance_pools.min_date}..{finance_pools.max_date})"
    )

    rng = random.Random(args.seed)
    energy_candidates = generate_energy_candidates(energy_pools, rng)
    finance_candidates = generate_finance_candidates(finance_pools, rng)

    energy_items = inherit_items(ENERGY_V2_PATH, source="v2") + [
        candidate.to_manifest_item("references/sql_gold")
        for candidate in energy_candidates
    ]
    finance_items = inherit_items(FINANCE_V1_PATH, source="finance_v1") + [
        candidate.to_manifest_item("references/sql_gold")
        for candidate in finance_candidates
    ]

    if len(energy_items) < MIN_ENERGY_TOTAL:
        print(f"Error: energy set has {len(energy_items)} < {MIN_ENERGY_TOTAL}")
        return 1
    if len(finance_items) < MIN_FINANCE_TOTAL:
        print(f"Error: finance set has {len(finance_items)} < {MIN_FINANCE_TOTAL}")
        return 1

    duplicate_ids = [
        item_id
        for item_id, count in Counter(
            item["id"] for item in energy_items + finance_items
        ).items()
        if count > 1
    ]
    if duplicate_ids:
        print(f"Error: duplicate question ids: {duplicate_ids}")
        return 1

    print_summary("energy_gold_questions_v3", energy_items)
    print_summary("finance_gold_questions_v2", finance_items)

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    for candidate in energy_candidates + finance_candidates:
        sql_path = SQL_DIR / f"{candidate.id}.sql"
        sql_path.write_text(candidate.sql + "\n", encoding="utf-8")

    ENERGY_V3_PATH.write_text(
        json.dumps(energy_items, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    FINANCE_V2_PATH.write_text(
        json.dumps(finance_items, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote {ENERGY_V3_PATH}")
    print(f"Wrote {FINANCE_V2_PATH}")
    print(
        f"Wrote {len(energy_candidates) + len(finance_candidates)} gold SQL "
        f"files under {SQL_DIR}"
    )
    print(
        "\nNext: python scripts/verify_gold_v3.py  (executes every gold SQL "
        "and cross-checks generated ones against pandas)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
