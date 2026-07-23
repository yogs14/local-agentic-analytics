"""Classify failed benchmark rows into the Fase 2 error taxonomy.

Reads sql_gold_eval / ablation_eval CSVs (and, when present, the per-model
run CSVs from the Fase 1 harness), classifies every failed row into one
exclusive category, and writes:

- reports/experiments/error_taxonomy.csv        (one row per failure)
- reports/experiments/error_taxonomy_summary.csv (category x group counts)

Examples:
    python scripts/run_error_taxonomy.py
    python scripts/run_error_taxonomy.py --inputs reports/experiments/ablation_eval.csv
    python scripts/run_error_taxonomy.py --no-execute
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.error_taxonomy import (
    TAXONOMY_CATEGORIES,
    TAXONOMY_ROW_COLUMNS,
    build_schema_vocabulary,
    classify_benchmark_rows,
    distribution_by_group,
)
from local_agentic_analytics.evaluation.model_benchmark import read_csv_rows
from local_agentic_analytics.evaluation.sql_gold_eval import (
    execute_sql,
    load_gold_questions,
    load_gold_sql,
    _load_default_duckdb_tool,
)


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "experiments" / "error_taxonomy.csv"
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT / "reports" / "experiments" / "error_taxonomy_summary.csv"
)

DEFAULT_QUESTION_FILES = (
    PROJECT_ROOT / "references" / "sql_gold" / "energy_gold_questions.json",
    PROJECT_ROOT / "references" / "sql_gold" / "energy_gold_questions_v2.json",
    PROJECT_ROOT / "references" / "sql_gold" / "finance_gold_questions.json",
)

DEFAULT_INPUT_CANDIDATES = (
    PROJECT_ROOT / "reports" / "experiments" / "sql_gold_eval.csv",
    PROJECT_ROOT / "reports" / "experiments" / "ablation_eval.csv",
)
MODEL_BENCHMARK_DIR = PROJECT_ROOT / "reports" / "experiments" / "model_benchmark"

_RUN_FILE_PATTERN = re.compile(r"^(?P<suite>.+)_run(?P<run>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify failed text-to-SQL rows into exclusive error categories "
            "(syntax, schema linking, unit conversion, ...)."
        )
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Benchmark CSVs to classify. Default: sql_gold_eval.csv, "
            "ablation_eval.csv, and every model_benchmark run CSV that exists."
        ),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Gold question JSONs used for gold SQL / expected unit lookup. "
            "Default: energy v1 + v2 + finance."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Per-failure classification CSV output.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Category x group distribution CSV output.",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help=(
            "Do not re-execute SQL via DuckDB for rows that lack stored "
            "results (disables the output_shape value-overlap check there)."
        ),
    )
    return parser.parse_args()


def discover_inputs() -> list[Path]:
    inputs = [path for path in DEFAULT_INPUT_CANDIDATES if path.is_file()]
    if MODEL_BENCHMARK_DIR.is_dir():
        inputs.extend(sorted(MODEL_BENCHMARK_DIR.glob("*/*_run*.csv")))
    return inputs


def input_metadata(path: Path) -> dict[str, Any]:
    """Derive suite/model/run metadata from a benchmark CSV path."""
    metadata: dict[str, Any] = {
        "source_file": _relative(path),
        "suite": path.stem,
        "model": "",
        "run_index": "",
    }
    try:
        parent = path.parent
        # Any harness output tree (model_benchmark, model_benchmark_v3, ...)
        # lays out <dir>/<model>/<suite>_run<i>.csv.
        if parent.parent.name.startswith("model_benchmark"):
            metadata["model"] = parent.name
            match = _RUN_FILE_PATTERN.match(path.stem)
            if match:
                metadata["suite"] = match.group("suite")
                metadata["run_index"] = match.group("run")
    except OSError:
        pass
    return metadata


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_question_map(paths: list[Path]) -> dict[str, dict[str, Any]]:
    questions_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            print(f"WARNING: questions file not found, skipped: {path}")
            continue
        try:
            for question in load_gold_questions(path):
                questions_by_id[str(question["id"])] = question
        except ValueError as exc:
            print(f"WARNING: could not load {path}: {exc}")
    return questions_by_id


def build_vocabulary(questions_by_id: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    gold_sqls: list[str] = []
    for question in questions_by_id.values():
        try:
            gold_sqls.append(load_gold_sql(question["gold_sql_file"]))
        except Exception:
            continue
    return build_schema_vocabulary(gold_sqls)


def make_sql_executor():
    """Wire a DuckDB executor; returns None when the database is unavailable.

    Probes the connection with a trivial query so a locked database (e.g. a
    benchmark still running in another process) fails LOUDLY here instead of
    silently disabling the result-based checks.
    """
    try:
        duckdb_tool = _load_default_duckdb_tool()
        probe = execute_sql(duckdb_tool, "SELECT 1")
        if not probe.success:
            raise RuntimeError(probe.error_message)
    except Exception as exc:
        print(
            "WARNING: DuckDB is unavailable (locked or missing); re-execution "
            "and output_shape/late-match checks are DISABLED for rows without "
            f"stored results. Reason: {exc}"
        )
        print(
            "         Re-run this script when no other process holds the "
            "database for a complete classification."
        )
        return None

    def executor(sql: str):
        result = execute_sql(duckdb_tool, sql)
        return result.success, result.result_text, result.error_message

    return executor


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(TAXONOMY_ROW_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in TAXONOMY_ROW_COLUMNS}
            )


def write_distribution(
    distribution: dict[str, dict[str, int]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "category", "count", "share"])
        for group in sorted(distribution):
            total = sum(distribution[group].values())
            for category in TAXONOMY_CATEGORIES:
                count = distribution[group].get(category, 0)
                if count == 0:
                    continue
                writer.writerow(
                    [group, category, count, f"{count / total:.4f}"]
                )


def print_distribution(distribution: dict[str, dict[str, int]]) -> None:
    if not distribution:
        print("\nNo failures were classified.")
        return

    print("\nError distribution per group:")
    for group in sorted(distribution):
        total = sum(distribution[group].values())
        print(f"\n[{group}] (n_failures={total})")
        for category in TAXONOMY_CATEGORIES:
            count = distribution[group].get(category, 0)
            if count == 0:
                continue
            print(f"  {category:<20} {count:>3}  ({count / total:.1%})")


def print_manual_review(classified: list[dict[str, Any]]) -> None:
    review_rows = [
        row for row in classified if bool(row.get("needs_manual_review"))
    ]
    if not review_rows:
        print("\nNo rows need manual review.")
        return

    print(f"\nRows needing manual review ({len(review_rows)}):")
    for row in review_rows:
        group = row.get("config") or row.get("model") or row.get("suite")
        print(
            f"  - {row['question_id']} [{group}] {row['category']}: "
            f"{row['evidence'][:100]}"
        )


def main() -> int:
    args = parse_args()

    input_paths = args.inputs if args.inputs else discover_inputs()
    input_paths = [path for path in input_paths if path.is_file()]
    if not input_paths:
        print(
            "Error: no input CSVs found. Run an evaluation first or pass "
            "--inputs explicitly."
        )
        return 1

    question_paths = (
        list(args.questions) if args.questions else list(DEFAULT_QUESTION_FILES)
    )
    questions_by_id = load_question_map(question_paths)
    if not questions_by_id:
        print("Error: no gold questions could be loaded.")
        return 1

    vocabulary = build_vocabulary(questions_by_id)
    sql_executor = None if args.no_execute else make_sql_executor()

    all_classified: list[dict[str, Any]] = []
    all_gold_errors: list[dict[str, Any]] = []
    all_late_matches: list[dict[str, Any]] = []
    for path in input_paths:
        # read_csv_rows raises the csv field limit: stored result columns can
        # exceed the 128 KiB default when agent SQL returns unaggregated rows.
        rows = read_csv_rows(path)
        classified, gold_errors, late_matches = classify_benchmark_rows(
            rows,
            questions_by_id,
            vocabulary=vocabulary,
            sql_executor=sql_executor,
            gold_sql_loader=load_gold_sql,
            metadata=input_metadata(path),
        )
        print(
            f"{_relative(path)}: {len(rows)} rows, "
            f"{len(classified)} failures classified, "
            f"{len(late_matches)} late row-set matches (not errors), "
            f"{len(gold_errors)} gold errors skipped"
        )
        all_classified.extend(classified)
        all_gold_errors.extend(gold_errors)
        all_late_matches.extend(late_matches)

    write_rows(all_classified + all_late_matches + all_gold_errors, args.output)
    distribution = distribution_by_group(all_classified)
    write_distribution(distribution, args.summary_output)

    print_distribution(distribution)
    print_manual_review(all_classified)
    if all_late_matches:
        print(
            f"\nNOTE: {len(all_late_matches)} rows fully match under the "
            "row-set comparison (result_match_full) and were NOT counted as "
            "errors; the legacy scalar metric had merely skipped them "
            "(labeled 'late_match_result_full' in the CSV)."
        )
    if all_gold_errors:
        print(
            f"\nNOTE: {len(all_gold_errors)} rows were skipped because the "
            "gold SQL itself failed (labeled 'gold_error' in the CSV)."
        )
    print(f"\nCSV:     {args.output}")
    print(f"Summary: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
