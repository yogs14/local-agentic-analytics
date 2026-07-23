"""Compare prompting strategies (zero-shot / few-shot / DIN-SQL decomposed).

Runs the selected strategies on ONE model over a gold set, apple-to-apple
(same schema context, temperature 0.0, same questions), and writes results in
the same summary format as the model benchmark:

- reports/experiments/prompting_comparison/<model>/rows.csv
- reports/experiments/prompting_comparison/<model>/summary.csv + .md
  (per-strategy rates + bootstrap 95% CI + exact McNemar vs zero_shot)

Examples:
    python scripts/run_prompting_comparison.py --model qwen2.5_1.5b --dry-run
    python scripts/run_prompting_comparison.py --model qwen2.5_1.5b
    python scripts/run_prompting_comparison.py --model gemma2_2b --strategies zero_shot,decomposed --limit 10
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.model_benchmark import (
    SUITE_SQL_GOLD_V2,
    is_tag_available,
    list_local_ollama_tags,
    load_model_benchmark_config,
    render_summary_markdown,
    resolve_ollama_base_url,
    summarize_run_rows,
    validate_locked_variables,
    write_summary_csv,
)
from local_agentic_analytics.evaluation.prompting_comparison import (
    PROMPTING_COLUMNS,
    STRATEGIES,
    build_schema_context,
    run_prompting_comparison,
)
from local_agentic_analytics.evaluation.sql_gold_eval import (
    load_gold_questions,
    _load_default_duckdb_tool,
)
from local_agentic_analytics.evaluation.statistics import (
    bootstrap_rate_ci,
    mcnemar_vs_baseline,
)
from local_agentic_analytics.evaluation.model_benchmark import mean_std
from local_agentic_analytics.tools.ollama_tool import OllamaTool


DEFAULT_QUESTIONS_PATH = (
    PROJECT_ROOT / "references" / "sql_gold" / "energy_gold_questions_v3.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "experiments" / "prompting_comparison"
)
BASELINE_STRATEGY = "zero_shot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare zero_shot / few_shot_static / decomposed prompting on "
            "one model over a gold set."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model key from configs/models/ (e.g. qwen2.5_1.5b).",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help=f"Comma-separated strategies (default: all of {STRATEGIES}).",
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Gold questions JSON (default: energy v3).",
    )
    parser.add_argument(
        "--domain",
        default="energy",
        choices=("energy", "finance"),
        help="Schema/domain for prompts (default: energy).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max questions per strategy (smoke tests).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/model/questions without generating.",
    )
    return parser.parse_args()


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(PROMPTING_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in PROMPTING_COLUMNS}
            )


def _stringify(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Match the CSV string form the shared summarizers expect."""
    return [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]


def build_summary(
    rows: list[dict[str, Any]],
    model_key: str,
    strategies: list[str],
) -> list[dict[str, Any]]:
    string_rows = _stringify(rows)
    # Group by strategy via the ablation-style "config" column.
    metrics = summarize_run_rows(string_rows, suite="ablation")

    outcomes: dict[str, dict[str, dict[str, bool]]] = {}
    for row in string_rows:
        strategy = row.get("config", "")
        outcomes.setdefault(strategy, {})[row.get("question_id", "")] = {
            "exec": row.get("execution_success") == "True",
            "full": (
                row.get("numeric_match") == "True"
                or row.get("result_match_full") == "True"
            ),
        }

    summary_rows: list[dict[str, Any]] = []
    baseline = outcomes.get(BASELINE_STRATEGY)
    for strategy in strategies:
        strategy_metrics = metrics.get(strategy)
        if strategy_metrics is None:
            continue
        strategy_outcomes = outcomes.get(strategy, {})
        exec_ci = bootstrap_rate_ci(
            [item["exec"] for item in strategy_outcomes.values()]
        )
        full_ci = bootstrap_rate_ci(
            [item["full"] for item in strategy_outcomes.values()]
        )
        row: dict[str, Any] = {
            "model": model_key,
            "suite": "prompting",
            "config": strategy,
            "runs": 1,
            "n_questions": strategy_metrics["n_questions"],
            "execution_success_mean": strategy_metrics["execution_success_rate"],
            "execution_success_std": 0.0,
            "numeric_match_compared_mean": strategy_metrics[
                "numeric_match_compared_rate"
            ],
            "numeric_match_compared_std": 0.0,
            "numeric_match_total_mean": strategy_metrics[
                "numeric_match_total_rate"
            ],
            "numeric_match_total_std": 0.0,
            "unit_correct_mean": None,
            "unit_correct_std": None,
            "full_accuracy_mean": strategy_metrics["full_accuracy_rate"],
            "full_accuracy_std": 0.0,
            "latency_p50_mean": strategy_metrics["latency_p50"],
            "latency_p50_std": 0.0,
            "latency_p95_mean": strategy_metrics["latency_p95"],
            "latency_p95_std": 0.0,
            "tokens_per_second_mean": strategy_metrics["tokens_per_second_mean"],
            "tokens_per_second_std": 0.0,
            "peak_rss_mb_max": None,
            "model_vram_mb_max": None,
            "execution_success_ci_low": exec_ci["ci_low"],
            "execution_success_ci_high": exec_ci["ci_high"],
            "full_accuracy_ci_low": full_ci["ci_low"],
            "full_accuracy_ci_high": full_ci["ci_high"],
            "baseline_model": BASELINE_STRATEGY,
            "p_exec_vs_baseline": None,
            "p_full_vs_baseline": None,
        }
        if baseline is not None and strategy != BASELINE_STRATEGY:
            row["p_exec_vs_baseline"] = mcnemar_vs_baseline(
                {qid: item["exec"] for qid, item in strategy_outcomes.items()},
                {qid: item["exec"] for qid, item in baseline.items()},
            )["p_value"]
            row["p_full_vs_baseline"] = mcnemar_vs_baseline(
                {qid: item["full"] for qid, item in strategy_outcomes.items()},
                {qid: item["full"] for qid, item in baseline.items()},
            )["p_value"]
        summary_rows.append(row)
    return summary_rows


def main() -> int:
    args = parse_args()

    strategies = [name.strip() for name in args.strategies.split(",") if name.strip()]
    unknown = [name for name in strategies if name not in STRATEGIES]
    if unknown:
        print(f"Error: unknown strategies {unknown} (valid: {STRATEGIES})")
        return 1
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be greater than 0")
        return 1

    try:
        config = load_model_benchmark_config(args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    problems = validate_locked_variables(config)
    if problems:
        print("Error: locked variables are not apple-to-apple:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    try:
        questions = load_gold_questions(args.questions_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    if args.limit is not None:
        questions = questions[: args.limit]

    base_url = resolve_ollama_base_url()
    local_tags = list_local_ollama_tags(base_url)

    if args.dry_run:
        available = (
            None if local_tags is None
            else is_tag_available(config.ollama_tag, local_tags)
        )
        print("=== Prompting comparison dry run ===")
        print(f"- model:      {config.key} -> {config.ollama_tag}")
        print(f"- strategies: {strategies}")
        print(f"- questions:  {len(questions)} ({args.questions_path})")
        print(f"- domain:     {args.domain}")
        print(f"- ollama:     reachable={local_tags is not None}, "
              f"tag_available={available}")
        total_calls = sum(
            len(questions) * (3 if name == "decomposed" else 1)
            for name in strategies
        )
        print(f"- total LLM calls if run: {total_calls}")
        return 0

    if local_tags is None:
        print(f"Error: Ollama is not reachable at {base_url}.")
        return 1
    if not is_tag_available(config.ollama_tag, local_tags):
        print(
            f"Error: tag '{config.ollama_tag}' is not available locally. "
            f"Run: ollama pull {config.ollama_tag}"
        )
        return 1

    # Model selection via the same env mechanism as the whole pipeline.
    os.environ["OLLAMA_MODEL"] = config.ollama_tag
    ollama_tool = OllamaTool.from_config()
    duckdb_tool = _load_default_duckdb_tool()
    schema_context = build_schema_context(args.domain)

    rows = run_prompting_comparison(
        questions,
        tuple(strategies),
        ollama_tool,
        duckdb_tool,
        domain=args.domain,
        schema_context=schema_context,
    )

    model_dir = args.output_dir / config.key
    rows_path = model_dir / "rows.csv"
    _write_rows(rows, rows_path)

    summary_rows = build_summary(rows, config.key, strategies)
    summary_csv = write_summary_csv(summary_rows, model_dir / "summary.csv")
    markdown = render_summary_markdown(summary_rows)
    (model_dir / "summary.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"Rows:    {rows_path}")
    print(f"Summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
