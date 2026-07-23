"""Benchmark registered Ollama models apple-to-apple on existing eval suites.

Examples (run from the repo root):
    python scripts/run_model_benchmark.py --models all --suite sql_gold_v2 --dry-run
    python scripts/run_model_benchmark.py --models gemma2_2b,qwen2.5_1.5b --suite sql_gold_v2
    python scripts/run_model_benchmark.py --models qwen2.5_3b --suite ablation --repeats 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.model_benchmark import (
    DEFAULT_OUTPUT_DIR,
    SUITES,
    build_dry_run_report,
    build_environment_info,
    hash_gold_dataset,
    is_tag_available,
    list_local_ollama_tags,
    list_model_keys,
    load_model_benchmark_config,
    load_suite_questions,
    pull_ollama_tag,
    resolve_ollama_base_url,
    run_model_suite,
    validate_locked_variables,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one evaluation suite against one or more models from "
            "configs/models/, holding every other variable fixed."
        )
    )
    parser.add_argument(
        "--models",
        required=True,
        help=(
            "Comma-separated model keys from configs/models/ (file stems), "
            "or 'all' for every registered model."
        ),
    )
    parser.add_argument(
        "--suite",
        required=True,
        choices=SUITES,
        help="Evaluation suite to run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of full suite repetitions per model (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate configs, dataset hashes, and Ollama availability "
            "without running any model."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of questions per run (smoke tests).",
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=None,
        help="Optional gold questions JSON override for the chosen suite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: reports/experiments/model_benchmark).",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Fail instead of pulling when a model tag is missing locally.",
    )
    return parser.parse_args()


def resolve_model_keys(models_arg: str) -> list[str]:
    if models_arg.strip().lower() == "all":
        keys = list_model_keys()
        if not keys:
            raise ValueError("No model configs found in configs/models/")
        return keys
    keys = [key.strip() for key in models_arg.split(",") if key.strip()]
    if not keys:
        raise ValueError("--models must name at least one model key")
    return keys


def print_dry_run_report(report: dict[str, Any]) -> bool:
    """Print the dry-run report; returns True when everything is runnable."""
    print("=== Model benchmark dry run ===")
    print(f"- suite:            {report['suite']}")
    print(f"- repeats:          {report['repeats']}")
    print(f"- ollama_base_url:  {report['ollama_base_url']}")
    print(f"- ollama_reachable: {report['ollama_reachable']}")
    if report.get("ollama_version"):
        print(f"- ollama_version:   {report['ollama_version']}")
    commit = report.get("environment", {}).get("commit")
    print(f"- commit:           {commit or 'unknown'}")
    hardware = report.get("environment", {}).get("hardware", {})
    print(
        f"- hardware:         {hardware.get('platform', '?')} | "
        f"RAM {hardware.get('ram_total_gb', '?')} GB | "
        f"GPU {hardware.get('gpu') or 'none detected'}"
    )

    ok = True
    if report.get("dataset_error"):
        ok = False
        print(f"\nDATASET ERROR: {report['dataset_error']}")
    else:
        dataset = report.get("dataset", {})
        print(f"\nDataset: {dataset.get('questions_path', '?')}")
        print(f"- n_questions:              {dataset.get('n_questions', '?')}")
        print(f"- questions_sha256:         {dataset.get('questions_sha256', '?')}")
        print(
            "- gold_sql_combined_sha256: "
            f"{dataset.get('gold_sql_combined_sha256', '?')}"
        )
        missing = dataset.get("missing_gold_sql_files") or []
        if missing:
            ok = False
            print(f"- MISSING gold SQL files:  {missing}")

    print("\nModels:")
    for entry in report.get("models", []):
        key = entry.get("model_key", "?")
        if not entry.get("config_ok"):
            ok = False
            print(f"  [FAIL] {key}: {entry.get('error', 'config error')}")
            continue

        problems = entry.get("locked_variable_problems") or []
        available = entry.get("tag_available_locally")
        if available is None:
            availability = "unknown (Ollama unreachable)"
        elif available:
            availability = "available locally"
        else:
            availability = "NOT local (will be pulled)"

        status = "OK  " if not problems else "FAIL"
        if problems:
            ok = False
        print(f"  [{status}] {key} -> {entry.get('ollama_tag')} ({availability})")
        for problem in problems:
            print(f"         - {problem}")

    print(
        "\nDry run result: "
        + ("READY (locked variables consistent)" if ok else "PROBLEMS FOUND")
    )
    if report["ollama_reachable"] is False:
        print(
            "NOTE: Ollama is not reachable; a real run needs the Ollama "
            "service started and the tags pulled."
        )
    return ok


def main() -> int:
    args = parse_args()

    if args.repeats < 1:
        print("Error: --repeats must be greater than 0")
        return 1
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be greater than 0")
        return 1

    try:
        model_keys = resolve_model_keys(args.models)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if args.dry_run:
        report = build_dry_run_report(
            model_keys,
            args.suite,
            args.repeats,
            questions_path=args.questions_path,
            limit=args.limit,
        )
        return 0 if print_dry_run_report(report) else 1

    base_url = resolve_ollama_base_url()
    local_tags = list_local_ollama_tags(base_url)
    if local_tags is None:
        print(
            f"Error: Ollama is not reachable at {base_url}. Start Ollama, or "
            "use --dry-run to validate the setup without models."
        )
        return 1

    try:
        questions_path, questions = load_suite_questions(
            args.suite, args.questions_path, args.limit
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    dataset_info = hash_gold_dataset(questions_path, questions)

    exit_code = 0
    for key in model_keys:
        try:
            config = load_model_benchmark_config(key)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[{key}] SKIPPED: {exc}")
            exit_code = 1
            continue

        problems = validate_locked_variables(config)
        if problems:
            print(f"[{key}] SKIPPED: locked variables are not apple-to-apple:")
            for problem in problems:
                print(f"  - {problem}")
            exit_code = 1
            continue

        if not is_tag_available(config.ollama_tag, local_tags):
            if args.no_pull:
                print(
                    f"[{key}] SKIPPED: tag '{config.ollama_tag}' is not local "
                    "and --no-pull was given."
                )
                exit_code = 1
                continue
            if not pull_ollama_tag(config.ollama_tag):
                print(f"[{key}] SKIPPED: could not pull '{config.ollama_tag}'.")
                exit_code = 1
                continue
            refreshed = list_local_ollama_tags(base_url)
            if refreshed is not None:
                local_tags = refreshed

        suite_entry = run_model_suite(
            model_config=config,
            suite=args.suite,
            questions=questions,
            repeats=args.repeats,
            output_dir=args.output_dir,
        )
        environment = build_environment_info(base_url, config.ollama_tag)
        manifest_path = write_manifest(
            model_dir=Path(args.output_dir) / config.key,
            model_config=config,
            suite=args.suite,
            suite_entry=suite_entry,
            dataset_info=dataset_info,
            environment=environment,
        )
        print(f"[{key}] done. Manifest: {manifest_path}")

    print(
        "\nNext: python scripts/summarize_model_benchmark.py "
        f"--input-dir \"{args.output_dir}\""
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
