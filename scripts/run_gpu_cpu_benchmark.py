"""Benchmark the local quantized model on GPU vs CPU on a fixed question set.

CPU stays the committed default (configs/model.yaml: num_gpu=0); this script
only overrides num_gpu at runtime to exercise the GPU path.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.batch_eval import (
    DEFAULT_QUESTIONS_PATH,
    load_questions,
)
from local_agentic_analytics.evaluation.gpu_cpu_benchmark import (
    DEFAULT_CSV_OUTPUT_PATH,
    DEFAULT_GPU_NUM_GPU,
    DEFAULT_SUMMARY_OUTPUT_PATH,
    default_modes,
    run_gpu_cpu_benchmark_repeated,
    write_gpu_cpu_benchmark_rows,
    write_gpu_cpu_benchmark_summary,
)

GPU_NUM_GPU_ENV_VAR = "GPU_BENCHMARK_NUM_GPU"


def _default_gpu_num_gpu() -> int:
    raw = os.getenv(GPU_NUM_GPU_ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_GPU_NUM_GPU
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_GPU_NUM_GPU
    return value if value >= 0 else DEFAULT_GPU_NUM_GPU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark GPU vs CPU for the local model on a fixed question set. "
            "num_gpu is overridden at runtime; configs/model.yaml is untouched."
        )
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to the evaluation questions JSON.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_OUTPUT_PATH,
        help="Path to the per-question per-mode CSV output.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_PATH,
        help="Path to the summary JSON output.",
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=_default_gpu_num_gpu(),
        help=(
            "Layers to offload for GPU mode (high = all that fit). "
            f"Overridable via the {GPU_NUM_GPU_ENV_VAR} env var."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of questions per mode (for smoke runs).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Number of times to run the full benchmark per mode. Latency and "
            "tokens/sec are aggregated (mean/sd/min/max) across runs."
        ),
    )
    return parser.parse_args()


def print_summary(summary: dict[str, Any]) -> None:
    modes = summary.get("modes", {})
    print("\nGPU vs CPU benchmark summary:")
    header = (
        f"{'mode':<8} {'label':<28} {'success':>8} {'avg_lat_s':>10} "
        f"{'tok/s':>8} {'vram_mb':>9} {'vram_delta':>11} {'rss_mb':>9} "
        f"{'fit_4gb':>8}"
    )
    print(header)
    print("-" * len(header))
    for mode_name, stats in modes.items():
        print(
            f"{mode_name:<8} {str(stats['label']):<28} "
            f"{stats['success_rate']:>7.1%} "
            f"{stats['avg_total_latency']:>10.2f} "
            f"{stats['avg_tokens_per_second']:>8.1f} "
            f"{_fmt(stats['vram_mb']):>9} "
            f"{_fmt(stats['vram_delta_mb']):>11} "
            f"{_fmt(stats['peak_rss_mb']):>9} "
            f"{str(stats['fit_in_4gb']):>8}"
        )

    print(
        "\nVRAM model-attributable from `ollama ps` (SIZE x GPU-fraction); "
        "vram_delta is the global memory.used baseline->peak cross-check."
    )
    for mode_name, stats in modes.items():
        print(
            f"- {mode_name}: processor_split='{stats.get('processor_split', '')}', "
            f"vram_mb={_fmt(stats['vram_mb'])} (ollama ps), "
            f"vram_delta_mb={_fmt(stats['vram_delta_mb'])} (global delta), "
            f"compute_apps={_fmt(stats.get('compute_apps_vram_mb'))} "
            f"(best-effort, N/A under WDDM)"
        )

    repeat = summary.get("repeat", 1)
    if repeat:
        print(f"\nAggregated across {repeat} run(s) per mode (mean +/- sd [min, max]):")
        for mode_name, stats in modes.items():
            latency = stats.get("latency_stats", {})
            tps = stats.get("tokens_per_second_stats", {})
            print(
                f"- {mode_name}: "
                f"latency_s={_fmt_stat(latency)} | tok/s={_fmt_stat(tps)}"
            )


def _fmt_stat(stat: dict[str, Any]) -> str:
    mean = stat.get("mean")
    if mean is None:
        return "n/a"
    sd = stat.get("sd") or 0.0
    return (
        f"{float(mean):.2f} +/- {float(sd):.2f} "
        f"[{float(stat['min']):.2f}, {float(stat['max']):.2f}]"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.0f}"


def main() -> int:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be greater than 0")
        return 1
    if args.num_gpu < 0:
        print("Error: --num-gpu must be non-negative")
        return 1
    if args.repeat < 1:
        print("Error: --repeat must be greater than 0")
        return 1

    try:
        questions = load_questions(args.questions_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if args.limit is not None:
        questions = questions[: args.limit]

    rows, summary = run_gpu_cpu_benchmark_repeated(
        questions,
        repeat=args.repeat,
        modes=default_modes(args.num_gpu),
    )

    write_gpu_cpu_benchmark_rows(rows, args.csv_path)
    write_gpu_cpu_benchmark_summary(summary, args.summary_path)

    print_summary(summary)
    print(f"\nCSV:     {args.csv_path}")
    print(f"Summary: {args.summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
