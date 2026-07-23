"""Assemble a fresh stat-block set and generate blinded narration pairs for judging.

Fase LLM-as-judge, data-prep step. This does NOT judge anything — it prepares
the input a separate (fresh) judge session will score.

What it does:
1. Samples >=30 FRESH energy stat blocks with `value_sampler.sample_variant`
   using a seed DIFFERENT from the fine-tune dataset seed (42), so the blocks
   are not the ones the narrators were trained/validated on. Every generated
   block is also checked verbatim against data/finetune/train.jsonl + val.jsonl
   and dropped on any exact match (leakage guard).
2. For each stat block, runs the SAME production narrator path the energy
   report uses (`InsightAgent` + `build_energy_finetune_prompt`) twice — once
   with the gemma fine-tune, once with the qwen fine-tune — via the
   OLLAMA model tags. Only the narrator model differs; the stat block, prompt,
   temperature (0.4), and max_tokens are identical.
3. Randomizes, per item, which narration is "A" and which is "B" (seeded), and
   writes:
   - reports/experiments/judge/narration_pairs.jsonl   (BLINDED — judge input)
   - reports/experiments/judge/blinding_key.json       (A/B -> model; NOT for the judge)

The judge (a fresh Claude Code session) reads only narration_pairs.jsonl.
`scripts/analyze_judge_results.py` uses blinding_key.json to unblind + run stats.

Examples:
    python scripts/export_report_narrations_for_judging.py --n-per-chart 6
    python scripts/export_report_narrations_for_judging.py --n-per-chart 8 --seed 20260713
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.agents.insight_agent import InsightAgent
from local_agentic_analytics.finetune.value_sampler import (
    ENERGY_CHARTS,
    format_stats_block,
    sample_variant,
)
from local_agentic_analytics.tools.ollama_tool import OllamaTool


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "experiments" / "judge"
TRAIN_PATH = PROJECT_ROOT / "data" / "finetune" / "train.jsonl"
VAL_PATH = PROJECT_ROOT / "data" / "finetune" / "val.jsonl"

# The two fine-tuned narrators under comparison. Keys are neutral condition
# labels used internally; the judge never sees them.
GEMMA_TAG = "gemma2-energy-insight:v3"
QWEN_TAG = "qwen25-energy-insight:v1"

# Dataset seed was 42; use a different one so sampled variants don't coincide.
DEFAULT_SEED = 20260713
NARRATION_TEMPERATURE = 0.4
NARRATION_MAX_TOKENS = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate blinded narration pairs (gemma vs qwen fine-tune) over a "
            "fresh, leakage-checked stat-block set for LLM-as-judge scoring."
        )
    )
    parser.add_argument(
        "--n-per-chart",
        type=int,
        default=6,
        help="Stat blocks to sample per chart type (6 charts; default 6 -> 36).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (must differ from the dataset seed 42; default {DEFAULT_SEED}).",
    )
    # Two arbitrary narrators are compared per run. Defaults reproduce the
    # original cross-family wave (gemma vs qwen); the same-family wave passes
    # its own tags/labels, e.g.
    #   --left-label gemma_cross --left-tag  gemma2-energy-insight:v3
    #   --right-label gemma_fam  --right-tag gemma2-energy-insight-fam:v1
    parser.add_argument(
        "--left-label",
        default="gemma",
        help="Internal label for the first narrator (never shown to the judge).",
    )
    parser.add_argument(
        "--right-label",
        default="qwen",
        help="Internal label for the second narrator (never shown to the judge).",
    )
    parser.add_argument(
        "--left-tag",
        "--gemma-tag",
        dest="left_tag",
        default=GEMMA_TAG,
        help=f"Ollama tag for the first narrator (default {GEMMA_TAG}).",
    )
    parser.add_argument(
        "--right-tag",
        "--qwen-tag",
        dest="right_tag",
        default=QWEN_TAG,
        help=f"Ollama tag for the second narrator (default {QWEN_TAG}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for narration_pairs.jsonl + blinding_key.json.",
    )
    return parser.parse_args()


def load_leakage_blocklist() -> set[str]:
    """Every `input` stat block already used in train/val (exact-match guard)."""
    blocks: set[str] = set()
    for path in (TRAIN_PATH, VAL_PATH):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            block = record.get("input")
            if isinstance(block, str):
                blocks.add(block.strip())
    return blocks


def assemble_stat_blocks(
    n_per_chart: int,
    seed: int,
    blocklist: set[str],
) -> list[dict[str, Any]]:
    """Sample fresh, non-leaking stat blocks across all energy chart types."""
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    counter = 1
    for chart_id in ENERGY_CHARTS:
        produced = 0
        attempts = 0
        # Oversample attempts so leakage drops don't starve a chart.
        while produced < n_per_chart and attempts < n_per_chart * 20:
            attempts += 1
            variant = sample_variant(chart_id, rng)
            block = format_stats_block(variant)
            if block.strip() in blocklist:
                continue
            stats = {key: value for key, value in variant.items() if key != "chart_id"}
            items.append(
                {
                    "item_id": f"J{counter:03d}",
                    "chart_id": str(variant.get("chart_id", chart_id)),
                    "stat_block": block,
                    "chart_context": {"chart_id": variant.get("chart_id", chart_id),
                                       "stats": stats},
                }
            )
            counter += 1
            produced += 1
        if produced < n_per_chart:
            print(
                f"WARNING: only produced {produced}/{n_per_chart} for {chart_id} "
                "after oversampling (leakage drops)."
            )
    return items


def build_narrator(tag: str) -> InsightAgent:
    """Production narrator path pinned to one Ollama tag (same code as the report)."""
    tool = OllamaTool.from_config(section="insight_model")
    tool.model = tag
    return InsightAgent.for_energy_finetune(
        ollama_tool=tool,
        temperature=NARRATION_TEMPERATURE,
        max_tokens=NARRATION_MAX_TOKENS,
    )


def narrate(agent: InsightAgent, chart_context: dict[str, Any]) -> str:
    try:
        return agent.generate_insight(chart_context)
    except Exception as exc:  # noqa: BLE001 - record failures, don't abort the batch
        return f"[GENERATION_FAILED: {type(exc).__name__}: {exc}]"


def main() -> int:
    args = parse_args()

    if args.seed == 42:
        print("Error: --seed must differ from the dataset seed (42) to avoid leakage.")
        return 1
    if args.n_per_chart < 1:
        print("Error: --n-per-chart must be >= 1")
        return 1

    blocklist = load_leakage_blocklist()
    print(f"Leakage blocklist: {len(blocklist)} train/val stat blocks loaded.")

    items = assemble_stat_blocks(args.n_per_chart, args.seed, blocklist)
    print(f"Assembled {len(items)} fresh stat blocks across {len(ENERGY_CHARTS)} charts.")
    if len(items) < 30:
        print(
            f"WARNING: only {len(items)} stat blocks (< 30). Increase --n-per-chart "
            "for stronger statistical power."
        )

    if args.left_label == args.right_label:
        raise SystemExit("--left-label and --right-label must differ")
    if args.left_tag == args.right_tag:
        raise SystemExit("--left-tag and --right-tag must differ")

    print(
        f"Building narrators: {args.left_label}={args.left_tag}, "
        f"{args.right_label}={args.right_tag}"
    )
    # OLLAMA_MODEL is irrelevant here (narrators use their own tags), but keep the
    # env clean so nothing downstream picks up a stale value.
    os.environ.pop("OLLAMA_INSIGHT_MODEL", None)
    left = build_narrator(args.left_tag)
    right = build_narrator(args.right_tag)

    order_rng = random.Random(args.seed + 1)
    pairs: list[dict[str, Any]] = []
    key: dict[str, dict[str, str]] = {}

    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['item_id']} ({item['chart_id']})")
        left_text = narrate(left, item["chart_context"])
        right_text = narrate(right, item["chart_context"])

        # Randomize A/B so the judge can't infer the model from position.
        if order_rng.random() < 0.5:
            a_model, a_text = args.left_label, left_text
            b_model, b_text = args.right_label, right_text
        else:
            a_model, a_text = args.right_label, right_text
            b_model, b_text = args.left_label, left_text

        pairs.append(
            {
                "item_id": item["item_id"],
                "chart_id": item["chart_id"],
                "stat_block": item["stat_block"],
                "narration_A": a_text,
                "narration_B": b_text,
            }
        )
        key[item["item_id"]] = {"A": a_model, "B": b_model}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "narration_pairs.jsonl"
    key_path = output_dir / "blinding_key.json"

    with pairs_path.open("w", encoding="utf-8") as file:
        for pair in pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")
    key_path.write_text(
        json.dumps(
            {
                # Generic label/tag pair; kept alongside the legacy gemma/qwen
                # keys so older analysis runs and reports still resolve.
                "left_label": args.left_label,
                "right_label": args.right_label,
                "left_tag": args.left_tag,
                "right_tag": args.right_tag,
                "gemma_tag": args.left_tag,
                "qwen_tag": args.right_tag,
                "seed": args.seed,
                "mapping": key,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    failed = sum(
        1
        for pair in pairs
        if pair["narration_A"].startswith("[GENERATION_FAILED")
        or pair["narration_B"].startswith("[GENERATION_FAILED")
    )
    print(f"\nWrote {len(pairs)} blinded pairs -> {pairs_path}")
    print(f"Wrote blinding key            -> {key_path}")
    if failed:
        print(f"WARNING: {failed} pairs had a failed generation — inspect before judging.")
    print(
        "\nNext: open a NEW Claude Code session and run the judge prompt "
        "(docs/llm_judge_prompt.md). It reads ONLY narration_pairs.jsonl."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
