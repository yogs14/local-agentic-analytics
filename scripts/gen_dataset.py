"""CLI to generate the synthetic insight-narration fine-tuning dataset.

Examples
--------
Smoke run (offline, fake teacher, 5 examples)::

    python scripts/gen_dataset.py --smoke

Full run with the fake teacher::

    python scripts/gen_dataset.py --n 200 --seed 42

Full run against a real Ollama teacher (endpoint/model via TEACHER_* env vars)::

    python scripts/gen_dataset.py --n 200 --teacher ollama
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.finetune.concept_bank import (  # noqa: E402
    load_concept_bank_text,
    resolve_concept_bank_path,
)
from local_agentic_analytics.finetune.dataset_builder import (  # noqa: E402
    build_dataset,
    write_outputs,
)
from local_agentic_analytics.finetune.teacher_client import (  # noqa: E402
    AnthropicTeacherClient,
    DeepSeekTeacherClient,
    FakeTeacherClient,
    GeminiTeacherClient,
    OllamaTeacherClient,
    OpenRouterTeacherClient,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "finetune"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="Total variants to generate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument(
        "--teacher",
        choices=["fake", "ollama", "anthropic", "deepseek", "gemini", "openrouter"],
        default="fake",
        help="Teacher backend (default: fake, fully offline). 'anthropic' uses "
        "the Claude API (Opus 4.8) via ANTHROPIC_API_KEY; 'deepseek', 'gemini' "
        "and 'openrouter' use their OpenAI-compatible APIs via DEEPSEEK_API_KEY / "
        "GEMINI_API_KEY / OPENROUTER_API_KEY plus TEACHER_MODEL. Use 'openrouter' "
        "for the same-family teacher wave (google/gemma-4-31b-it, Qwen/Qwen3.5-27B).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Teacher model id. Takes precedence over TEACHER_MODEL in the "
        "environment and .env, so the model actually used is visible in the "
        "command itself (e.g. --model google/gemma-4-31b-it).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick offline run: 5 examples with the fake teacher, prints JSONL",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--concept-bank",
        type=Path,
        default=None,
        help="Path to concept bank markdown (default: auto-detect under data/finetune)",
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args(argv)


def _build_client(args: argparse.Namespace):
    if args.smoke or args.teacher == "fake":
        return FakeTeacherClient(seed=args.seed)

    # --model wins over TEACHER_MODEL so a wave cannot silently inherit the
    # model id left in .env by a previous wave.
    overrides = {"model": args.model} if args.model else {}

    if args.teacher == "anthropic":
        return AnthropicTeacherClient.from_env(**overrides)
    if args.teacher == "deepseek":
        return DeepSeekTeacherClient.from_env(**overrides)
    if args.teacher == "gemini":
        return GeminiTeacherClient.from_env(**overrides)
    if args.teacher == "openrouter":
        return OpenRouterTeacherClient.from_env(**overrides)
    return OllamaTeacherClient.from_env(**overrides)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    n = 5 if args.smoke else args.n
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR / "smoke" if args.smoke else DEFAULT_OUT_DIR

    concept_bank_text = load_concept_bank_text(args.concept_bank)
    bank_path = resolve_concept_bank_path(args.concept_bank)
    print(f"Concept bank: {bank_path if bank_path else '(none found)'}")

    client = _build_client(args)
    # Print the resolved model id: a wave must never be ambiguous about which
    # teacher produced it (the id also belongs in the run manifest).
    model_id = getattr(client, "model", "(n/a)")
    print(
        f"Teacher: {type(client).__name__} | model: {model_id} | "
        f"target variants: {n} | seed: {args.seed}"
    )

    result = build_dataset(
        client,
        n=n,
        seed=args.seed,
        concept_bank_text=concept_bank_text,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        val_fraction=args.val_fraction,
    )

    paths = write_outputs(out_dir, result)

    print("\n=== Summary ===")
    print(f"Generated : {result.total_generated}")
    print(f"Accepted  : {len(result.accepted)}")
    print(f"Rejected  : {len(result.rejected)}")
    print(f"Acceptance: {result.acceptance_rate:.1%}")
    print(f"Per-chart : {result.per_chart_accepted}")
    print(f"Train     : {len(result.train)} -> {paths['train']}")
    print(f"Val       : {len(result.val)} -> {paths['val']}")
    print(f"Rejected  : {paths['rejected']}")

    if args.smoke:
        print("\n=== train.jsonl ===")
        for example in result.train:
            print(json.dumps(example.to_record(), ensure_ascii=False, indent=2))
        print("\n=== val.jsonl ===")
        for example in result.val:
            print(json.dumps(example.to_record(), ensure_ascii=False, indent=2))
        if result.rejected:
            print("\n=== rejected ===")
            for rejection in result.rejected:
                print(f"- [{rejection.chart_id}] {'; '.join(rejection.reasons)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
