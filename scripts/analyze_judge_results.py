"""Unblind LLM-as-judge scores and run the statistics for the narrator A/B.

Reads:
- reports/experiments/judge/judge_scores.csv   (produced by the judge session)
- reports/experiments/judge/blinding_key.json  (A/B -> gemma/qwen mapping)

Produces per-model absolute-score summaries (mean +/- std + bootstrap 95% CI
per rubric dimension) and pairwise preference results (win rate + exact sign
test / McNemar on discordant pairs), written to judge_summary.csv / .md.

Expected judge_scores.csv columns (one row per item):
    item_id,
    A_faithfulness, A_unit_correctness, A_domain_richness, A_coherence, A_fluency_id,
    B_faithfulness, B_unit_correctness, B_domain_richness, B_coherence, B_fluency_id,
    pref_semantic, pref_coherence          (each: A | B | tie)
    note                                    (optional free text)

Example:
    python scripts/analyze_judge_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.statistics import mcnemar_exact


DEFAULT_DIR = PROJECT_ROOT / "reports" / "experiments" / "judge"
DIMENSIONS = (
    "faithfulness",
    "unit_correctness",
    "domain_richness",
    "coherence",
    "fluency_id",
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unblind and analyze LLM-as-judge narrator A/B scores."
    )
    parser.add_argument(
        "--judge-dir",
        type=Path,
        default=DEFAULT_DIR,
        help="Directory holding judge_scores.csv + blinding_key.json.",
    )
    parser.add_argument(
        "--scores-name",
        default="judge_scores.csv",
        help="Judge output CSV filename (default judge_scores.csv).",
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number


def _norm_pref(value: Any) -> str | None:
    text = str(value).strip().upper()
    if text in ("A", "B"):
        return text
    if text in ("TIE", "SERI", "="):
        return "tie"
    return None


def bootstrap_mean_ci(
    values: list[float],
    n_boot: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float | None, float | None, float | None]:
    """Percentile bootstrap 95% CI for the mean of 1-5 scores."""
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    rng = np.random.default_rng(seed)
    resampled = array[rng.integers(0, array.size, size=(n_boot, array.size))].mean(axis=1)
    low, high = np.quantile(resampled, [0.025, 0.975])
    return mean, float(low), float(high)


def load_key(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "mapping" not in data:
        raise ValueError("blinding_key.json missing 'mapping'")
    return data


def main() -> int:
    args = parse_args()
    scores_path = args.judge_dir / args.scores_name
    key_path = args.judge_dir / "blinding_key.json"

    if not scores_path.is_file():
        print(f"Error: judge scores not found: {scores_path}")
        print("Run the judge session first (writes judge_scores.csv).")
        return 1
    if not key_path.is_file():
        print(f"Error: blinding key not found: {key_path}")
        return 1

    key = load_key(key_path)
    mapping = key["mapping"]

    with scores_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        print("Error: judge_scores.csv is empty.")
        return 1

    # Labels come from the blinding key so any narrator pair can be analysed
    # (cross-family wave: gemma/qwen; same-family wave: *_cross vs *_fam).
    left_label = str(key.get("left_label", "gemma"))
    right_label = str(key.get("right_label", "qwen"))

    # model -> dimension -> list of scores
    per_model: dict[str, dict[str, list[float]]] = {
        left_label: {dim: [] for dim in DIMENSIONS},
        right_label: {dim: [] for dim in DIMENSIONS},
    }
    pairwise = {
        "semantic": {left_label: 0, right_label: 0, "tie": 0},
        "coherence": {left_label: 0, right_label: 0, "tie": 0},
    }
    n_items = 0
    skipped: list[str] = []

    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        item_key = mapping.get(item_id)
        if not item_key:
            skipped.append(item_id or "(blank)")
            continue
        n_items += 1

        for slot in ("A", "B"):
            model = item_key[slot]
            for dim in DIMENSIONS:
                score = _to_float(row.get(f"{slot}_{dim}"))
                if score is not None:
                    per_model[model][dim].append(score)

        for aspect, column in (("semantic", "pref_semantic"),
                               ("coherence", "pref_coherence")):
            pref = _norm_pref(row.get(column))
            if pref == "tie":
                pairwise[aspect]["tie"] += 1
            elif pref in ("A", "B"):
                pairwise[aspect][item_key[pref]] += 1

    # ---- absolute scores per model per dimension ----
    abs_rows: list[dict[str, Any]] = []
    for model in (left_label, right_label):
        for dim in DIMENSIONS:
            values = per_model[model][dim]
            mean, low, high = bootstrap_mean_ci(values)
            std = statistics.stdev(values) if len(values) >= 2 else 0.0
            abs_rows.append(
                {
                    "model": model,
                    "dimension": dim,
                    "n": len(values),
                    "mean": mean,
                    "std": std if values else None,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    # ---- pairwise ----
    pair_rows: list[dict[str, Any]] = []
    for aspect in ("semantic", "coherence"):
        counts = pairwise[aspect]
        test = mcnemar_exact(counts[left_label], counts[right_label])
        pair_rows.append(
            {
                "aspect": aspect,
                "gemma_wins": counts[left_label],
                "qwen_wins": counts[right_label],
                "ties": counts["tie"],
                "p_value": test["p_value"],
            }
        )

    # ---- write outputs ----
    csv_path = args.judge_dir / "judge_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["section", "key", "n", "mean_or_wins", "std_or_ties",
                         "ci_low_or_p", "ci_high"])
        for row in abs_rows:
            writer.writerow([
                "absolute", f"{row['model']}/{row['dimension']}", row["n"],
                _fmt(row["mean"]), _fmt(row["std"]), _fmt(row["ci_low"]),
                _fmt(row["ci_high"]),
            ])
        for row in pair_rows:
            writer.writerow([
                "pairwise", row["aspect"], n_items,
                f"gemma={row['gemma_wins']};qwen={row['qwen_wins']}",
                row["ties"], _fmt(row["p_value"]), "",
            ])

    md_path = args.judge_dir / "judge_summary.md"
    md_path.write_text(_render_markdown(key, n_items, abs_rows, pair_rows, skipped),
                       encoding="utf-8")

    print(_render_markdown(key, n_items, abs_rows, pair_rows, skipped))
    print(f"\nCSV:      {csv_path}")
    print(f"Markdown: {md_path}")
    if skipped:
        print(f"WARNING: {len(skipped)} judged items had no blinding-key entry: {skipped}")
    return 0


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_markdown(
    key: dict[str, Any],
    n_items: int,
    abs_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    skipped: list[str],
) -> str:
    left_label = str(key.get("left_label", "gemma"))
    right_label = str(key.get("right_label", "qwen"))
    left_tag = key.get("left_tag", key.get("gemma_tag"))
    right_tag = key.get("right_tag", key.get("qwen_tag"))

    lines = [
        f"# LLM-as-Judge Summary — Narrator A/B ({left_label} vs {right_label})",
        "",
        f"- {left_label} tag: `{left_tag}`",
        f"- {right_label} tag: `{right_tag}`",
        f"- items judged (with key): {n_items}",
        "",
        "## Absolute scores (1–5, mean [95% CI bootstrap])",
        "",
        f"| dimension | {left_label} | {right_label} |",
        "|---|---|---|",
    ]
    by_dim: dict[str, dict[str, dict[str, Any]]] = {}
    for row in abs_rows:
        by_dim.setdefault(row["dimension"], {})[row["model"]] = row
    for dim in DIMENSIONS:
        cells = []
        for model in (left_label, right_label):
            row = by_dim.get(dim, {}).get(model)
            if row and row["mean"] is not None:
                cells.append(
                    f"{row['mean']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
                )
            else:
                cells.append("-")
        lines.append(f"| {dim} | {cells[0]} | {cells[1]} |")

    lines += [
        "",
        "## Pairwise preference (blind; exact sign test on discordant pairs)",
        "",
        f"| aspect | {left_label} wins | {right_label} wins | ties | p-value |",
        "|---|---|---|---|---|",
    ]
    for row in pair_rows:
        p = row["p_value"]
        p_text = "<0.001" if p is not None and p < 0.001 else f"{p:.3f}"
        lines.append(
            f"| {row['aspect']} | {row['gemma_wins']} | {row['qwen_wins']} "
            f"| {row['ties']} | {p_text} |"
        )
    lines += [
        "",
        "Interpretation: overlapping CIs on absolute scores, or p >= 0.05 on the "
        "pairwise test, means the difference is not statistically supported at "
        f"n={n_items}. This is a single-session LLM-judge draft and must be "
        "validated against the two-human-rater subset (see "
        "docs/llm_judge_report_eval_plan.md).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
