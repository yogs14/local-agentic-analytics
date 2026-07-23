"""Grafik perbandingan strategi prompting untuk pembahasan Bab 4.

Membaca ``reports/experiments/prompting_comparison/<model>/summary.csv`` dan
menghasilkan grafik batang berkelompok full accuracy tiga strategi (zero-shot,
few-shot statis, decomposed) untuk kedua model, dilengkapi error bar selang
kepercayaan 95%. Sebagai pembanding, garis putus-putus menandai full accuracy
sistem berkonfigurasi lengkap pada gold set yang sama.

Pemakaian (dari root repo)::

    python scripts/plot_prompting_comparison.py
    python scripts/plot_prompting_comparison.py --format both
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMP_DIR = PROJECT_ROOT / "reports" / "experiments" / "prompting_comparison"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "buku" / "gambar-4-prompting-comparison"

MODELS = [("gemma2_2b", "gemma2:2b"), ("qwen2.5_1.5b", "qwen2.5:1.5b")]
STRATEGIES = ["zero_shot", "few_shot_static", "decomposed"]
STRATEGY_LABELS = ["Zero-shot", "Few-shot statis", "Decomposed"]
STRATEGY_COLORS = ["#c8c8c8", "#5b87b5", "#9b9b9b"]

# Full accuracy pipeline penuh pada gold set 104 (dari model_benchmark_v3),
# untuk garis acuan pembanding. Sumber: Tabel 4.4.
FULL_PIPELINE = {"gemma2_2b": 0.2404, "qwen2.5_1.5b": 0.3077}


def read_summary(model_key: str) -> dict[str, dict[str, float]]:
    path = COMP_DIR / model_key / "summary.csv"
    out: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        out[row["config"]] = {
            "mean": float(row["full_accuracy_mean"]),
            "lo": float(row["full_accuracy_ci_low"]),
            "hi": float(row["full_accuracy_ci_high"]),
        }
    return out


def fmt_id(value: float) -> str:
    return f"{value*100:.1f}".replace(".", ",")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--format", choices=["png", "svg", "both"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), dpi=args.dpi, sharey=True)

    for ax, (model_key, model_name) in zip(axes, MODELS):
        data = read_summary(model_key)
        x = np.arange(len(STRATEGIES))
        for i, strat in enumerate(STRATEGIES):
            d = data[strat]
            err = [[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]]
            bar = ax.bar(x[i], d["mean"], 0.62, color=STRATEGY_COLORS[i],
                         edgecolor="black", linewidth=0.8,
                         yerr=np.array(err), capsize=4,
                         error_kw={"elinewidth": 0.9})
            ax.annotate(fmt_id(d["mean"]),
                        (x[i], d["hi"]), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8.5)

        # Garis acuan pipeline penuh.
        full = FULL_PIPELINE.get(model_key)
        if full is not None:
            ax.axhline(full, color="#b5482f", linestyle="--", linewidth=1.2)
            ax.annotate(f"pipeline penuh {fmt_id(full)}%",
                        (len(STRATEGIES) - 0.5, full), textcoords="offset points",
                        xytext=(0, 3), ha="right", fontsize=8, color="#b5482f")

        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGY_LABELS, fontsize=9.5)
        ax.set_title(model_name, fontsize=11)
        ax.set_ylim(0, 0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Full accuracy", fontsize=10)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    formats = ["png", "svg"] if args.format == "both" else [args.format]
    for ext in formats:
        target = args.out.with_suffix(f".{ext}")
        fig.savefig(target, bbox_inches="tight")
        print(f"Tersimpan: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
