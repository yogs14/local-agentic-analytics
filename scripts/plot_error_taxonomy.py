"""Grafik distribusi taksonomi kesalahan untuk pembahasan Bab 4.

Membaca ``reports/experiments/error_taxonomy_v3_summary.csv`` dan menghasilkan
grafik batang berkelompok proporsi kategori kesalahan untuk ketiga model,
diurutkan menurun berdasarkan rerata proporsi lintas model.

Pemakaian (dari root repo)::

    python scripts/plot_error_taxonomy.py
    python scripts/plot_error_taxonomy.py --format both
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "reports" / "experiments" / "error_taxonomy_v3_summary.csv"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "buku" / "gambar-4-error-taxonomy"

MODELS = ["gemma2_2b", "qwen2.5_1.5b", "qwen2.5_3b"]
MODEL_LABELS = ["gemma2:2b", "qwen2.5:1.5b", "qwen2.5:3b"]
MODEL_COLORS = ["#c8c8c8", "#5b87b5", "#7a7a7a"]
CATEGORY_LABEL = {
    "date_filter": "Filter tanggal",
    "aggregation_choice": "Pemilihan agregasi",
    "syntax_error": "Galat sintaksis",
    "unit_conversion": "Konversi satuan",
    "nl_understanding_id": "Pemahaman bahasa",
    "schema_linking": "Penautan skema",
    "grouping_logic": "Logika pengelompokan",
    "other": "Lainnya",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--format", choices=["png", "svg", "both"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    share: dict[str, dict[str, float]] = defaultdict(dict)
    for row in csv.DictReader(open(SRC, encoding="utf-8")):
        if row["category"] == "category":
            continue
        share[row["category"]][row["group"]] = float(row["share"])

    cats = sorted(share, key=lambda c: -np.mean([share[c].get(m, 0) for m in MODELS]))
    labels = [CATEGORY_LABEL.get(c, c) for c in cats]

    x = np.arange(len(cats))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=args.dpi)
    for i, (m, ml) in enumerate(zip(MODELS, MODEL_LABELS)):
        vals = [share[c].get(m, 0) * 100 for c in cats]
        ax.bar(x + (i - 1) * w, vals, w, label=ml, color=MODEL_COLORS[i],
               edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Proporsi kesalahan (%)", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (["png", "svg"] if args.format == "both" else [args.format]):
        target = args.out.with_suffix(f".{ext}")
        fig.savefig(target, bbox_inches="tight")
        print(f"Tersimpan: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
