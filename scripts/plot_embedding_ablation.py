"""Plot perbandingan model embedding untuk Gambar 3.8 buku TA.

Membaca ``reports/experiments/rag_embedding_ablation.csv`` dan menghasilkan
grafik batang perbandingan MRR dan Recall@k antara model embedding berbahasa
Inggris dan model multibahasa.

Pemakaian (dari root repo)::

    python scripts/plot_embedding_ablation.py
    python scripts/plot_embedding_ablation.py --format svg
    python scripts/plot_embedding_ablation.py --out docs/buku/gambar-3-8

Keluaran default: ``docs/buku/gambar-3-8-embedding-ablation.png`` (300 dpi).
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
DEFAULT_CSV = PROJECT_ROOT / "reports" / "experiments" / "rag_embedding_ablation.csv"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "buku" / "gambar-3-8-embedding-ablation"

METRIC_KEYS = ["mrr", "recall_at_1", "recall_at_3", "recall_at_5"]
METRIC_LABELS = ["MRR", "Recall@1", "Recall@3", "Recall@5"]

# Label seri mengikuti urutan baris pada CSV.
SERIES_LABELS = {
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2 (Inggris)",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": (
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
}
SERIES_COLORS = ["#c8c8c8", "#5b87b5"]  # abu terang, biru redup (ramah cetak)


def format_id(value: float) -> str:
    """Format angka gaya Indonesia (koma desimal), 3 digit."""
    return f"{value:.3f}".replace(".", ",")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Path keluaran tanpa ekstensi")
    parser.add_argument("--format", choices=["png", "svg", "both"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if len(rows) < 2:
        raise SystemExit(f"CSV hanya berisi {len(rows)} baris model: {args.csv}")

    x = np.arange(len(METRIC_KEYS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=args.dpi)

    for i, row in enumerate(rows[:2]):
        values = [float(row[k]) for k in METRIC_KEYS]
        label = SERIES_LABELS.get(row["embedding_model"], row["embedding_model"])
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label,
                      color=SERIES_COLORS[i], edgecolor="black", linewidth=0.8)
        for bar in bars:
            ax.annotate(format_id(bar.get_height()),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylabel("Skor", fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
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
