"""Grafik pengaruh asal pengajar (dua cabang) untuk pembahasan Bab 4.

Membaca ``reports/experiments/judge_gemma_fam/judge_summary.csv`` dan
``reports/experiments/judge_qwen_fam/judge_summary.csv`` lalu menghasilkan dua
panel skor juri lima dimensi, membandingkan pengajar lintas keluarga dan
sekeluarga pada tiap cabang, dengan error bar selang kepercayaan 95%.

Temuan yang divisualkan: asimetri arah pengaruh — cabang Gemma tidak berubah,
cabang Qwen membaik nyata di seluruh dimensi.

Pemakaian (dari root repo)::

    python scripts/plot_teacher_effect.py
    python scripts/plot_teacher_effect.py --format both
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
EXP = PROJECT_ROOT / "reports" / "experiments"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "buku" / "gambar-4-teacher-effect"

DIMS = ["faithfulness", "unit_correctness", "domain_richness", "coherence", "fluency_id"]
DIM_LABELS = ["Kesetiaan", "Satuan", "Istilah\ndomain", "Koherensi", "Kelancaran"]

BRANCHES = [
    ("judge_gemma_fam", "Cabang Gemma", "gemma_cross", "gemma_fam"),
    ("judge_qwen_fam", "Cabang Qwen", "qwen_cross", "qwen_fam"),
]
COLOR_CROSS = "#c8c8c8"
COLOR_FAM = "#5b87b5"


def read_abs(judge_dir: Path) -> dict[str, dict[str, tuple[float, float, float]]]:
    """dimension -> model -> (mean, ci_low, ci_high) dari judge_summary.csv."""
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for row in csv.DictReader(open(judge_dir / "judge_summary.csv", encoding="utf-8")):
        if row["section"] != "absolute":
            continue
        model, dim = row["key"].split("/", 1)
        out.setdefault(dim, {})[model] = (
            float(row["mean_or_wins"]),
            float(row["ci_low_or_p"]),
            float(row["ci_high"]),
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--format", choices=["png", "svg", "both"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), dpi=args.dpi, sharey=True)
    x = np.arange(len(DIMS))
    w = 0.36
    handles: list = []

    for ax, (folder, title, cross_key, fam_key) in zip(axes, BRANCHES):
        data = read_abs(EXP / folder)

        for j, (mkey, color, off) in enumerate(
            [(cross_key, COLOR_CROSS, -w / 2), (fam_key, COLOR_FAM, w / 2)]
        ):
            means, errlo, errhi = [], [], []
            for d in DIMS:
                m, lo, hi = data[d][mkey]
                means.append(m); errlo.append(m - lo); errhi.append(hi - m)
            bar = ax.bar(x + off, means, w,
                         label="Pengajar lintas keluarga" if j == 0
                         else "Pengajar sekeluarga",
                         color=color, edgecolor="black", linewidth=0.7,
                         yerr=[errlo, errhi], capsize=3, error_kw={"elinewidth": 0.8})
            if ax is axes[0]:
                handles.append(bar)

        ax.set_xticks(x)
        ax.set_xticklabels(DIM_LABELS, fontsize=8.5)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylim(0, 5.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Skor juri (1–5)", fontsize=10)
    # Satu legenda bersama di atas kedua panel agar tidak menutupi batang.
    fig.legend(handles, ["Pengajar lintas keluarga", "Pengajar sekeluarga"],
               fontsize=9.5, frameon=False, loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (["png", "svg"] if args.format == "both" else [args.format]):
        target = args.out.with_suffix(f".{ext}")
        fig.savefig(target, bbox_inches="tight")
        print(f"Tersimpan: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
