"""Grafik komposisi waktu eksekusi untuk pembahasan Bab 4.

Dua panel:
(a) Perbandingan latensi rata-rata GPU vs CPU (dari gpu_cpu_benchmark_summary.json).
(b) Komposisi waktu total: inferensi model vs operasi deterministik, dihitung
    dari tool_call_audit.jsonl.

Pemakaian (dari root repo)::

    python scripts/plot_latency_composition.py
    python scripts/plot_latency_composition.py --format both
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP = PROJECT_ROOT / "reports" / "experiments"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "buku" / "gambar-4-latency-composition"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--format", choices=["png", "svg", "both"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Panel (a): GPU vs CPU
    summ = json.loads((EXP / "gpu_cpu_benchmark_summary.json").read_text())
    gpu = summ["modes"]["gpu"]["avg_total_latency"]
    cpu = summ["modes"]["cpu"]["avg_total_latency"]

    # Panel (b): komposisi waktu dari audit log
    llm_total = det_total = 0.0
    for line in (EXP / "tool_call_audit.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("latency_seconds")
        if not s:
            continue
        if str(r.get("component")) == "ollama":
            llm_total += s
        else:
            det_total += s
    total = llm_total + det_total

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), dpi=args.dpi)

    # (a) bar GPU/CPU
    ax = axes[0]
    bars = ax.bar(["GPU", "CPU"], [gpu, cpu], 0.55,
                  color=["#5b87b5", "#c8c8c8"], edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, [gpu, cpu]):
        ax.annotate(f"{v:.2f} s".replace(".", ","),
                    (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=9.5)
    ax.set_ylabel("Rata-rata latensi total (detik)", fontsize=10)
    ax.set_title("(a) Latensi GPU vs CPU", fontsize=11)
    ax.set_ylim(0, cpu * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    # (b) komposisi waktu
    ax = axes[1]
    parts = [llm_total / total * 100, det_total / total * 100]
    ax.barh(["Waktu\neksekusi"], [parts[0]], color="#5b87b5",
            edgecolor="black", linewidth=0.8, label="Inferensi model")
    ax.barh(["Waktu\neksekusi"], [parts[1]], left=[parts[0]], color="#c8c8c8",
            edgecolor="black", linewidth=0.8, label="Operasi deterministik")
    ax.annotate(f"Inferensi model\n{parts[0]:.1f}%".replace(".", ","),
                (parts[0] / 2, 0), ha="center", va="center", fontsize=10, color="white")
    ax.annotate(f"{parts[1]:.1f}%".replace(".", ","),
                (parts[0] + parts[1] / 2, 0), ha="center", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Porsi total waktu (%)", fontsize=10)
    ax.set_title("(b) Komposisi waktu eksekusi", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, -0.42))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (["png", "svg"] if args.format == "both" else [args.format]):
        target = args.out.with_suffix(f".{ext}")
        fig.savefig(target, bbox_inches="tight")
        print(f"Tersimpan: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
