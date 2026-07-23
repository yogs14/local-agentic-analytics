"""Ringkas hasil uji coba kecil pembangkitan dataset per guru.

Dipakai sebelum gelombang penuh: menjalankan ``gen_dataset.py --n 8`` untuk tiap
kandidat guru, lalu skrip ini merangkum tingkat penerimaan, alasan penolakan,
dan menandai gejala kebocoran mode thinking pada narasi.

Pemakaian (dari root repo)::

    python scripts/inspect_teacher_trial.py data/finetune/trial_gemma
    python scripts/inspect_teacher_trial.py data/finetune/trial_gemma data/finetune/trial_qwen
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

# Penanda blok penalaran yang seharusnya tidak pernah muncul di narasi final.
THINKING_MARKERS = ("<think>", "</think>", "<reasoning>", "Let me think", "First, I")
# Penanda sisa format markdown / bahasa Inggris yang tidak diinginkan.
FORMAT_MARKERS = ("**", "##", "- ", "Based on", "The data", "shows that")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_rejected(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(out_dir: Path) -> None:
    train = load_jsonl(out_dir / "train.jsonl")
    val = load_jsonl(out_dir / "val.jsonl")
    rejected = load_rejected(out_dir / "rejected.csv")
    accepted = train + val

    print("=" * 72)
    print(f"Direktori : {out_dir}")
    total = len(accepted) + len(rejected)
    if total == 0:
        print("Tidak ada keluaran ditemukan. Pastikan gen_dataset.py sudah dijalankan.")
        return

    rate = len(accepted) / total * 100
    print(f"Diterima  : {len(accepted)} (train {len(train)}, val {len(val)})")
    print(f"Ditolak   : {len(rejected)}")
    print(f"Acceptance: {rate:.1f}%")

    if rejected:
        reasons = Counter()
        for row in rejected:
            for reason in str(row.get("reasons", "")).split("|"):
                reason = reason.strip()
                if reason:
                    # Kelompokkan ungrounded_number:<nilai> menjadi satu kategori.
                    reasons[reason.split(":")[0]] += 1
        print("\nAlasan penolakan:")
        for reason, count in reasons.most_common():
            print(f"  {count:>3}  {reason}")

    if not accepted:
        return

    # Pemeriksaan gejala masalah pada narasi yang lolos validator.
    thinking_hits, format_hits, lengths = [], [], []
    for row in accepted:
        text = str(row.get("output", ""))
        lengths.append(len(text.split()))
        if any(marker in text for marker in THINKING_MARKERS):
            thinking_hits.append(text[:80])
        if any(marker in text for marker in FORMAT_MARKERS):
            format_hits.append(text[:80])

    print(f"\nPanjang narasi: rata-rata {sum(lengths)/len(lengths):.0f} kata "
          f"(min {min(lengths)}, maks {max(lengths)})")
    print(f"Indikasi kebocoran thinking : {len(thinking_hits)}")
    print(f"Indikasi markdown/Inggris   : {len(format_hits)}")
    for sample in thinking_hits[:2]:
        print(f"  [thinking] {sample}...")
    for sample in format_hits[:2]:
        print(f"  [format]   {sample}...")

    print("\nContoh narasi diterima:")
    for row in accepted[:2]:
        chart = ""
        for line in str(row.get("input", "")).splitlines():
            if line.startswith("chart_id:"):
                chart = line.split(":", 1)[1].strip()
                break
        print(f"\n  chart_id: {chart}")
        print(f"  {row.get('output', '')}")

    print("\nPeriksa manual: kewajaran bahasa Indonesia, ketepatan istilah domain, "
          "dan kesesuaian angka dengan blok statistik masukan.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dirs", nargs="+", type=Path,
                        help="Direktori keluaran gen_dataset.py")
    args = parser.parse_args()
    for out_dir in args.out_dirs:
        summarize(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
