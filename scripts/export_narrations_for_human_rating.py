"""Susun lembar penilaian manusia untuk memvalidasi skor LLM-as-judge.

Rencana evaluasi (docs/llm_judge_report_eval_plan.md) mensyaratkan subset 20-30
narasi dinilai dua penilai manusia dengan rubrik yang sama, lalu dilaporkan
korelasi judge-vs-manusia. Skrip ini mengambil subset acak berseed dari pasangan
narasi yang sudah dinilai juri, lalu menulis satu lembar CSV siap isi.

Lembar tetap BUTA: kolom identitas model tidak disertakan, dan tiap baris berisi
satu narasi tunggal (bukan pasangan) agar penilai tidak terpancing membandingkan
dan agar skornya sebanding dengan skor absolut juri.

Pemakaian (dari root repo)::

    python scripts/export_narrations_for_human_rating.py \
        --judge-dir reports/experiments/judge_qwen_fam --n-items 12

    # gabungkan beberapa perbandingan dalam satu lembar
    python scripts/export_narrations_for_human_rating.py \
        --judge-dir reports/experiments/judge_gemma_fam \
        --judge-dir reports/experiments/judge_qwen_fam --n-items 12

Keluaran: ``reports/experiments/human_validation/rating_sheet.csv`` dan
``rating_key.json`` (pemetaan ke model; JANGAN diberikan kepada penilai).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "experiments" / "human_validation"
DEFAULT_SEED = 20260722

DIMENSIONS = (
    "faithfulness",
    "unit_correctness",
    "domain_richness",
    "coherence",
    "fluency_id",
)
RATERS = ("r1", "r2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-dir",
        type=Path,
        action="append",
        required=True,
        help="Direktori hasil penjurian (boleh diulang untuk beberapa perbandingan).",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=12,
        help="Jumlah ITEM per direktori (tiap item menghasilkan 2 baris narasi).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_pairs(judge_dir: Path) -> tuple[list[dict], dict]:
    pairs_path = judge_dir / "narration_pairs.jsonl"
    key_path = judge_dir / "blinding_key.json"
    if not pairs_path.is_file():
        raise SystemExit(f"Tidak ditemukan: {pairs_path}")
    if not key_path.is_file():
        raise SystemExit(f"Tidak ditemukan: {key_path}")

    pairs = [
        json.loads(line)
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    key = json.loads(key_path.read_text(encoding="utf-8"))
    return pairs, key


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    rows: list[dict] = []
    key_map: dict[str, dict] = {}
    serial = 0

    for judge_dir in args.judge_dir:
        pairs, key = load_pairs(judge_dir)
        mapping = key.get("mapping", {})
        comparison = judge_dir.name

        chosen = rng.sample(pairs, min(args.n_items, len(pairs)))
        for pair in chosen:
            item_id = pair["item_id"]
            slots = mapping.get(item_id, {})
            for slot in ("A", "B"):
                serial += 1
                rating_id = f"H{serial:03d}"
                rows.append(
                    {
                        "rating_id": rating_id,
                        "chart_id": pair["chart_id"],
                        "stat_block": pair["stat_block"],
                        "narasi": pair[f"narration_{slot}"],
                        **{
                            f"{rater}_{dim}": ""
                            for rater in RATERS
                            for dim in DIMENSIONS
                        },
                        "r1_catatan": "",
                        "r2_catatan": "",
                    }
                )
                # Kunci disimpan terpisah: penilai tidak boleh melihat ini.
                key_map[rating_id] = {
                    "comparison": comparison,
                    "item_id": item_id,
                    "slot": slot,
                    "model": slots.get(slot, "(tidak diketahui)"),
                }

    # Acak urutan baris agar narasi dari item yang sama tidak berdampingan.
    rng.shuffle(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = args.out_dir / "rating_sheet.csv"
    key_path = args.out_dir / "rating_key.json"

    with sheet_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    key_path.write_text(
        json.dumps(
            {"seed": args.seed, "dimensions": list(DIMENSIONS), "mapping": key_map},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Lembar penilaian : {sheet_path}  ({len(rows)} baris narasi)")
    print(f"Kunci (RAHASIA)  : {key_path}")
    print()
    print("Langkah berikutnya:")
    print("  1. Kirim HANYA rating_sheet.csv kepada dua penilai, terpisah.")
    print("  2. Tiap penilai mengisi kolom r1_* atau r2_* secara independen,")
    print("     mengikuti rubrik pada docs/human_narration_rubric.md.")
    print("  3. Kumpulkan sebagai rating_sheet_r1.csv dan rating_sheet_r2.csv,")
    print("     lalu jalankan scripts/analyze_human_validation.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
