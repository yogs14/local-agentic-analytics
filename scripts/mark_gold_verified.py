"""Tandai butir gold set sebagai terverifikasi setelah lolos pemeriksaan eksekusi.

Skrip verifikasi (``verify_finance_gold.py`` / ``verify_gold_v2.py``) hanya
melaporkan hasil dan tidak mengubah manifest. Skrip ini yang menuliskan status
``verified`` agar manifest konsisten dengan hasil verifikasi, sekaligus mencatat
cara verifikasinya supaya tidak diklaim lebih kuat daripada yang sebenarnya
dilakukan.

Penting: status yang ditulis adalah verifikasi EKSEKUSIONAL, yaitu kueri acuan
berhasil dijalankan dan hasilnya wajar (tidak kosong, tidak NULL, tidak nol).
Ini BUKAN pembacaan manual yang memastikan kueri benar-benar menjawab maksud
pertanyaannya. Butir yang sudah diperiksa manual sebaiknya ditandai terpisah
melalui ``--method manual``.

Pemakaian (dari root repo)::

    python scripts/mark_gold_verified.py --manifest references/sql_gold/finance_gold_questions_v2.json --dry-run
    python scripts/mark_gold_verified.py --manifest references/sql_gold/finance_gold_questions_v2.json
    python scripts/mark_gold_verified.py --manifest ... --ids F101,F102 --method manual
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Berkas manifest gold set yang diperbarui")
    parser.add_argument("--ids", default="",
                        help="Daftar id dipisah koma; kosong berarti semua butir")
    parser.add_argument("--method", default="execution",
                        choices=["execution", "manual"],
                        help="Cara verifikasi yang dicatat (default: execution)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Tampilkan perubahan tanpa menulis berkas")
    args = parser.parse_args()

    path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    if not path.is_file():
        raise SystemExit(f"Manifest tidak ditemukan: {path}")

    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise SystemExit("Manifest harus berupa daftar butir pertanyaan")

    target_ids = {i.strip() for i in args.ids.split(",") if i.strip()}
    stamp = dt.date.today().isoformat()

    changed = []
    for question in questions:
        qid = str(question.get("id", ""))
        if target_ids and qid not in target_ids:
            continue
        before = question.get("verified")
        question["verified"] = True
        question["verified_method"] = args.method
        question["verified_at"] = stamp
        if before is not True:
            changed.append(qid)

    print(f"Manifest        : {path.relative_to(PROJECT_ROOT)}")
    print(f"Total butir     : {len(questions)}")
    print(f"Metode          : {args.method}")
    print(f"Berubah menjadi terverifikasi: {len(changed)}")
    if changed:
        print("  " + ", ".join(changed))

    if args.dry_run:
        print("\n(--dry-run: berkas tidak ditulis)")
        return 0

    path.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nManifest diperbarui: {path.relative_to(PROJECT_ROOT)}")
    print("Jalankan ulang skrip checksum bila proyek mencatat hash gold set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
