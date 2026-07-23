"""Catat manifest dataset instruksi fine-tuning untuk reproduksibilitas.

Menghitung SHA256 dan jumlah baris tiap gelombang dataset (DeepSeek lintas
keluarga, Gemma sekeluarga, Qwen sekeluarga), lalu menulis satu berkas manifest
yang menjadi rujukan Bab 3 dan lampiran. Hash berkas inilah yang dipakai untuk
memverifikasi bahwa notebook fine-tune benar-benar menerima dataset yang
dimaksud.

Pemakaian (dari root repo)::

    python scripts/record_dataset_manifest.py
    python scripts/record_dataset_manifest.py --waves teacher_gemma teacher_qwen
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINETUNE_DIR = PROJECT_ROOT / "data" / "finetune"
DEFAULT_OUT = FINETUNE_DIR / "dataset_manifest.json"

# Nama gelombang -> (subdirektori, label guru, slug Kaggle). Gelombang DeepSeek
# memakai berkas di root data/finetune bila subdirektorinya tidak ada.
WAVES = {
    "deepseek": ("", "DeepSeek (lintas keluarga)", "finetune"),
    "teacher_gemma": ("teacher_gemma", "google/gemma-4-31b-it (sekeluarga)",
                      "finetune-teacher-gemma"),
    "teacher_qwen": ("teacher_qwen", "Qwen/Qwen3.5-27B (sekeluarga)",
                     "finetune-teacher-qwen"),
}
DATASET_FILES = ("train.jsonl", "val.jsonl")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_manifest(wave_names: list[str]) -> dict:
    manifest = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": 42,
        "temperature": 0.4,
        "note": (
            "Seluruh gelombang memakai resep pembangkitan identik; "
            "yang berbeda hanya model guru. Hash memverifikasi berkas dataset "
            "yang diumpankan ke notebook fine-tune."
        ),
        "waves": {},
    }

    for name in wave_names:
        if name not in WAVES:
            raise SystemExit(f"Gelombang tidak dikenal: {name}")
        subdir, teacher, slug = WAVES[name]
        base = FINETUNE_DIR / subdir if subdir else FINETUNE_DIR

        files = {}
        total = 0
        for filename in DATASET_FILES:
            path = base / filename
            if not path.is_file():
                files[filename] = {"status": "hilang", "path": str(path)}
                continue
            lines = count_lines(path)
            total += lines
            files[filename] = {
                "sha256": sha256_file(path),
                "lines": lines,
            }

        manifest["waves"][name] = {
            "teacher": teacher,
            "kaggle_slug": slug,
            "directory": str(base.relative_to(PROJECT_ROOT)),
            "total_examples": total,
            "files": files,
        }

    return manifest


def print_summary(manifest: dict) -> None:
    print("=" * 72)
    print(f"Manifest dataset  |  {manifest['generated_at']}")
    print(f"seed={manifest['seed']}  temperature={manifest['temperature']}")
    for name, wave in manifest["waves"].items():
        print(f"\n[{name}]  guru: {wave['teacher']}")
        print(f"  slug Kaggle : {wave['kaggle_slug']}")
        print(f"  total       : {wave['total_examples']} contoh")
        for filename, info in wave["files"].items():
            if "sha256" in info:
                print(f"  {filename:<12} {info['lines']:>4} baris  "
                      f"{info['sha256'][:16]}...")
            else:
                print(f"  {filename:<12} HILANG ({info['path']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waves", nargs="+",
                        default=list(WAVES.keys()),
                        help="Gelombang yang dicatat (default: semua)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = build_manifest(args.waves)
    print_summary(manifest)

    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nManifest ditulis: {args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
