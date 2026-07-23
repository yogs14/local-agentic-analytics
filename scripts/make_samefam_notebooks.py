"""Turunkan notebook fine-tune gelombang guru sekeluarga dari notebook asli.

Gelombang kedua (guru sekeluarga) harus IDENTIK dengan gelombang pertama
(guru lintas keluarga / DeepSeek) pada seluruh hyperparameter, base model,
chat template, dan jalur ekspor. Satu-satunya yang berbeda adalah dataset
instruksi yang dipakai dan nama artefak keluarannya.

Karena itu notebook baru tidak ditulis ulang, melainkan diturunkan otomatis
dari notebook asli dengan substitusi terarah. Pendekatan ini menjamin tidak ada
hyperparameter yang tanpa sengaja bergeser di antara kedua gelombang.

Pemakaian (dari root repo)::

    python scripts/make_samefam_notebooks.py
    python scripts/make_samefam_notebooks.py --check     # hanya laporkan rencana

Keluaran::

    finetune-gemma-energy-samefam.ipynb
    finetune-qwen2-5-energy-samefam.ipynb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Substitusi per notebook: (berkas asal, berkas tujuan, daftar (cari, ganti)).
# Setiap pasangan diverifikasi kemunculannya agar kegagalan substitusi tidak
# lolos diam-diam.
JOBS = [
    {
        "src": "finetune-gemma-energy.ipynb",
        "dst": "finetune-gemma-energy-samefam.ipynb",
        "teacher": "google/gemma-4-31b-it",
        "dataset_slug": "finetune-teacher-gemma",
        "replacements": [
            # Dataset gelombang guru sekeluarga.
            (
                '/kaggle/input/datasets/yogafsyahputra/finetune',
                '/kaggle/input/datasets/yogafsyahputra/finetune-teacher-gemma',
            ),
            # Nama artefak agar tidak menimpa hasil gelombang pertama.
            ("gemma2-energy-insight-lora", "gemma2-energy-insight-fam-lora"),
            ("gemma2-energy-insight-gguf", "gemma2-energy-insight-fam-gguf"),
            (
                "gemma2-energy-insight-q4_k_m.gguf",
                "gemma2-energy-insight-fam-q4_k_m.gguf",
            ),
        ],
    },
    {
        "src": "finetune-qwen2-5-energy.ipynb",
        "dst": "finetune-qwen2-5-energy-samefam.ipynb",
        "teacher": "Qwen/Qwen3.5-27B",
        "dataset_slug": "finetune-teacher-qwen",
        "replacements": [
            (
                '/kaggle/input/datasets/yogafsyahputra/finetune',
                '/kaggle/input/datasets/yogafsyahputra/finetune-teacher-qwen',
            ),
            ("qwen25-energy-insight-lora", "qwen25-energy-insight-fam-lora"),
            ("qwen25-energy-insight-gguf", "qwen25-energy-insight-fam-gguf"),
            (
                "qwen25-energy-insight-q4_k_m.gguf",
                "qwen25-energy-insight-fam-q4_k_m.gguf",
            ),
        ],
    },
]

BANNER = """# Gelombang Guru Sekeluarga (same-family teacher)

> **Notebook ini diturunkan otomatis** dari `{src}` melalui
> `scripts/make_samefam_notebooks.py`. Jangan menyunting hyperparameter di sini.
> Bila resep pelatihan perlu berubah, ubah notebook asalnya lalu bangkitkan
> ulang berkas ini agar kedua gelombang tetap identik.

**Perbedaan satu-satunya terhadap notebook asal:**

| Aspek | Gelombang 1 (asal) | Gelombang 2 (notebook ini) |
|---|---|---|
| Guru pembangkit dataset | DeepSeek (lintas keluarga) | `{teacher}` (sekeluarga) |
| Direktori dataset Kaggle | `finetune` | `{dataset_slug}` |
| Nama artefak keluaran | tanpa sufiks | bersufiks `-fam` |

Seluruh hyperparameter QLoRA, base model, chat template, seed, dan jalur ekspor
GGUF **tidak berubah**, sehingga perbedaan kualitas narasi yang teramati dapat
diatribusikan pada asal guru, bukan pada perlakuan pelatihan.

**Wajib dicatat setelah run:** SHA256 kedua berkas dataset, tanggal run,
penyedia hulu OpenRouter yang melayani guru, dan tag Ollama final.
"""


def patch_notebook(job: dict, check_only: bool) -> None:
    src_path = PROJECT_ROOT / job["src"]
    dst_path = PROJECT_ROOT / job["dst"]

    if not src_path.is_file():
        raise SystemExit(f"Notebook asal tidak ditemukan: {src_path}")

    notebook = json.loads(src_path.read_text(encoding="utf-8"))

    counts = {old: 0 for old, _ in job["replacements"]}
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        text = "".join(source)
        original = text
        for old, new in job["replacements"]:
            if old in text:
                counts[old] += text.count(old)
                text = text.replace(old, new)
        if text != original:
            # Simpan kembali sebagai list baris agar diff notebook tetap rapi.
            cell["source"] = text.splitlines(keepends=True)

    missing = [old for old, hits in counts.items() if hits == 0]
    if missing:
        raise SystemExit(
            "Substitusi gagal, pola berikut tidak ditemukan di "
            f"{job['src']}:\n  - " + "\n  - ".join(missing)
        )

    banner = BANNER.format(
        src=job["src"], teacher=job["teacher"], dataset_slug=job["dataset_slug"]
    )
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": banner.splitlines(keepends=True),
        },
    )

    print(f"{job['src']} -> {job['dst']}")
    for old, hits in counts.items():
        print(f"    {hits:>2}x  {old}")

    if check_only:
        print("    (--check: berkas tidak ditulis)")
        return

    dst_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"    ditulis: {dst_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Tampilkan rencana substitusi tanpa menulis berkas")
    args = parser.parse_args()

    for job in JOBS:
        patch_notebook(job, args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
