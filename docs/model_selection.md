# Justifikasi Pemilihan Model (Berbasis Data)

Dokumen ini merangkum kandidat SLM untuk pipeline text-to-SQL lokal, hasil
empiris pada hardware target, dan keputusan final beserta trade-off-nya.

- **Hardware target:** laptop Windows, RAM 8 GB, NVIDIA GTX 1650 4 GB, Ollama 0.30.11.
- **Sumber angka empiris:** `reports/experiments/model_benchmark_v3/model_benchmark_summary.csv`
  (energy gold v3, n=104, 3 repeats), `reports/experiments/model_benchmark/model_benchmark_summary.csv`
  (finance, n=8), `reports/experiments/prompting_comparison/*/summary.csv`.
  Tidak ada angka di dokumen ini yang bukan hasil eksekusi nyata; sel yang
  belum diukur ditandai `—` atau `<diisi manual>`.

## 1. Matriks Kandidat (fakta dari registry + pengukuran)

| Kandidat | Parameter | Kuantisasi | VRAM terukur (Ollama, GTX 1650) | Lisensi | Dukungan bahasa Indonesia | Skor SQL/coding literatur |
|---|---|---|---|---|---|---|
| gemma2:2b (baseline) | 2,6 B | Q4_0 | 1843 MB (100% GPU) | Gemma Terms of Use | Terbatas (fokus Inggris, multibahasa parsial) | `<diisi manual dari literatur>` |
| qwen2.5:1.5b | 1,5 B | Q4_K_M | 1126 MB (100% GPU) | Apache-2.0 | Eksplisit multibahasa (29+ bahasa, termasuk Indonesia) | `<diisi manual>` |
| qwen2.5:3b | 3,1 B | Q4_K_M | 2150 MB (100% GPU) | Qwen Research License (non-komersial) — periksa ulang sebelum publikasi/produk | Eksplisit multibahasa | `<diisi manual>` |
| sqlcoder:7b (pembanding khusus SQL) | 7 B | Q4_0 | belum diukur — `cpu_only: true` (bobot ±4,1 GB > VRAM 4 GB) | CC-BY-SA-4.0 (periksa ketentuan defog) | Tidak (prompt bahasa Inggris/SQL) | `<diisi manual>` |

Catatan kuantisasi: gemma2:2b memakai Q4_0 (default registry Ollama, artefak
yang sama dengan seluruh angka baseline historis), qwen memakai Q4_K_M.
Perbedaan ini dicatat sebagai keterbatasan komparabilitas minor; manifest tiap
run merekam kuantisasi aktual dari `ollama show`.

## 2. Hasil Empiris — Energy Gold v3 (n=104, 3 repeats, temperature 0.0)

| Model | Exec success | Full accuracy [95% CI] | Latency p50 | Tokens/s | VRAM | McNemar vs baseline (exec / full) |
|---|---|---|---|---|---|---|
| gemma2:2b | 67,3% | 24,0% [16,3%, 32,7%] | 10,36 s | 50,4 | 1843 MB | baseline |
| qwen2.5:1.5b | 96,2% | 30,8% [22,1%, 40,4%] | 6,57 s | 81,9 | 1126 MB | p<0,001 / p=0,039 |
| qwen2.5:3b | 86,9% | 34,9% [25,0%, 44,2%] | 7,71 s | 54,0 | 2150 MB | p<0,001 / p=0,027 |

Finance (n=8, 3 repeats): exec 100% ketiga model; full accuracy gemma 50,0%,
qwen1.5b 62,5%, qwen3b 62,5% (n terlalu kecil untuk uji beda; lihat CI di CSV).

Distribusi error (taksonomi v3, run 1): `date_filter` adalah mode kegagalan
dominan lintas model — gemma 40/79 (50,6%), qwen1.5b 45/72 (62,5%),
qwen3b 32/67 (47,8%).

## 3. Hasil Empiris — Strategi Prompting (qwen2.5:1.5b, n=104)

| Strategi | Exec success | Full accuracy [95% CI] | Latency p50 | LLM calls |
|---|---|---|---|---|
| zero_shot | 25,0% | 1,0% [0,0%, 2,9%] | 3,80 s | 1 |
| few_shot_static | 93,3% | **54,8% [45,2%, 64,4%]** | 3,26 s | 1 |
| decomposed (DIN-SQL) | 56,7% | 6,7% [2,9%, 12,5%] | 10,75 s | 3 |
| pipeline penuh (scaffolded) | 96,2% | 30,8% [22,1%, 40,4%] | 6,57 s | 1–2 |

McNemar berpasangan few_shot_static vs pipeline penuh (soal & model sama):
30 vs 5 diskordan, **p = 2,2e-05**. Dekomposisi ala DIN-SQL merugikan pada
ukuran model ini (schema-linking stage menyesatkan draft; 44% kegagalannya
schema_linking) — kontras dengan literatur DIN-SQL pada model besar.

Hasil gemma2:2b untuk perbandingan lintas model: `reports/experiments/`
`prompting_comparison/gemma2_2b/summary.md` (diisi otomatis saat run selesai).

## 4. Keputusan & Trade-off

**Rekomendasi model: qwen2.5:1.5b** sebagai model utama pipeline, menggantikan
gemma2:2b, dengan justifikasi terukur:

1. **Akurasi**: +6,8 poin full accuracy (p=0,039) dan +28,9 poin execution
   success (p<0,001) atas baseline pada n=104.
2. **Sumber daya**: VRAM 39% lebih kecil (1126 vs 1843 MB) — headroom penting
   pada GPU 4 GB yang juga menampung model insight fine-tuned; latency p50
   37% lebih cepat; throughput +61%.
3. **Lisensi**: Apache-2.0 (paling permisif di antara kandidat).
4. **Bahasa**: dukungan Indonesia eksplisit — konsisten dengan gold set
   berbahasa Indonesia.

**Trade-off yang dicatat secara eksplisit:**

- qwen2.5:3b lebih akurat secara absolut (34,9% vs 30,8%) tetapi CI keduanya
  tumpang tindih (belum ada uji berpasangan langsung 1.5b-vs-3b yang
  signifikan), VRAM hampir 2x lipat (2150 MB, menyisakan sedikit ruang di
  4 GB), latency +17%, dan lisensinya non-komersial. 3B layak dipertimbangkan
  hanya jika akurasi maksimal lebih penting daripada resource dan lisensi.
- **Rekomendasi sistem** (temuan terbesar): mengganti prompt berisi aturan
  panjang + scaffolding dengan few-shot statis yang menyasar mode kegagalan
  dominan (contoh rentang tanggal BETWEEN) menaikkan akurasi dari 30,8% ke
  54,8% pada model yang sama. Perubahan prompt pipeline adalah langkah dengan
  rasio biaya/manfaat terbaik — sebelum fine-tuning apa pun.
- Fine-tuning (gelombang 2, resep di `docs/finetune_recipe.md`) sebaiknya
  menunggu hasil integrasi few-shot ke pipeline, karena baseline yang akan
  di-fine-tune berubah.

**Keputusan final:** `<diisi manual setelah review — kandidat terpilih, tanggal, penanda tangan>`
