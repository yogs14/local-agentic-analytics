# Rencana Pengembangan: Improvisasi Evaluasi & Validitas

Dokumen ini merinci rencana eksekusi untuk lima area yang masih lemah: benchmark, analisis keterbatasan, justifikasi pemilihan model, validitas sampel, dan reproducibility. Rencana disusun agar seluruh pengukuran bersifat **apple-to-apple**: pertanyaan sama, prompt sama, hardware sama, scaffolding sama — hanya satu variabel yang berubah per eksperimen.

Kondisi awal (baseline saat ini):

- Model: `gemma2:2b` via Ollama (fine-tuned insight model dilatih di Kaggle).
- Gold SQL: energy v1 = 12, energy v2 = 36, finance = 8 pertanyaan.
- Ablasi (36 soal v2): execution success 55,6% → 69,4% (A→D), numeric match stagnan 22,2%.
- Belum ada: harness multi-model, benchmark publik, metrik RAG, taksonomi error, uji statistik.

---

## Fase 1 — Multi-Model Benchmark Harness (fondasi semua improvisasi)

**Tujuan:** repo bisa mem-benchmark SLM apa pun secara apple-to-apple tanpa mengubah kode workflow.

### 1.1 Model Registry

Buat `configs/models/` berisi satu YAML per kandidat:

```text
configs/models/gemma2_2b.yaml        # baseline saat ini
configs/models/qwen2.5_1.5b.yaml
configs/models/qwen2.5_3b.yaml
configs/models/gemma2_2b_ft.yaml     # hasil fine-tune Kaggle
configs/models/qwen2.5_1.5b_ft.yaml  # hasil fine-tune Kaggle
```

Setiap file mencatat variabel yang harus dikunci agar perbandingan valid: `ollama_tag`, `param_count`, `quantization` (samakan, mis. Q4_K_M), `context_window: 2048`, `temperature: 0.0`, `prompt_template` (default: template yang sama untuk semua), dan `modelfile_sha256` untuk model hasil fine-tune.

### 1.2 Benchmark Runner

Buat `scripts/run_model_benchmark.py`:

- Argumen: `--models qwen2.5_1.5b,gemma2_2b --suite sql_gold_v2|ablation|finance --repeats 3`.
- Untuk tiap model: pull/verifikasi tag Ollama → jalankan suite dengan `PipelineToggles` identik → tulis hasil ke `reports/experiments/model_benchmark/<model>/<suite>.csv` + `manifest.json` (versi model, hash dataset, commit repo, tanggal, hardware).
- Metrik per model: execution success rate, numeric match rate, unit correctness, latency per tahap (p50/p95), VRAM/RAM puncak (pakai `resource_monitor.py` yang sudah ada).
- Agregator `scripts/summarize_model_benchmark.py` → satu tabel perbandingan lintas model.

### 1.3 Alur Fine-tune Kaggle → Lokal (standar)

**Fine-tuning bukan prasyarat benchmark.** Alur dua gelombang untuk hemat kuota GPU Kaggle:

- **Gelombang 1 (tanpa fine-tune):** benchmark semua kandidat *base* langsung dari registry Ollama (`ollama pull`). Hasilnya menentukan kandidat mana yang layak di-fine-tune.
- **Gelombang 2:** fine-tune hanya 1–2 kandidat terbaik di Kaggle → import ke Ollama → re-benchmark. Ini menghasilkan dua sumbu perbandingan: antar-model (base vs base) dan kontribusi fine-tuning (base vs fine-tuned per model).

Agar model fine-tune antar-kandidat juga apple-to-apple, bakukan resep di notebook Kaggle:

1. Dataset fine-tune identik (`data/finetune/`, snapshot di-hash).
2. Konfigurasi LoRA identik (rank, alpha, epoch, lr) — dicatat di `docs/finetune_recipe.md`.
3. Export merge → GGUF → quantize Q4_K_M → `Modelfile` Ollama dengan system prompt identik.
4. Simpan `Modelfile` + hash + link notebook Kaggle di `configs/models/*.yaml`.

**Deliverable Fase 1:** harness jalan untuk gemma2:2b vs qwen2.5:1.5b vs qwen2.5:3b (base), tabel perbandingan pertama.

---

## Fase 2 — Klasifikasi Error Ablasi (analisis keterbatasan)

**Tujuan:** menjawab *mengapa* numeric match terhenti di 22,2% — bukan sekadar "SLM 2B kesulitan agregasi kompleks".

### 2.1 Taksonomi Error

Perluas `analyze_sql_gold_mismatches.py` (atau modul baru `evaluation/error_taxonomy.py`) dengan kategori eksklusif dan terurut (satu error utama per soal):

1. `syntax_error` — SQL tidak valid DuckDB.
2. `schema_linking` — kolom/tabel salah atau halusinasi kolom.
3. `unit_conversion` — lupa `/60.0`, salah satuan.
4. `date_filter` — salah rentang/format tanggal.
5. `aggregation_choice` — fungsi agregasi salah (AVG vs SUM vs MAX).
6. `grouping_logic` — GROUP BY / DATE_TRUNC / subquery salah (mis. kasus E009).
7. `nl_understanding_id` — salah memahami maksud pertanyaan bahasa Indonesia (SQL valid tapi menjawab pertanyaan lain).
8. `output_shape` — hasil benar secara logika tapi kolom/format beda sehingga numeric compare gagal.
9. `other/unknown`.

### 2.2 Prosedur

- Klasifikasi otomatis berbasis heuristik (parse AST SQL dengan `sqlglot`) + verifikasi manual seluruh kegagalan (36 soal × 4 konfigurasi ablasi → hanya baris gagal).
- Pisahkan juga penyebab `numeric_compared_count` hanya 15/36: soal yang tidak sebanding (multi-row, non-numerik) harus punya mekanisme perbandingan sendiri (set comparison / row match), bukan di-skip — ini kemungkinan penyumbang terbesar angka 22,2% yang tampak rendah.
- Uji hipotesis bahasa: jalankan subset 36 soal versi terjemahan Inggris → jika akurasi naik signifikan, `nl_understanding_id` terkonfirmasi sebagai faktor.

**Deliverable:** `reports/experiments/error_taxonomy.csv`, distribusi error per konfigurasi ablasi (A–D) dan per model (dari Fase 1), analisis naratif di laporan.

---

## Fase 3 — Perluasan Gold SQL & Validitas Statistik

**Tujuan:** menaikkan kekuatan statistik klaim; 36 soal terlalu kecil untuk membedakan konfigurasi.

### 3.1 Perluasan sampel

- Energy: 36 → **100+** soal; Finance: 8 → **30+** soal.
- Stratifikasi kesulitan mengikuti kriteria hardness SPIDER (easy / medium / hard / extra) berdasarkan jumlah klausa, agregasi, nesting — dicatat per soal di JSON gold.
- Perluas jenis: multi-kolom, GROUP BY + HAVING, window function sederhana, perbandingan antar periode, top-N.

### 3.2 Prosedur validasi gold

- Setiap gold SQL dieksekusi dan hasilnya diverifikasi manual + cross-check perhitungan pandas independen (perluas `verify_gold_v2.py`).
- Review kedua oleh anotator lain / dosen pembimbing untuk sampel acak 20% (catat tingkat kesepakatan).

### 3.3 Uji statistik

Tambah ke agregator benchmark:

- Bootstrap 95% CI untuk semua rate (execution success, numeric match).
- McNemar test untuk perbandingan berpasangan antar konfigurasi ablasi dan antar model (paired pada soal yang sama).
- Laporkan `n` di setiap tabel; hindari klaim beda jika CI tumpang tindih.

**Deliverable:** `energy_gold_questions_v3.json` (100+), `finance_gold_questions_v2.json` (30+), modul `evaluation/statistics.py`.

---

## Fase 4 — Benchmark Eksternal

### 4.1 Sistem text-to-SQL lain

Pembanding pada gold set internal (apple-to-apple, DuckDB + pertanyaan Indonesia yang sama):

1. **Model specialized:** `sqlcoder` (7B Q4 jika muat VRAM+RAM; jika tidak, `duckdb-nsql:7b` via CPU dengan catatan latency) — pembanding "SLM umum + scaffolding vs model SQL khusus".
2. **Strategi prompting alternatif:** zero-shot polos, few-shot statis, dan dekomposisi ala DIN-SQL pada model yang sama — pembanding "sistem" tanpa ganti model.
3. Opsional: framework open-source (mis. Vanna) dengan backend Ollama yang sama, sebagai pembanding sistem end-to-end.

### 4.2 Evaluasi RAG (finance_news)

- Bangun gold retrieval set: ~50 query → daftar ID dokumen relevan (dilabel manual dari koleksi `finance_news`).
- Metrik retrieval: Recall@k (k=1,3,5), MRR, Hit@k — implement di `evaluation/rag_eval.py`, hasil ke `reports/experiments/rag_eval.csv`.
- Metrik generasi (jawaban hybrid/RAG): faithfulness & answer relevance ala RAGAS — dinilai rubrik manual (skala 1–5, dua penilai) karena judge LLM lokal 2B tidak reliabel; opsional cross-check judge API bila diizinkan.
- Ablasi kecil: variasi k dan embedding model (jika ada alternatif ringan) untuk justifikasi konfigurasi.

**Deliverable Fase 4:** tabel perbandingan sistem, metrik RAG lengkap.

### 4.3 (Ditangguhkan) Benchmark publik SPIDER/BIRD

Ditangguhkan dengan alasan: berbahasa Inggris (tidak langsung menguji klaim inti bahasa Indonesia), setup paling mahal (dataset, protokol EX resmi, skema asing), dan nilai utamanya hanya titik banding literatur. Gold set internal yang diperluas (Fase 3) + pembanding sistem (4.1) sudah cukup untuk klaim komparatif. Jika nanti dibutuhkan reviewer/penguji, rencana teknisnya: subset stratified ~150 soal SPIDER dev, eksekusi di SQLite asli, metrik Execution Accuracy, dua mode (raw model vs pipeline penuh), via `scripts/run_public_benchmark.py`.

---

## Fase 5 — Justifikasi Model & Reproducibility

### 5.1 Justifikasi pemilihan model (berbasis data, bukan asumsi)

Tulis `docs/model_selection.md` berisi:

- Matriks kandidat: parameter, VRAM Q4, lisensi, dukungan bahasa Indonesia, skor coding/SQL dari literatur.
- Hasil empiris Fase 1 + Fase 4 (akurasi, latency, resource) di hardware target (GTX 1650 4GB / RAM 8GB).
- Keputusan final + trade-off eksplisit (mis. "qwen2.5:1.5b lebih akurat X% tapi latency +Y%").

### 5.2 Reproducibility

- `requirements.lock` (pip freeze) + versi Ollama + versi model tag di-pin.
- `manifest.json` per run eksperimen (sudah dirancang di Fase 1): commit hash, dataset hash, config hash, seed, hardware.
- Satu perintah reproduksi: `python scripts/reproduce_all.py --phase 1..5` yang menjalankan ulang seluruh suite dari manifest.
- Snapshot dataset evaluasi di-hash (SHA256) dan dicatat di `references/CHECKSUMS.txt`.
- Publikasikan link notebook Kaggle fine-tune + resep di `docs/finetune_recipe.md`.
- Dokumentasikan keterbatasan reproducibility yang tersisa (non-determinisme GPU inference meski temperature 0.0) dan mitigasinya (repeats ≥3, laporkan varians).

---

## Urutan Eksekusi & Estimasi

| Fase | Isi | Prasyarat | Estimasi |
|---|---|---|---|
| 1 | Harness multi-model + registry + fine-tune Qwen di Kaggle | — | 1–2 minggu |
| 2 | Taksonomi & klasifikasi error ablasi | Fase 1 (agar error dianalisis lintas model) | 1 minggu |
| 3 | Gold SQL 100+/30+ + uji statistik | bisa paralel dengan Fase 1–2 | 1–2 minggu |
| 4 | Sistem pembanding + RAG eval (SPIDER/BIRD ditangguhkan) | Fase 1, 3 | 1–1,5 minggu |
| 5 | Dokumen justifikasi model + reproducibility | Fase 1–4 | 3–5 hari |

Prioritas: **Fase 1 → 3 → 2 → 4.2 (RAG) → 4.1 (sistem lain) → 5**. Fase 1 dan 3 adalah fondasi — tanpa harness dan sampel yang cukup, hasil benchmark lain tidak kuat secara metodologis. Benchmark publik (4.3) hanya dikerjakan jika diminta penguji.

## Kriteria Sukses

- Semua rate dilaporkan dengan n dan 95% CI; klaim perbedaan didukung McNemar p<0,05.
- Minimal 3 model SLM dibandingkan apple-to-apple (base + fine-tuned) pada gold set 100+ soal.
- Numeric match 22,2% terurai menjadi distribusi kategori error dengan bukti per soal.
- RAG punya metrik kuantitatif (Recall@k, MRR) dan penilaian faithfulness.
- Seluruh eksperimen dapat direproduksi dari manifest dengan satu perintah.
