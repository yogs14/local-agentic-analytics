# Prompt Eksekusi untuk Claude Code

Salin seluruh blok di bawah garis ke Claude Code (dijalankan dari root repo `local-agentic-analytics`).

---

Baca dulu `docs/rencana_pengembangan_improvisasi.md` — itu rencana induk. Tugasmu mengimplementasikan seluruh rencana tersebut secara bertahap, fase per fase. Kerjakan dengan todo list, dan **berhenti di akhir setiap fase** untuk saya review sebelum lanjut.

## Konteks

- Sistem agentic analytics lokal: DuckDB + Ollama (SLM lokal) + ChromaDB, workflow di `src/local_agentic_analytics/`, eval scripts di `scripts/`, hasil di `reports/experiments/`.
- Baseline: `gemma2:2b`, temperature 0.0, context 2048. Gold SQL: energy v2 = 36 soal, finance = 8. Ablasi 36 soal: execution success 55,6%→69,4% (config A→D), numeric match stagnan 22,2% (hanya 15/36 soal terbanding numerik).
- Hardware target: laptop Windows, RAM 8GB, GTX 1650 4GB. Semua harus jalan lokal.
- Modul yang sudah ada dan HARUS dipakai ulang, bukan ditulis ulang: `evaluation/resource_monitor.py`, `evaluation/sql_gold_eval.py`, `evaluation/ablation_eval.py`, `scripts/analyze_sql_gold_mismatches.py`, `scripts/verify_gold_v2.py`.

## Aturan Global

1. Jangan mengubah perilaku workflow existing (energy & finance harus tetap lolos eval lama). Jalankan suite eval lama sebagai regression check setelah tiap fase.
2. Semua eksperimen apple-to-apple: prompt sama, temperature 0.0, `PipelineToggles` identik, pertanyaan sama; satu-satunya variabel adalah yang sedang diuji.
3. **Jangan pernah memfabrikasi angka hasil benchmark.** Jika Ollama/model tidak tersedia saat kamu jalan, buat kode + dry-run mode, lalu beri saya perintah untuk menjalankannya sendiri.
4. Jangan mengubah isi gold set existing (`energy_gold_questions_v2.json`, 36 soal) — file baru untuk perluasan.
5. Dependensi baru minimal; yang direncanakan: `sqlglot`, `scipy` (McNemar), sisanya tanya dulu.
6. Setiap skrip baru punya `--help`, bisa dijalankan dari root repo di Windows, dan menulis output ke `reports/experiments/`.

## Fase 1 — Multi-Model Benchmark Harness

1. Buat `configs/models/` dengan skema YAML: `ollama_tag`, `param_count`, `quantization`, `context_window`, `temperature`, `prompt_template`, `modelfile_sha256` (nullable), `source` (base/finetuned), `kaggle_notebook_url` (nullable). Isi: `gemma2_2b.yaml` (baseline), `qwen2.5_1.5b.yaml`, `qwen2.5_3b.yaml`.
2. Buat `scripts/run_model_benchmark.py`: argumen `--models`, `--suite sql_gold_v2|ablation|finance`, `--repeats N` (default 3), `--dry-run`. Per model: verifikasi tag tersedia di Ollama (`ollama list`/pull), set model via mekanisme config existing, jalankan suite, tulis `reports/experiments/model_benchmark/<model>/<suite>_run<i>.csv` + `manifest.json` (commit hash, SHA256 dataset gold, config model, seed, timestamp, hardware, versi Ollama).
3. Buat `scripts/summarize_model_benchmark.py`: agregasi lintas model → tabel (execution success, numeric match, unit correctness, latency p50/p95, RAM/VRAM puncak) dengan mean±std antar repeats → `model_benchmark_summary.csv` + markdown.
4. Buat `docs/finetune_recipe.md`: template resep fine-tune Kaggle terstandar (dataset hash, konfigurasi LoRA, export GGUF Q4_K_M, Modelfile Ollama dengan system prompt identik baseline) — dokumen instruksi untuk saya isi, bukan untuk kamu eksekusi. Fine-tuning dilakukan manual di Kaggle, bukan tugasmu.
5. Checkpoint: tunjukkan hasil `--dry-run` semua model + regression check, lalu berhenti.

## Fase 2 — Taksonomi & Klasifikasi Error

1. Buat `src/local_agentic_analytics/evaluation/error_taxonomy.py`: klasifikasi kegagalan dengan kategori eksklusif terurut (ambil yang pertama cocok): `syntax_error`, `schema_linking`, `unit_conversion`, `date_filter`, `aggregation_choice`, `grouping_logic`, `nl_understanding_id`, `output_shape`, `other`. Gunakan `sqlglot` untuk parse AST (bandingkan struktur agent SQL vs gold SQL); `nl_understanding_id` dan kasus ambigu ditandai `needs_manual_review`.
2. Buat `scripts/run_error_taxonomy.py`: baca hasil `sql_gold_eval.csv` / hasil ablasi per konfigurasi, klasifikasikan semua baris gagal → `reports/experiments/error_taxonomy.csv` + distribusi per konfigurasi ablasi (dan per model bila hasil Fase 1 ada).
3. Perbaiki mekanisme numeric compare: soal multi-row/multi-kolom yang sekarang di-skip (penyebab hanya 15/36 terbanding) harus dibandingkan via row-set comparison (sorted rows, toleransi float). Tambahkan sebagai mode baru tanpa mengubah metrik lama — laporkan keduanya (`numeric_match_legacy`, `result_match_full`).
4. Buat `references/sql_gold/energy_gold_questions_v2_en.json`: terjemahan Inggris dari 36 soal (SQL gold sama) + dukungan `--questions` di runner, untuk uji hipotesis pemahaman bahasa Indonesia.
5. Checkpoint: tabel distribusi error + daftar baris `needs_manual_review` untuk saya label manual.

## Fase 3 — Perluasan Gold SQL & Statistik

1. Buat `scripts/generate_gold_candidates.py` + draft `references/sql_gold/energy_gold_questions_v3.json` (100+ soal) dan `finance_gold_questions_v2.json` (30+ soal): pertanyaan bahasa Indonesia + gold SQL + field `hardness` (easy/medium/hard/extra, kriteria ala SPIDER) + `category`. Variasikan: multi-kolom, GROUP BY + HAVING, top-N, perbandingan antar periode, window function sederhana. Tandai semua `verified: false`.
2. Perluas `verify_gold_v2.py` → `scripts/verify_gold_v3.py`: eksekusi semua gold SQL, cross-check dengan perhitungan pandas independen, laporkan yang gagal/mencurigakan. Saya yang memverifikasi manual dan mengubah `verified: true`.
3. Buat `src/local_agentic_analytics/evaluation/statistics.py`: bootstrap 95% CI untuk rate, McNemar test berpasangan antar konfigurasi/model. Integrasikan ke `summarize_model_benchmark.py` (kolom CI, p-value vs baseline).
4. Checkpoint: hasil verify + ringkasan distribusi hardness.

## Fase 4 — Pembanding Sistem & Evaluasi RAG

1. Prompting-strategy comparator: `scripts/run_prompting_comparison.py` dengan strategi `zero_shot`, `few_shot_static`, `decomposed` (ala DIN-SQL: schema linking → draft → refine) pada model dan gold set yang sama; hasil masuk format summary yang sama.
2. Tambah config `configs/models/sqlcoder_7b.yaml` (atau `duckdb-nsql:7b`) dengan catatan `cpu_only: true` bila melebihi VRAM; pastikan harness Fase 1 bisa menjalankannya.
3. RAG eval: buat `src/local_agentic_analytics/evaluation/rag_eval.py` (Recall@1/3/5, MRR, Hit@k terhadap koleksi `finance_news`) + `scripts/run_rag_eval.py`. Buat template `references/rag_gold/finance_news_retrieval_gold.json` (~50 query, field `relevant_doc_ids` kosong untuk saya label manual) + `scripts/build_rag_gold_template.py` yang mengisi kandidat query dari data existing.
4. Buat rubrik penilaian faithfulness/answer-relevance (skala 1–5) di `docs/rag_generation_rubric.md` + `scripts/export_rag_answers_for_rating.py` yang menyiapkan CSV untuk penilaian manual dua penilai.
5. Checkpoint: RAG eval jalan end-to-end dengan gold berlabel sebagian (yang sudah saya isi).
6. Benchmark publik SPIDER/BIRD **ditangguhkan** — jangan dikerjakan.

## Fase 5 — Justifikasi Model & Reproducibility

1. Buat `docs/model_selection.md`: matriks kandidat (parameter, VRAM Q4, lisensi, dukungan bahasa Indonesia, skor literatur) + slot untuk hasil empiris Fase 1/4 + keputusan dan trade-off. Isi bagian faktual dari config; bagian hasil diisi dari CSV summary nyata, bukan karangan.
2. Reproducibility: `requirements.lock` (pip freeze), `references/CHECKSUMS.txt` (SHA256 semua file gold/eval), `scripts/reproduce_all.py --phase N` yang menjalankan ulang suite dari manifest, dan bagian baru di README tentang cara reproduksi.
3. Verifikasi akhir: jalankan `scripts/check_project_health.py`, seluruh regression eval, dan `reproduce_all.py --phase 1 --dry-run`. Laporkan hasil apa adanya.

Mulai dari Fase 1. Buat todo list dulu, lalu kerjakan.
