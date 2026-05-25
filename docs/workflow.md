# Workflows

## Q&A Workflow

Q&A workflow menjawab pertanyaan analitik terhadap tabel `electric_power`.

Urutan:

1. User menjalankan CLI `ask`.
2. Sistem memeriksa database DuckDB.
3. Schema `electric_power` dibaca melalui `DuckDBTool`.
4. SQL Agent menghasilkan SQL DuckDB.
5. SQL dieksekusi oleh DuckDB.
6. Jika gagal, Repair Agent memperbaiki SQL satu kali.
7. Query hasil repair dieksekusi ulang.
8. Reporter Agent membuat jawaban bahasa Indonesia.
9. Latency dan status disimpan ke log eksperimen.

Command:

```powershell
python -m local_agentic_analytics.cli ask "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
```

## Batch Evaluation Workflow

Batch evaluation menguji stabilitas workflow pada banyak pertanyaan.

Urutan:

1. Baca daftar pertanyaan dari `data/evaluation/energy_questions.json`.
2. Jalankan Q&A workflow untuk tiap pertanyaan secara sequential.
3. Simpan generated SQL, repaired SQL, success, error, final answer, dan latency.
4. Hitung ringkasan total questions, success count, failed count, success rate, average latency, dan max latency.

Command:

```powershell
python scripts/run_batch_eval.py
```

Output:

```text
reports/experiments/batch_eval_energy.csv
```

## Gold SQL Evaluation Workflow

Gold SQL evaluation membandingkan SQL agent dengan SQL manual.

Urutan:

1. Baca daftar gold question.
2. Jalankan workflow agent untuk menghasilkan SQL.
3. Eksekusi SQL agent.
4. Eksekusi gold SQL.
5. Bandingkan hasil numerik jika memungkinkan.
6. Simpan absolute error, relative error, dan numeric match.

Command:

```powershell
python scripts/run_sql_gold_eval.py
python scripts/analyze_sql_gold_mismatches.py
```

## Report Generation Workflow

Report workflow menghasilkan laporan energi otomatis.

Urutan:

1. Connect ke `databases/duckdb/analytics.duckdb`.
2. Generate semua chart ke `reports/figures/`.
3. Hitung statistik ringkas per chart.
4. Generate insight per chart secara sequential memakai Insight Agent.
5. Susun `AnalysisReport`.
6. Render LaTeX ke `reports/latex/energy_analysis_report.tex`.
7. Compile PDF ke `reports/pdf/energy_analysis_report.pdf` jika compiler tersedia.
8. Simpan metadata run ke `reports/experiments/report_generation_log.json`.

Command:

```powershell
python -m local_agentic_analytics.cli report energy
```
