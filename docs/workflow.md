# Workflows

## Q&A Workflow

Q&A workflow menjawab pertanyaan analitik terhadap tabel `electric_power`.

Urutan:

1. User menjalankan CLI `ask`.
2. Sistem memeriksa database DuckDB.
3. DatasetProfile domain `energy` dimuat dari `domains/energy/profile.yaml`.
4. Schema `electric_power` dibaca melalui `DuckDBTool`.
5. Rule-based SQL resolver mencoba membuat SQL deterministik untuk query umum.
6. Jika rule tidak cocok, SQL Agent menghasilkan SQL DuckDB dengan compact DatasetProfile context.
7. SQL semantic guard memvalidasi aturan energi penting.
8. SQL dieksekusi oleh DuckDB.
9. Jika gagal atau tidak lolos semantic guard, Repair Agent memperbaiki SQL satu kali.
10. Query hasil repair divalidasi dan dieksekusi ulang.
11. Reporter Agent membuat jawaban bahasa Indonesia.
12. Latency, selected tools, tool calls, dan status disimpan ke log eksperimen.

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

## Tool Call Audit Trail

Setiap step Q&A workflow dicatat sebagai tool call. Jejak ini tersedia di:

```text
reports/experiments/tool_call_audit.jsonl
```

Tool call yang umum muncul:

- `duckdb.schema`
- `rule_based_sql_resolver.resolve`
- `ollama.sql_generation` jika resolver tidak cocok
- `sql_semantic_guard.validate`
- `duckdb.query`
- `ollama.sql_repair` jika repair terjadi
- `duckdb.query_repaired` jika repair terjadi
- `ollama.reporting`

Setiap record minimal berisi timestamp, component, action, tool, status, latency, input summary, output summary, error message, dan metadata. Untuk tool Ollama, metadata dapat menunjukkan `load_duration`, `prompt_eval_duration`, `eval_duration`, `prompt_eval_count`, dan `eval_count`.

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

## Report Evaluation Workflow

Report evaluation membandingkan artefak laporan dengan ground truth.

Urutan:

1. Baca ground truth dari `references/gold_reports/energy_report_ground_truth.json`.
2. Baca metadata report dari `reports/experiments/report_generation_log.json`.
3. Baca LaTeX dari `reports/latex/energy_analysis_report.tex`.
4. Hitung section completeness, chart validity, PDF compile status, LaTeX existence, dan final score.
5. Simpan hasil ke `reports/experiments/report_eval.json`.

Command:

```powershell
python scripts/evaluate_report.py
```
