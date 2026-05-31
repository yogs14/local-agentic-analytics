# Evaluation Plan

## Latency

Latency dicatat per tahap workflow:

- schema lookup,
- rule-based SQL resolution,
- SQL generation,
- semantic SQL validation,
- SQL execution,
- SQL repair jika terjadi,
- reporting,
- total latency.

Log utama tersedia di:

```text
reports/experiments/runs.csv
reports/experiments/batch_eval_energy.csv
reports/experiments/tool_call_audit.jsonl
```

## SQL Execution Success Rate

SQL execution success rate dihitung dari jumlah query yang berhasil dieksekusi dibanding total pertanyaan evaluasi.

Indikator ini penting karena text-to-SQL lokal tidak hanya harus menghasilkan SQL yang terlihat benar, tetapi juga harus valid terhadap DuckDB dan schema dataset.

## Unit Correctness

Unit correctness mengecek apakah SQL memakai satuan dan transformasi domain yang tepat.

Contoh aturan:

- `Global_active_power` adalah kW.
- Total energi kWh dari data per menit dihitung dengan `SUM(Global_active_power) / 60.0`.
- Voltage memakai Volt.
- Global_intensity memakai Ampere.
- Missing value dihitung dengan `COUNT(*) FILTER (WHERE column IS NULL)`.

Aturan ini ditanam di DatasetProfile, prompt SQL Agent, RuleBasedSQLResolver, dan `sql_semantic_guard`. Guard saat ini sengaja sempit: hanya menjaga regresi lama untuk total energi kWh dan missing value count.

## Gold SQL Comparison

Gold SQL comparison membandingkan hasil SQL agent dengan SQL manual untuk pertanyaan penting.

Metrik:

- `agent_success`,
- `gold_success`,
- `numeric_match`,
- `absolute_error`,
- `relative_error`.

Output:

```text
reports/experiments/sql_gold_eval.csv
reports/experiments/sql_gold_mismatch_report.md
```

## Semantic Mismatch Analysis

Semantic mismatch analysis membaca hasil SQL gold evaluation dan menandai kemungkinan penyebab mismatch.

Kategori diagnosis awal:

- `possible_unit_conversion_issue`
- `possible_aggregation_issue`
- `possible_grouping_issue`
- `possible_date_filter_issue`
- `unknown`

Script:

```powershell
python scripts/analyze_sql_gold_mismatches.py
```

Analisis ini membantu memperbaiki prompt, DatasetProfile, resolver, atau semantic guard tanpa menebak-nebak dari satu query manual.

## Report Generation Success

Report generation dievaluasi dari:

- chart berhasil dibuat,
- insight berhasil dibuat,
- file `.tex` berhasil dirender,
- PDF berhasil dikompilasi jika compiler tersedia,
- metadata run tersimpan.

Output:

```text
reports/experiments/report_generation_log.json
reports/latex/energy_analysis_report.tex
reports/pdf/energy_analysis_report.pdf
```

## Report Ground Truth Evaluation

Report evaluation membandingkan artefak report dengan ground truth:

```text
references/gold_reports/energy_report_ground_truth.json
```

Metrik awal:

- `section_completeness`
- `chart_validity`
- `pdf_compile_success`
- `latex_exists`
- `required_chart_count`
- `existing_chart_count`
- `final_score`

Numeric accuracy untuk isi LaTeX belum diparse otomatis pada tahap ini dan ditandai `not_implemented`. Output:

```text
reports/experiments/report_eval.json
```

Command:

```powershell
python scripts/evaluate_report.py
```

## Tool Call Logs

Tool-call logs dipakai untuk auditability dan diagnosis latency. Setiap record mencatat:

- component dan action,
- nama tool canonical,
- status sukses/gagal,
- latency,
- ringkasan input/output,
- error message,
- metadata.

Untuk Ollama, metadata dapat menunjukkan apakah lambat karena model loading, prompt evaluation, atau output generation.

## Memory and Resource Logging

Project sudah memiliki dependency `psutil`, tetapi resource logging detail belum menjadi workflow utama. Untuk tahap saat ini, kesiapan resource dipantau melalui desain sequential, konfigurasi DuckDB ringan, dan pemilihan satu model lokal.

Pengembangan berikutnya dapat menambahkan logging:

- peak memory process,
- CPU utilization,
- durasi Ollama generation,
- ukuran database,
- ukuran artefak output.
