# Evaluation Plan

## Latency

Latency dicatat per tahap workflow:

- schema lookup,
- SQL generation,
- SQL execution,
- SQL repair jika terjadi,
- reporting,
- total latency.

Log utama tersedia di:

```text
reports/experiments/runs.csv
reports/experiments/batch_eval_energy.csv
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

## Memory and Resource Logging

Project sudah memiliki dependency `psutil`, tetapi resource logging detail belum menjadi workflow utama. Untuk tahap saat ini, kesiapan resource dipantau melalui desain sequential, konfigurasi DuckDB ringan, dan pemilihan satu model lokal.

Pengembangan berikutnya dapat menambahkan logging:

- peak memory process,
- CPU utilization,
- durasi Ollama generation,
- ukuran database,
- ukuran artefak output.
