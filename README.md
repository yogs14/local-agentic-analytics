# local-agentic-analytics

## 1. Project Overview

`local-agentic-analytics` adalah sistem agentic data analytics lokal untuk analisis konsumsi daya listrik rumah tangga. Sistem ini menggabungkan DuckDB, Ollama, ChromaDB, matplotlib, dan generator laporan LaTeX agar satu laptop lokal dapat menjalankan Q&A data terstruktur, evaluasi, visualisasi, insight, dan report generation tanpa layanan cloud.

Fokus implementasi saat ini adalah workflow DuckDB text-to-SQL untuk dataset energi. RAG berbasis ChromaDB sudah tersedia sebagai modul awal, tetapi tetap dipisahkan dari workflow utama agar sistem ringan dan mudah diuji.

## 2. Research Context

Project ini dikembangkan untuk tugas akhir dengan batasan perangkat lokal: RAM 8GB dan GPU GTX 1650 4GB. Karena itu desain sistem dibuat sequential, modular, dan hemat resource. Satu model lokal melalui Ollama digunakan untuk beberapa role agent, bukan satu model berbeda untuk setiap agent.

Pertanyaan riset praktis yang didukung project ini adalah bagaimana small language model lokal dapat membantu proses analisis data terstruktur, mulai dari konversi pertanyaan bahasa natural ke SQL, repair SQL sederhana, ringkasan hasil query, evaluasi akurasi, sampai penyusunan laporan analisis.

## 3. System Architecture

Arsitektur utama bersifat sequential:

1. User memberikan pertanyaan atau meminta laporan.
2. DuckDB menyediakan schema dan menjalankan query data terstruktur.
3. SQL Agent menghasilkan SQL DuckDB dari pertanyaan dan schema.
4. Repair Agent memperbaiki SQL satu kali jika eksekusi gagal.
5. Reporter Agent menjawab hasil query dalam bahasa Indonesia.
6. Visualization module membuat grafik deterministik dari agregasi DuckDB.
7. Insight Agent membuat narasi singkat dari metadata chart dan statistik ringkas.
8. Reporting module merender LaTeX dan mencoba compile PDF.

ChromaDB digunakan hanya untuk RAG/dokumen, bukan untuk data terstruktur. DuckDB tetap menjadi engine utama untuk dataset energi.

## 4. Directory Structure

```text
local-agentic-analytics/
|-- configs/
|-- data/
|   |-- evaluation/
|   |-- raw/
|   `-- processed/
|-- databases/
|   |-- chromadb/
|   `-- duckdb/
|-- docs/
|-- notebooks/
|-- references/
|   `-- sql_gold/
|-- reports/
|   |-- experiments/
|   |-- figures/
|   |-- latex/
|   `-- pdf/
|-- scripts/
|-- src/
|   `-- local_agentic_analytics/
|       |-- agents/
|       |-- core/
|       |-- evaluation/
|       |-- graph/
|       |-- prompts/
|       |-- reporting/
|       |-- tools/
|       `-- visualization/
`-- tests/
```

## 5. Setup Environment

Gunakan Python 3.10 atau 3.11 di Windows PowerShell.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Pastikan Ollama berjalan dan model tersedia.

```powershell
ollama pull gemma2:2b
ollama list
```

Default konfigurasi memakai `num_gpu: 0` untuk mengurangi risiko out-of-memory pada GTX 1650 4GB.

## 6. Download Dataset

Dataset yang digunakan adalah Individual Household Electric Power Consumption dari UCI Machine Learning Repository. Unduh file dataset, lalu letakkan file mentah di:

```text
data/raw/energy/household_power_consumption.txt
```

Jika folder `data/raw/energy/` belum ada, buat folder tersebut terlebih dahulu.

## 7. Ingest Energy Dataset

Jalankan ingestion untuk membuat DuckDB database dan tabel `electric_power`.

```powershell
python scripts/ingest_energy.py
```

Output utama:

```text
databases/duckdb/analytics.duckdb
table: electric_power
```

Script ingestion menggunakan DuckDB langsung dan menghindari full processing dataset besar dengan pandas.

## 8. Run Q&A Workflow

Mode Q&A melalui CLI:

```powershell
python -m local_agentic_analytics.cli ask "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
```

Script lama tetap tersedia dan memanggil mode CLI yang sama:

```powershell
python scripts/run_workflow.py "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
```

Output mencakup generated SQL, repaired SQL jika ada, result, final answer, latency, dan status.

## 9. Run Batch Evaluation

Batch evaluation menjalankan banyak pertanyaan energi dari `data/evaluation/energy_questions.json`.

```powershell
python scripts/run_batch_eval.py
```

Output disimpan ke:

```text
reports/experiments/batch_eval_energy.csv
```

Gold SQL benchmark dapat dijalankan dengan:

```powershell
python scripts/run_sql_gold_eval.py
python scripts/analyze_sql_gold_mismatches.py
```

Output utama:

```text
reports/experiments/sql_gold_eval.csv
reports/experiments/sql_gold_mismatch_report.md
```

## 10. Generate Energy Charts

Generate semua grafik energi deterministik dari agregasi DuckDB:

```powershell
python scripts/generate_energy_charts.py
```

Output:

```text
reports/figures/daily_active_power_trend.png
reports/figures/hourly_consumption_pattern.png
reports/figures/power_distribution.png
reports/figures/voltage_distribution.png
reports/figures/correlation_heatmap.png
reports/figures/sub_metering_comparison.png
```

## 11. Generate LaTeX/PDF Report

Mode report melalui CLI:

```powershell
python -m local_agentic_analytics.cli report energy
```

Script lama tetap tersedia:

```powershell
python scripts/generate_energy_report.py
```

Output:

```text
reports/latex/energy_analysis_report.tex
reports/pdf/energy_analysis_report.pdf
reports/experiments/report_generation_log.json
```

PDF membutuhkan `tectonic` atau `pdflatex`. Jika compiler tidak tersedia, file `.tex` dan log tetap disimpan.

## 12. Evaluation Metrics

Metrik evaluasi yang sudah didukung:

- Latency per tahap workflow.
- SQL execution success rate.
- Batch evaluation success rate.
- Gold SQL numeric match rate.
- Absolute error dan relative error untuk hasil numerik.
- Report generation success, termasuk status LaTeX dan PDF.
- Health check readiness untuk environment, database, tabel, dan folder output.

Jalankan health check:

```powershell
python scripts/check_project_health.py
```

## 13. Known Limitations

- Kualitas SQL bergantung pada model Ollama lokal dan prompt.
- Workflow utama masih single-table untuk `electric_power`.
- Repair Agent hanya melakukan satu kali repair per query.
- RAG masih tahap awal dengan dokumen dummy kecil.
- PDF generation membutuhkan compiler LaTeX eksternal.
- Analisis anomali belum final karena membutuhkan baseline atau pembanding historis eksplisit.

## 14. Future Work

- Menambahkan schema selection untuk multi-table analytics.
- Mengembangkan RAG ingestion dari dokumentasi dataset yang lebih lengkap.
- Menambahkan resource logging yang lebih detail untuk memori, CPU, dan durasi model.
- Menambahkan evaluasi kualitas narasi insight dan laporan.
- Menambahkan UI atau API server jika workflow inti sudah stabil.
- Mengintegrasikan LangGraph secara bertahap tanpa mengubah prinsip sequential execution.
