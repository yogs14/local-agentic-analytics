# local-agentic-analytics

## 1. Project Overview

`local-agentic-analytics` adalah sistem agentic data analytics lokal yang menjalankan Q&A data terstruktur, evaluasi, visualisasi, insight, report generation, dan benchmark end-to-end sepenuhnya di satu laptop tanpa layanan cloud. Sistem ini menggabungkan DuckDB (data terstruktur), Ollama (small language model lokal), ChromaDB (RAG dokumen), matplotlib (chart deterministik), LangGraph (orkestrator alternatif), dan generator laporan LaTeX.

Sistem bersifat **multi-domain**. Domain awal adalah **energy** (konsumsi daya listrik rumah tangga), dan domain kedua adalah **finance** (harga saham harian) yang sekaligus membuktikan dua hal: workflow Q&A tidak hardcode ke satu dataset (domain-agnostic), dan sistem dapat menghubungkan dua sumber data terpisah — DuckDB terstruktur dan ChromaDB tidak terstruktur — pada lapisan aplikasi (hybrid connectivity).

Engine default tetap `custom` (sequential), sedangkan LangGraph tersedia sebagai alternatif eksplisit untuk Q&A dan report energy. RAG berbasis ChromaDB dipisahkan dari workflow utama agar sistem ringan dan mudah diuji.

## 2. Research Context

Project ini dikembangkan untuk tugas akhir dengan batasan perangkat lokal: RAM 8GB dan GPU GTX 1650 4GB. Karena itu desain sistem dibuat sequential, modular, dan hemat resource. Satu model lokal melalui Ollama digunakan untuk beberapa role agent, bukan satu model berbeda untuk setiap agent.

Pertanyaan riset praktis yang didukung project ini adalah bagaimana small language model lokal dapat membantu proses analisis data terstruktur, mulai dari konversi pertanyaan bahasa natural ke SQL, repair SQL sederhana, ringkasan hasil query, evaluasi akurasi, sampai penyusunan laporan analisis — dan apakah scaffolding (rule-based resolver, semantic guard, domain adapter) dapat menutup keterbatasan SLM lokal. Eksperimen ablation dan benchmark GPU vs CPU disediakan untuk memisahkan kontribusi LLM dari kontribusi scaffolding dan untuk mengukur biaya resource.

## 3. System Architecture

Arsitektur utama bersifat sequential:

1. User memberikan pertanyaan atau meminta laporan, beserta domain (default `energy`).
2. DuckDB menyediakan schema dan menjalankan query data terstruktur.
3. DatasetProfile menyediakan metadata domain, table name, unit, dan semantic SQL rules.
4. Rule-based SQL resolver mencoba menyelesaikan query umum secara deterministik.
5. Jika rule tidak cocok, SQL Agent menghasilkan SQL DuckDB dari pertanyaan dan schema, dibantu few-shot dari DomainAdapter bila domain menyediakannya.
6. SQL semantic guard memeriksa kesalahan lama seperti total kWh tanpa `/60.0` dan missing value yang salah.
7. Repair Agent memperbaiki SQL satu kali jika query gagal atau melanggar semantic guard.
8. Reporter Agent menjawab hasil query dalam bahasa Indonesia.
9. Visualization module membuat grafik deterministik dari agregasi DuckDB.
10. Insight Agent membuat narasi singkat dari metadata chart dan statistik ringkas.
11. Reporting module merender LaTeX dan mencoba compile PDF.

ChromaDB digunakan hanya untuk RAG/dokumen, bukan untuk data terstruktur. DuckDB tetap menjadi engine utama untuk data terstruktur di semua domain.

Agent memakai satu model lokal melalui Ollama dengan role prompt berbeda. SQL Agent, Repair Agent, Reporter Agent, dan Insight Agent bukan model terpisah; semuanya berbagi satu backend SLM agar cocok dengan batasan laptop lokal.

### Dataset Profile / Domain Adapter

Metadata domain disimpan per domain di `domains/<domain>/profile.yaml`, mis. `domains/energy/profile.yaml` dan `domains/finance/profile.yaml`. Profile mendefinisikan nama tabel canonical, kolom datetime, daftar kolom, satuan, dan semantic SQL rules. Contoh aturan energy:

- `Global_active_power` adalah daya dalam `kW`.
- Total energi dari data per menit dihitung dengan `SUM(Global_active_power) / 60.0`.
- Missing value dihitung dengan `COUNT(*) FILTER (WHERE column IS NULL)`.
- Filter tanggal memakai `CAST(datetime AS DATE) = DATE 'YYYY-MM-DD'`.

`DatasetProfile` membuat workflow tidak hardcoded ke satu dataset. Pemilihan domain dilakukan via flag `--domain` dan diteruskan ke `SequentialAnalyticsWorkflow(domain=...)`.

### Domain Registry dan Few-shot Adapter

Modul `src/local_agentic_analytics/domain/` berisi registry yang memetakan nama domain ke `DomainAdapter` opsional (`registry.get_domain_adapter`). Energy sengaja mengembalikan `None` agar jalur prompt-nya tidak berubah, sedangkan finance mengembalikan `FinanceDomainAdapter` sehingga few-shot SQL finance ikut diinjeksikan ke SQL Agent. Pola ini memungkinkan menambah domain baru tanpa merombak workflow inti.

### Hybrid Connectivity (Finance)

Domain finance memakai **dua sumber data terpisah**:

- **Terstruktur (DuckDB)** — harga saham harian dari `yfinance`, tabel `stock_prices` pada database yang sama (`databases/duckdb/analytics.duckdb`), terpisah dari tabel `electric_power`. Kolom: `date, ticker, open, high, low, close, volume`. Ticker: NVDA, NFLX, TSLA, GOOGL. Rentang: 2019-01-01 s/d 2020-06-10.
- **Tidak terstruktur (ChromaDB)** — headline berita analis, di-embed ke koleksi `finance_news`, terpisah dari koleksi RAG `local_agentic_analytics`. Metadata per dokumen: `ticker, date, publisher, url`.

Kedua sumber baru terhubung di lapisan aplikasi melalui hybrid query, bukan di dalam satu store. Detail lengkap ada di `domains/finance/README.md`.

### Tool Calling and Audit Log

Tool calling dilakukan secara eksplisit melalui wrapper lokal:

- `DuckDBTool` untuk schema lookup dan query data terstruktur.
- `ChromaDBTool` untuk retrieval dokumen RAG.
- `sql_semantic_guard` untuk validasi semantik SQL energi.
- Visualization module untuk chart deterministik.
- LaTeX renderer dan PDF compiler untuk report artifacts.

Setiap tool call di Q&A workflow dicatat ke `state.tool_calls` dan `reports/experiments/tool_call_audit.jsonl`. LangGraph report workflow juga mencatat node penting ke `tool_calls` dan JSONL audit log, termasuk `report.generate_charts`, `report.build_chart_contexts`, `ollama.generate_insights`, `report.build_report`, `report.render_latex`, `report.compile_pdf`, dan `report.write_log`.

Audit log menyimpan `timestamp`, `component`, `action`, `tool`, `status`, latency, input/output summary, error message, dan metadata. Untuk Ollama, metadata dapat berisi `load_duration`, `prompt_eval_duration`, `eval_duration`, `prompt_eval_count`, dan `eval_count`.

### Report Ground Truth Evaluation

Report evaluation memakai ground truth di:

```text
references/gold_reports/energy_report_ground_truth.json
```

Evaluator membaca metadata report, LaTeX, folder figures, folder PDF, dan ground truth. Skor yang dihitung meliputi `section_completeness`, `chart_validity`, `pdf_compile_success`, `latex_exists`, `unit_rule_compliance`, `numeric_fact_coverage`, dan `final_report_score`.

Unit rules yang diperiksa:

- Rata-rata `Global_active_power` harus memakai `kW`.
- Total energy dari `Global_active_power` harus memakai `kWh`.
- `Voltage` harus memakai `V`.
- `Global_intensity` harus memakai `A`.

Numeric fact coverage dilakukan secara deterministik dari LaTeX, misalnya nilai `3.0534747475` dapat cocok dengan bentuk rounding seperti `3.05` atau `3.053`. Evaluator tidak memakai LLM untuk menilai report. Output evaluasi disimpan ke:

```text
reports/experiments/report_eval.json
```

### LangGraph Alternative Orchestrator

Workflow Q&A dan report default tetap memakai engine `custom`. Untuk Q&A, engine custom adalah `SequentialAnalyticsWorkflow`. Untuk report energy, engine custom adalah `EnergyReportWorkflow`. LangGraph tersedia sebagai orkestrator alternatif melalui `--engine langgraph`, bukan sebagai pengganti langsung workflow lama. Report finance saat ini hanya mendukung engine `custom`.

Integrasi LangGraph dipakai untuk merepresentasikan workflow sebagai graph state, node, edge, dan conditional routing. Jalur Q&A tidak mengganti komponen domain dan tool yang sudah ada: `DuckDBTool`, `SQLAgent`, `SQLRepairAgent`, `ReporterAgent`, `DatasetProfile`, semantic SQL guard, dan tool-call audit tetap dipakai.

LangGraph report workflow juga bersifat alternatif. Jalurnya tetap sequential: initialize state, generate charts, build chart contexts, generate insights, build report, render LaTeX, compile PDF, write log, dan finalize. Workflow ini memakai ulang logic dari `EnergyReportWorkflow`. Jika satu insight gagal, fallback insight dipakai dan workflow tetap lanjut.

Kedua integrasi LangGraph tetap sequential agar sesuai dengan batasan laptop lokal dan prinsip single local SLM.

## 4. Directory Structure

```text
local-agentic-analytics/
|-- configs/
|-- data/
|   |-- evaluation/          # energy_questions.json, finance_questions.json
|   |-- raw/                 # energy/, finance/ (dataset mentah, gitignored)
|   `-- processed/
|-- databases/
|   |-- chromadb/
|   `-- duckdb/
|-- docs/
|-- domains/
|   |-- energy/              # profile.yaml, report_config.yaml
|   `-- finance/             # profile.yaml, report_config.yaml, README.md
|-- notebooks/
|-- references/
|   |-- gold_reports/
|   `-- sql_gold/            # E0xx, E1xx (v2), F0xx + manifest JSON + review
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
|       |-- domain/          # adapters, registry, finance adapter
|       |-- evaluation/
|       |-- graph/
|       |-- prompts/
|       |-- reporting/
|       |-- tools/
|       `-- visualization/   # energy + finance charts
`-- tests/
```

Generated/local artifacts seperti report output, database DuckDB, vector store ChromaDB, dan dataset mentah tidak dipush ke repository. Foldernya tetap disediakan dengan `.gitkeep`, sedangkan isi lokal seperti `reports/figures/*`, `reports/latex/*`, `reports/pdf/*`, `reports/experiments/*`, `databases/duckdb/*`, `databases/chromadb/*`, dan `data/raw/*` diabaikan oleh `.gitignore`.

## 5. Setup Environment

Gunakan Python 3.10 atau 3.11 di Windows PowerShell.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

`requirements.txt` mencakup dependency untuk domain finance (mis. `yfinance` untuk ingestion harga saham). Embedding RAG memakai `sentence-transformers/all-MiniLM-L6-v2` yang dikonfigurasi di `configs/chromadb.yaml`; Ollama hanya dipakai untuk text generation, bukan embedding.

Pastikan Ollama berjalan dan model tersedia.

```powershell
ollama pull gemma2:2b
ollama list
```

Default konfigurasi memakai `num_gpu: 0` untuk mengurangi risiko out-of-memory pada GTX 1650 4GB.

## 6. Download / Ingest Dataset

### Energy

Dataset Individual Household Electric Power Consumption dari UCI Machine Learning Repository. Unduh dan letakkan file mentah di:

```text
data/raw/energy/household_power_consumption.txt
```

Jalankan ingestion untuk membuat tabel `electric_power`:

```powershell
python scripts/ingest_energy.py
```

Output utama: `databases/duckdb/analytics.duckdb` dengan table `electric_power`. Script ingestion memakai DuckDB langsung dan menghindari full processing dataset besar dengan pandas.

### Finance (opsional, untuk domain hybrid)

```powershell
python scripts/ingest_finance_prices.py   # DuckDB: tabel stock_prices (yfinance)
python scripts/ingest_finance_news.py     # ChromaDB: koleksi finance_news
```

Berita analis mentah dibaca dari `data/raw/finance/raw_analyst_ratings.csv`.

## 7. Run Q&A Workflow

Mode Q&A melalui CLI, domain dipilih via `--domain` (default `energy`):

```powershell
python -m local_agentic_analytics.cli ask "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
python -m local_agentic_analytics.cli ask --domain finance "Berapa rata-rata harga penutupan NVDA antara 2 Januari 2019 dan 31 Januari 2019?"
```

Engine dapat dipilih eksplisit:

```powershell
python -m local_agentic_analytics.cli ask "..." --engine custom
python -m local_agentic_analytics.cli ask "..." --engine langgraph
```

Script lama tetap tersedia dan memanggil jalur yang sama:

```powershell
python scripts/run_workflow.py "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
python scripts/run_workflow.py --engine langgraph "..."
```

Output mencakup generated SQL, repaired SQL jika ada, result, final answer, latency, route, dan status.

## 8. Run RAG dan Hybrid Query (Finance)

```powershell
python scripts/run_finance_rag_query.py "berita terbaru tentang NVDA"
python scripts/run_finance_hybrid_query.py NVDA --start 2019-06-01 --end 2019-06-30
```

RAG mengambil headline dari koleksi `finance_news`; hybrid menggabungkan agregasi harga `stock_prices` (DuckDB) dengan retrieval berita (ChromaDB) pada lapisan aplikasi. RAG energy/dokumen umum tetap tersedia via `scripts/run_rag_query.py`.

## 9. Run Batch Evaluation dan Gold SQL Benchmark

Batch evaluation menjalankan banyak pertanyaan energi dari `data/evaluation/energy_questions.json`:

```powershell
python scripts/run_batch_eval.py
python scripts/compare_workflow_engines.py
```

Output: `reports/experiments/batch_eval_energy.csv` dan `reports/experiments/engine_comparison.csv`.

### Gold SQL benchmark (energy v1)

```powershell
python scripts/run_sql_gold_eval.py
python scripts/analyze_sql_gold_mismatches.py
```

Output: `reports/experiments/sql_gold_eval.csv` dan `reports/experiments/sql_gold_mismatch_report.md`.

Manifest gold energy (`references/sql_gold/energy_gold_questions.json`, `E001.sql`..) memakai ID kanonik yang konsisten dengan `data/evaluation/energy_questions.json` (rentang E001–E022), sehingga join analisis pada `question_id` valid. Pertanyaan sub-metering gabungan lama dipecah menjadi tiga entri scalar yang dapat dibandingkan numerik (E011/E012/E013).

### Gold SQL benchmark v2 (36 pertanyaan, real database)

Manifest `references/sql_gold/energy_gold_questions_v2.json` (E101–E136) berisi pertanyaan energy yang lebih beragam (rentang jam, agregasi bulanan/mingguan, statistik, timestamp). Nilainya digrounding ke database nyata:

```powershell
python scripts/profile_dataset_v2.py   # profil dataset -> dataset_profile_v2.md
python scripts/verify_gold_v2.py       # eksekusi tiap gold SQL -> gold_review_v2.md
```

### Gold SQL finance (F001–F008)

```powershell
python scripts/verify_finance_gold.py
```

Manifest `references/sql_gold/finance_gold_questions.json` berisi 8 gold SQL finance. Set evaluasi `data/evaluation/finance_questions.json` berisi 15 pertanyaan (8 SQL, 4 RAG, 3 hybrid).

### Ablation dan benchmark resource

```powershell
python scripts/run_ablation_eval.py        # isolasi kontribusi LLM vs scaffolding
python scripts/run_gpu_cpu_benchmark.py     # bandingkan GPU vs CPU (mendukung --repeat)
```

## 10. Generate Energy Charts

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

Chart finance dideterministik melalui `finance_chart_registry`/`finance_charts` dan dipakai oleh report finance.

## 11. Generate LaTeX/PDF Report

Mode report melalui CLI:

```powershell
python -m local_agentic_analytics.cli report energy
python -m local_agentic_analytics.cli report finance
```

Engine report energy dapat dipilih eksplisit (default `custom`):

```powershell
python -m local_agentic_analytics.cli report energy --engine custom
python -m local_agentic_analytics.cli report energy --engine langgraph
```

Report finance hanya mendukung `--engine custom` pada tahap ini. Script lama tetap tersedia:

```powershell
python scripts/generate_energy_report.py
python scripts/generate_finance_report.py
```

Output report energy:

```text
reports/latex/energy_analysis_report.tex
reports/pdf/energy_analysis_report.pdf
reports/experiments/report_generation_log.json
reports/experiments/tool_call_audit.jsonl
```

Output CLI report menampilkan domain, engine, success, `tex_success`, `pdf_success`, path LaTeX/PDF/log, chart count, latency, ringkasan tool calls, dan error jika ada. Return code `0` diberikan selama `tex_success=True`, sehingga workflow tetap dianggap berguna meskipun PDF compiler tidak tersedia. PDF membutuhkan `tectonic` atau `pdflatex`; jika compiler tidak tersedia, file `.tex` dan log tetap disimpan.

Bandingkan workflow report custom vs LangGraph:

```powershell
python scripts/compare_report_workflows.py
```

Output: `reports/experiments/report_engine_comparison.json`.

## 12. Evaluation Metrics

Metrik evaluasi yang sudah didukung:

- Latency per tahap workflow.
- Tool-call audit trail per step workflow.
- SQL execution success rate.
- Batch evaluation success rate.
- Gold SQL numeric match rate (energy v1, energy v2, finance).
- Semantic mismatch analysis untuk SQL gold evaluation.
- Absolute error dan relative error untuk hasil numerik.
- Ablation evaluation untuk memisahkan kontribusi LLM dari scaffolding.
- GPU vs CPU benchmark untuk biaya resource model lokal.
- Report generation success, termasuk status LaTeX dan PDF.
- Report ground truth evaluation untuk section, chart, LaTeX, PDF, unit rules, numeric fact coverage, dan final report score.
- Report engine comparison untuk custom vs LangGraph.
- End-to-end benchmark untuk Q&A custom, Q&A LangGraph, report custom, dan report LangGraph.
- Health check readiness untuk environment, database, tabel, dan folder output.

```powershell
python scripts/check_project_health.py
python scripts/evaluate_report.py
python scripts/compare_report_workflows.py
python scripts/run_end_to_end_benchmark.py
```

Output benchmark:

```text
reports/experiments/end_to_end_benchmark.csv
reports/experiments/end_to_end_benchmark_summary.json
```

## 13. Known Limitations

- Kualitas SQL bergantung pada model Ollama lokal dan prompt.
- Workflow utama masih single-table per domain (`electric_power`, `stock_prices`).
- Repair Agent hanya melakukan satu kali repair per query.
- RAG/dokumen umum masih tahap awal dengan korpus kecil; finance news bergantung pada file mentah yang disediakan.
- Report finance belum mendukung engine LangGraph.
- PDF generation membutuhkan compiler LaTeX eksternal.
- `unit_rule_compliance` dan `numeric_fact_coverage` mengecek teks LaTeX secara deterministik, bukan menilai kualitas narasi secara semantik.
- Benchmark end-to-end tidak memakai LLM sebagai judge; success rate bergantung pada eksekusi workflow, gold SQL numeric match jika tersedia, dan metadata report.
- Analisis anomali belum final karena membutuhkan baseline atau pembanding historis eksplisit.

## 14. Future Work

- Menambahkan schema selection untuk multi-table analytics.
- Menyamakan dukungan LangGraph untuk report finance.
- Mengembangkan RAG ingestion dari dokumentasi dataset yang lebih lengkap.
- Menambahkan resource logging yang lebih detail untuk memori, CPU, dan durasi model.
- Menambahkan evaluasi kualitas narasi insight dan laporan.
- Menambahkan UI atau API server jika workflow inti sudah stabil.
- Memperluas cakupan domain dan benchmark tanpa mengubah prinsip sequential execution.
