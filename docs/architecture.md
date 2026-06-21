# System Architecture

## Overview

`local-agentic-analytics` memakai arsitektur modular dan sequential untuk menjaga penggunaan resource tetap ringan pada laptop lokal. Sistem tidak menjalankan agent secara paralel dan tidak memakai model berbeda untuk setiap role.

Arsitektur konseptualnya dapat dibaca sebagai:

- Brain: Ollama local SLM yang menjalankan satu model lokal untuk beberapa role agent.
- Hand: tools yang melakukan aksi nyata, seperti DuckDB query, ChromaDB retrieval, SQL semantic guard, visualization, LaTeX render, dan PDF compile.
- Memory: DuckDB untuk data terstruktur, ChromaDB untuk dokumen, DatasetProfile untuk metadata domain, dan log eksperimen untuk audit.
- Orchestrator: workflow custom sequential atau LangGraph sebagai alternatif untuk Q&A.

## Sequential Multi-Agent Workflow

Workflow analitik berjalan berurutan:

1. User memberi pertanyaan bahasa natural.
2. Planner memilih rute retrieval (`STRUCTURED_SQL`, `RAG_NEWS`, atau `HYBRID`).
3. Untuk `STRUCTURED_SQL`: baca schema dari DuckDB, rule-based SQL resolver,
   lalu SQL Agent jika rule tidak cocok, semantic guard, eksekusi DuckDB, dan
   Repair Agent satu kali jika gagal.
4. Untuk `RAG_NEWS`: retrieval headline dari ChromaDB `finance_news`.
5. Untuk `HYBRID`: ringkasan harga DuckDB + berita ChromaDB, dengan degrade aman.
6. Reporter Agent membuat jawaban bahasa Indonesia dari hasil rute terpilih.

Pola ini disebut multi-agent secara role-based, bukan parallel multi-agent. Setiap role memiliki prompt dan tanggung jawab berbeda, tetapi tetap dijalankan satu per satu.

## Agentic Route Planning

Planner adalah satu keputusan nyata yang dipindahkan ke agen: pemilihan rute
retrieval. Untuk pertama kalinya tujuan node berikutnya ditentukan oleh
reasoning, bukan flag boolean yang dihitung kode.

Mengikuti pola rule-based-vs-LLM yang sudah ada, setiap keputusan planner
dipasangkan dengan resolver deterministik:

- `RuleBasedRouteResolver` (deterministik, murni, mudah diuji) berjalan lebih
  dulu dan menangani sinyal berita/harga yang eksplisit.
- `PlannerAgent` (LLM lokal yang sama, `gemma2:2b`, temperature 0.0, max_tokens
  kecil) hanya dipanggil ketika resolver ambigu dan `use_planner=True`. Parsing
  keluarannya tangguh terhadap model 2B yang lemah.
- Energy selalu `STRUCTURED_SQL` tanpa memanggil LLM (short-circuit), sehingga
  perilakunya tidak berubah.
- Kegagalan apa pun (parsing, timeout, model tak tersedia) degrade aman ke
  `STRUCTURED_SQL`. Planner tidak pernah memecahkan workflow.

Keputusan planner dicatat sebagai tool call (`planner.route` untuk rule-based,
`ollama.planning` untuk LLM) lengkap dengan metadata Ollama, dan tersimpan di
`state.planned_route`, `state.route_source`, serta `state.route_reasoning`.
Prinsip single-SLM dan sequential execution dipertahankan; tidak ada model atau
paralelisme baru.

## One Local Model, Multiple Roles

Satu model lokal melalui Ollama digunakan oleh beberapa agent:

- SQL Agent untuk text-to-SQL.
- Repair Agent untuk perbaikan SQL.
- Reporter Agent untuk jawaban ringkas.
- Insight Agent untuk narasi chart.

Desain ini menghindari overhead multi-model dan cocok untuk RAM 8GB serta GPU GTX 1650 4GB.

## Dataset Profile and Domain Adapter

DatasetProfile berada di `domains/energy/profile.yaml`. File ini menjadi memori domain yang menjelaskan:

- nama dataset dan domain,
- table name canonical,
- kolom datetime,
- daftar kolom dan satuan,
- semantic SQL rules untuk energi.

SQL Agent dan workflow memakai compact SQL context dari DatasetProfile agar prompt tetap pendek. Domain adapter disiapkan sebagai jalur untuk menambah domain baru secara bertahap tanpa membuat workflow utama terlalu hardcoded ke dataset energi.

## Tool Layer

Tool layer adalah tangan sistem. Tool dipanggil secara eksplisit dan sequential:

- `DuckDBTool`: schema lookup, query, sample rows.
- `ChromaDBTool`: persistent vector store untuk RAG.
- `sql_semantic_guard`: validasi semantik SQL energi.
- Visualization module: chart deterministik dari query agregat DuckDB.
- Reporting module: render LaTeX dan compile PDF.

Setiap tool call di Q&A workflow dicatat ke `state.tool_calls` dan `reports/experiments/tool_call_audit.jsonl`.

## DuckDB for Structured Data

DuckDB menjadi engine utama untuk data terstruktur. Dataset energi dimuat ke database lokal:

```text
databases/duckdb/analytics.duckdb
table: electric_power
```

DuckDB dipakai untuk schema lookup, query agregasi, evaluasi gold SQL, dan statistik chart. Pandas hanya digunakan untuk output kecil atau dataframe hasil query, bukan untuk memuat dataset besar secara penuh.

## ChromaDB for RAG

ChromaDB disiapkan untuk retrieval dokumen. Modul ini dipisahkan dari workflow DuckDB agar tanggung jawabnya jelas:

- DuckDB: analisis data terstruktur.
- ChromaDB: retrieval dokumen dan konteks teks.

Pada domain finance, koleksi `finance_news` di ChromaDB kini terhubung langsung
ke Q&A workflow lewat planner: rute `RAG_NEWS` mengambil headline dari koleksi
ini, dan rute `HYBRID` menggabungkannya dengan ringkasan harga DuckDB menjadi
satu jawaban. Report generation tetap belum memakai RAG sebagai bagian utama.

## Logs as Memory

Log eksperimen dipakai sebagai memori observabilitas:

- `runs.csv` menyimpan ringkasan run Q&A.
- `tool_call_audit.jsonl` menyimpan jejak tool call dan metadata Ollama.
- `batch_eval_energy.csv` menyimpan evaluasi batch.
- `sql_gold_eval.csv` menyimpan perbandingan SQL agent dengan gold SQL.
- `planner_eval.csv` dan `planner_eval_summary.json` menyimpan akurasi routing planner.
- `report_generation_log.json` menyimpan metadata report generation.
- `report_eval.json` menyimpan evaluasi report terhadap ground truth.

## Visualization and Report Generator

Visualization module membuat grafik deterministik dari query agregat DuckDB. Insight Agent kemudian membaca metadata chart dan statistik ringkas, bukan dataframe mentah.

LaTeX report generator menjadi output unggulan karena menyatukan:

- grafik energi,
- statistik ringkas,
- insight naratif,
- template akademik,
- artefak `.tex` dan `.pdf`.

PDF dibuat melalui `tectonic` jika tersedia, lalu fallback ke `pdflatex`.

## LangGraph Position

LangGraph sudah tersedia sebagai orkestrator alternatif untuk Q&A workflow. Default CLI tetap memakai workflow custom sequential, sedangkan LangGraph dapat dipilih eksplisit dengan `--engine langgraph`.

Integrasi LangGraph merepresentasikan workflow sebagai graph state, node, edge, dan conditional routing. Jalur ini mempertahankan prinsip single-SLM, sequential execution, dan local-first analytics. Node LangGraph tetap memanggil komponen yang sama dengan workflow custom: DatasetProfile, RuleBasedSQLResolver, SQLAgent, SQLRepairAgent, ReporterAgent, DuckDBTool, semantic SQL guard, dan tool-call audit log. Report workflow sengaja belum dipindahkan ke LangGraph.

Command Q&A:

```powershell
python -m local_agentic_analytics.cli ask "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?" --engine custom
python -m local_agentic_analytics.cli ask "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?" --engine langgraph
python scripts/compare_workflow_engines.py
```
