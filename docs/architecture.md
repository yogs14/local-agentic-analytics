# System Architecture

## Overview

`local-agentic-analytics` memakai arsitektur modular dan sequential untuk menjaga penggunaan resource tetap ringan pada laptop lokal. Sistem tidak menjalankan agent secara paralel dan tidak memakai model berbeda untuk setiap role.

Arsitektur konseptualnya dapat dibaca sebagai:

- Brain: Ollama local SLM yang menjalankan satu model lokal untuk beberapa role agent.
- Hand: tools yang melakukan aksi nyata, seperti DuckDB query, ChromaDB retrieval, SQL semantic guard, visualization, LaTeX render, dan PDF compile.
- Memory: DuckDB untuk data terstruktur, ChromaDB untuk dokumen, DatasetProfile untuk metadata domain, dan log eksperimen untuk audit.

## Sequential Multi-Agent Workflow

Workflow analitik berjalan berurutan:

1. User memberi pertanyaan bahasa natural.
2. Sistem membaca schema tabel dari DuckDB.
3. Rule-based SQL resolver mencoba memetakan pertanyaan umum secara deterministik.
4. Jika rule tidak cocok, SQL Agent menghasilkan DuckDB SQL.
5. SQL semantic guard memvalidasi aturan domain energi yang berisiko, seperti konversi kWh dan missing value.
6. DuckDB mengeksekusi SQL.
7. Repair Agent memperbaiki SQL satu kali jika query gagal atau tidak lolos semantic guard.
8. Reporter Agent membuat jawaban bahasa Indonesia dari hasil query.

Pola ini disebut multi-agent secara role-based, bukan parallel multi-agent. Setiap role memiliki prompt dan tanggung jawab berbeda, tetapi tetap dijalankan satu per satu.

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

RAG masih tahap awal dan belum menjadi bagian utama report generation.

## Logs as Memory

Log eksperimen dipakai sebagai memori observabilitas:

- `runs.csv` menyimpan ringkasan run Q&A.
- `tool_call_audit.jsonl` menyimpan jejak tool call dan metadata Ollama.
- `batch_eval_energy.csv` menyimpan evaluasi batch.
- `sql_gold_eval.csv` menyimpan perbandingan SQL agent dengan gold SQL.
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

LangGraph belum diintegrasikan pada tahap ini. Workflow custom sequential dipertahankan sampai DatasetProfile, semantic SQL rules, tool-call audit log, SQL evaluation, dan report evaluation stabil. Setelah fondasi ini konsisten, LangGraph dapat dipakai sebagai orchestration layer tanpa mengubah prinsip single-SLM, sequential execution, dan local-first analytics.
