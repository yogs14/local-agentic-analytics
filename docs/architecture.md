# System Architecture

## Overview

`local-agentic-analytics` memakai arsitektur modular dan sequential untuk menjaga penggunaan resource tetap ringan pada laptop lokal. Sistem tidak menjalankan agent secara paralel dan tidak memakai model berbeda untuk setiap role.

## Sequential Multi-Agent Workflow

Workflow analitik berjalan berurutan:

1. User memberi pertanyaan bahasa natural.
2. Sistem membaca schema tabel dari DuckDB.
3. SQL Agent menghasilkan DuckDB SQL.
4. DuckDB mengeksekusi SQL.
5. Repair Agent memperbaiki SQL satu kali jika query gagal.
6. Reporter Agent membuat jawaban bahasa Indonesia dari hasil query.

Pola ini disebut multi-agent secara role-based, bukan parallel multi-agent. Setiap role memiliki prompt dan tanggung jawab berbeda, tetapi tetap dijalankan satu per satu.

## One Local Model, Multiple Roles

Satu model lokal melalui Ollama digunakan oleh beberapa agent:

- SQL Agent untuk text-to-SQL.
- Repair Agent untuk perbaikan SQL.
- Reporter Agent untuk jawaban ringkas.
- Insight Agent untuk narasi chart.

Desain ini menghindari overhead multi-model dan cocok untuk RAM 8GB serta GPU GTX 1650 4GB.

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

## Visualization and Report Generator

Visualization module membuat grafik deterministik dari query agregat DuckDB. Insight Agent kemudian membaca metadata chart dan statistik ringkas, bukan dataframe mentah.

LaTeX report generator menjadi output unggulan karena menyatukan:

- grafik energi,
- statistik ringkas,
- insight naratif,
- template akademik,
- artefak `.tex` dan `.pdf`.

PDF dibuat melalui `tectonic` jika tersedia, lalu fallback ke `pdflatex`.
