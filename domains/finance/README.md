# Finance Domain Profile (Hybrid: DuckDB + ChromaDB)

Folder ini membuktikan dua hal:

1. Workflow Q&A tidak hardcode ke domain energy (domain-agnostic).
2. Sistem dapat menghubungkan **dua sumber data yang terpisah** untuk satu
   domain (hybrid connectivity).

## Dua sumber data terpisah (by design)

- **Terstruktur (DuckDB)** — harga saham harian dari `yfinance`, disimpan di
  tabel `stock_prices` pada database yang sama (`databases/duckdb/analytics.duckdb`),
  terpisah dari tabel `electric_power`.
  Kolom: `date, ticker, open, high, low, close, volume`.
  Ticker: NVDA, NFLX, TSLA, GOOGL. Rentang: 2019-01-01 s/d 2020-06-10.
- **Tidak terstruktur (ChromaDB)** — headline berita analis dari
  `data/raw/finance/raw_analyst_ratings.csv`, di-embed ke koleksi `finance_news`
  (terpisah dari koleksi RAG `local_agentic_analytics`).
  Metadata per dokumen: `ticker, date, publisher, url`.

Kedua sumber baru terhubung di lapisan aplikasi (lihat hybrid query), bukan di
dalam satu store. Itulah klaim utama yang dibuktikan.

## Ingestion (jalankan manual)

```powershell
python scripts/ingest_finance_prices.py   # DuckDB: tabel stock_prices (yfinance)
python scripts/ingest_finance_news.py     # ChromaDB: koleksi finance_news
```

## Q&A terstruktur

```powershell
python -m local_agentic_analytics.cli ask --domain finance "Berapa rata-rata harga penutupan NVDA antara 2 Januari 2019 dan 31 Januari 2019?"
python -m local_agentic_analytics.cli ask --domain energy "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
```

## RAG (sumber tidak terstruktur) dan Hybrid

```powershell
python scripts/run_finance_rag_query.py "berita terbaru tentang NVDA"
python scripts/run_finance_hybrid_query.py NVDA --start 2019-06-01 --end 2019-06-30
```

## Catatan desain resolver

Berbeda dari catatan awal "finance = LLM-only", domain ini sengaja
**mempertahankan rule-based SQL resolver deterministik** untuk pertanyaan umum
(rata-rata/maksimum harga penutupan, rata-rata volume), mengikuti gaya yang sama
dengan domain energy. Pertanyaan tersebut akan ter-resolve dengan
`route = rule_based_sql`; pertanyaan lain (mis. rentang tanggal kompleks,
perbandingan antar-ticker) jatuh ke jalur LLM yang dibantu few-shot
`FinanceDomainAdapter`.

## File pendukung

- `profile.yaml` — DatasetProfile YAML (table, kolom, unit, sql_rules).
- `report_config.yaml` — placeholder; report finance belum dibuat pada tahap ini.
- `src/local_agentic_analytics/domain/finance.py` — `FinanceDomainAdapter`
  dan few-shot SQL.
- `data/evaluation/finance_questions.json` — 15 pertanyaan (8 SQL, 4 RAG, 3 hybrid).
- `references/sql_gold/finance_gold_questions.json` + `F001.sql`..`F008.sql` —
  gold SQL; verifikasi dengan `python scripts/verify_finance_gold.py`.
