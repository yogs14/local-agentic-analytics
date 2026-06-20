# Finance Domain Profile

Folder ini membuktikan workflow Q&A tidak hardcode ke domain energy.

- `profile.yaml` mendeskripsikan tabel `stock_prices`, kolom `date`, `close_price`, `volume`, dan `return_pct`.
- `evaluation_questions.json` berisi pertanyaan evaluasi ringan untuk agregasi harga, volume, return, dan missing value.
- `report_config.yaml` hanya placeholder karena report finance belum dibuat pada tahap ini.

Ingest dataset sintetis:

```powershell
python scripts/ingest_finance.py
```

Contoh Q&A:

```powershell
python -m local_agentic_analytics.cli ask --domain finance "Berapa rata-rata harga penutupan?"
python -m local_agentic_analytics.cli ask --domain energy "Berapa rata-rata konsumsi daya aktif pada tanggal 16 Desember 2006?"
```
