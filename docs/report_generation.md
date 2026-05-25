# Report Generation

## Purpose

Report generation mengubah hasil analisis energi menjadi artefak akademik berupa file LaTeX dan PDF. Modul ini dibuat agar hasil eksperimen siap dimasukkan ke Bab 4 tanpa menyusun grafik dan narasi secara manual.

## Components

Komponen utama:

- `visualization/energy_charts.py`: membuat grafik dari query agregat DuckDB.
- `visualization/chart_stats.py`: membuat statistik ringkas per chart.
- `agents/insight_agent.py`: menghasilkan narasi dari metadata chart dan stats.
- `reporting/report_schema.py`: struktur data laporan.
- `reporting/latex_builder.py`: render template Jinja2 ke `.tex`.
- `reporting/pdf_compiler.py`: compile `.tex` ke PDF.
- `graph/report_workflow.py`: orkestrasi end-to-end secara sequential.

## Generated Figures

Chart yang dibuat:

- daily active power trend,
- hourly consumption pattern,
- power distribution,
- voltage distribution,
- correlation heatmap,
- sub metering comparison.

Semua chart disimpan di:

```text
reports/figures/
```

## LaTeX and PDF Output

Output utama:

```text
reports/latex/energy_analysis_report.tex
reports/pdf/energy_analysis_report.pdf
```

Jika `tectonic` atau `pdflatex` tidak tersedia, workflow tetap menyimpan file `.tex` dan mencatat error kompilasi PDF.

## Metadata Log

Setiap run menyimpan metadata ke:

```text
reports/experiments/report_generation_log.json
```

Metadata berisi status chart, jumlah insight berhasil, status LaTeX, status PDF, path output, dan pesan error jika ada.

## Command

```powershell
python -m local_agentic_analytics.cli report energy
```

Script lama tetap tersedia:

```powershell
python scripts/generate_energy_report.py
```
