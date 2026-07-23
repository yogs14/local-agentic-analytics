"""Prompt template for chart insight generation."""

from __future__ import annotations

import json
from typing import Any

from local_agentic_analytics.finetune.value_sampler import format_stats_block


INSIGHT_PROMPT_TEMPLATE = """Anda adalah analis data energi.

Tugas:
Tuliskan narasi analisis singkat dalam bahasa Indonesia formal akademik
berdasarkan metadata grafik dan statistik ringkas yang diberikan.

Aturan:
- Gunakan hanya angka dan informasi yang tersedia pada stats.
- Jangan mengarang angka di luar stats yang diberikan.
- Jangan menyebut kesimpulan yang tidak didukung data.
- Gunakan satuan yang benar:
  - Global_active_power = kW.
  - Total energy dari SUM(Global_active_power) / 60 = kWh.
  - Voltage = Volt.
  - Global_intensity = Ampere.
- Jika data tidak cukup untuk menyimpulkan anomali, tulis bahwa perlu pembanding historis.
- Jangan menulis markdown table.
- Output berupa 1-2 paragraf pendek.

Metadata grafik:
- chart_id: {chart_id}
- chart_title: {chart_title}
- chart_path: {chart_path}

Stats:
{stats}

Narasi analisis:
"""


FINANCE_INSIGHT_PROMPT_TEMPLATE = """Anda adalah analis data keuangan pasar saham.

Tugas:
Tuliskan narasi analisis singkat dalam bahasa Indonesia formal akademik
berdasarkan metadata grafik dan statistik ringkas yang diberikan.

Aturan:
- Gunakan hanya angka dan informasi yang tersedia pada stats.
- Jangan mengarang angka di luar stats yang diberikan.
- Jangan menyebut kesimpulan yang tidak didukung data.
- Gunakan satuan yang benar:
  - Harga (open, high, low, close) = USD.
  - Volume = lembar saham (shares).
  - Return harian = persen (%).
  - Korelasi tidak memiliki satuan.
- Sebut ticker (NVDA, NFLX, TSLA, GOOGL) secara eksplisit jika relevan.
- Jika data tidak cukup untuk menyimpulkan anomali, tulis bahwa perlu pembanding historis.
- Jangan menulis markdown table.
- Output berupa 1-2 paragraf pendek.

Metadata grafik:
- chart_id: {chart_id}
- chart_title: {chart_title}
- chart_path: {chart_path}

Stats:
{stats}

Narasi analisis:
"""


_INSIGHT_TEMPLATES = {
    "energy": INSIGHT_PROMPT_TEMPLATE,
    "finance": FINANCE_INSIGHT_PROMPT_TEMPLATE,
}


def build_energy_finetune_prompt(chart_context: dict[str, Any]) -> str:
    """Render a flat ``key: value`` stats block for the energy fine-tune model.

    ``gemma2-energy-insight`` was trained on the same compact stats blocks the
    fine-tune dataset emits (Indonesian comma decimals, one ``key: value`` per
    line, leading ``chart_id``) and carries its own system prompt, so the model
    only needs the raw statistics rather than the ruled prompt used for the
    generic :func:`build_insight_prompt` path.
    """
    chart_id = chart_context.get("chart_id")
    if chart_id in (None, ""):
        raise ValueError("Missing chart context keys: chart_id")

    stats = chart_context.get("stats")
    if not isinstance(stats, dict) or not stats:
        raise ValueError("stats must be a non-empty dict")

    block = {"chart_id": str(chart_id).strip(), **_flatten_energy_stats(stats)}
    return format_stats_block(block)


def _flatten_energy_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Flatten chart stats into the scalar-only shape the fine-tune expects.

    Most energy charts already expose flat scalars; the correlation heatmap nests
    its strongest pairs and a full correlation matrix, which are collapsed into
    the flat ``strongest_*_pair`` / ``strongest_*_r`` keys used during training.
    Non-scalar or empty values are dropped so the block stays free of Python
    container reprs.
    """
    flat: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif key in ("strongest_positive", "strongest_negative", "strongest_absolute") \
                and isinstance(value, dict):
            pair = value.get("pair")
            correlation = value.get("correlation")
            if pair is not None:
                flat[f"{key}_pair"] = pair
            if isinstance(correlation, (int, float)):
                flat[f"{key}_r"] = correlation
        elif key == "numeric_columns" and isinstance(value, (list, tuple)):
            flat[key] = ", ".join(str(item) for item in value)
        # Skip remaining non-scalar values (e.g. the full correlation matrix);
        # the strongest pairs above already ground the narrative.
    return flat


def build_insight_prompt(chart_context: dict[str, Any]) -> str:
    required_keys = ["chart_id", "chart_title", "chart_path", "stats"]
    missing_keys = [
        key
        for key in required_keys
        if key not in chart_context or chart_context[key] in (None, "")
    ]
    if missing_keys:
        raise ValueError(f"Missing chart context keys: {', '.join(missing_keys)}")

    stats = chart_context["stats"]
    if not isinstance(stats, dict) or not stats:
        raise ValueError("stats must be a non-empty dict")

    domain = str(chart_context.get("domain", "energy")).strip().lower()
    template = _INSIGHT_TEMPLATES.get(domain, INSIGHT_PROMPT_TEMPLATE)

    formatted_stats = json.dumps(stats, ensure_ascii=False, indent=2)
    return template.format(
        chart_id=str(chart_context["chart_id"]).strip(),
        chart_title=str(chart_context["chart_title"]).strip(),
        chart_path=str(chart_context["chart_path"]).strip(),
        stats=formatted_stats,
    )


ENERGY_CONCLUSION_PROMPT = """Anda adalah analis senior yang meringkas laporan analisis energi.

Tugas:
Tuliskan kesimpulan komprehensif dalam 4-6 paragraf pendek bahasa Indonesia formal akademik
yang mensintesis seluruh temuan dari analisis konsumsi daya listrik rumah tangga di bawah.
Pisahkan setiap paragraf dengan satu baris kosong.

Paragraf 1: Ringkas cakupan data dan gambaran umum konsumsi.
Paragraf 2: Bahas pola beban puncak-lembah harian dan implikasinya pada load shifting.
Paragraf 3: Bahas distribusi daya dan stabilitas tegangan serta artinya.
Paragraf 4: Bahas korelasi antar variabel kelistrikan dan maknanya.
Paragraf 5: Bahas kontribusi sub-metering dan rekomendasi efisiensi tertarget.
Paragraf 6: Tutup dengan signifikansi keseluruhan dan rekomendasi demand side management.

Aturan:
- Gunakan hanya angka dan temuan yang tersedia pada konteks di bawah.
- Jangan mengarang data atau statistik di luar yang diberikan.
- Satukan pola-pola yang saling terkait menjadi narasi yang koheren.
- Gunakan satuan yang benar: daya aktif dalam kW, energi dalam kWh, tegangan dalam V, arus dalam A.
- Jangan menulis markdown, heading, bold, atau tabel.
- Setiap paragraf maksimal 3 kalimat.

Konteks analisis per grafik:
{analysis_context}

Kesimpulan (4-6 paragraf, pisahkan dengan baris kosong):"""


def build_conclusion_prompt(insight_records: list[dict]) -> str:
    segments: list[str] = []
    for idx, record in enumerate(insight_records, start=1):
        if not record.get("success"):
            continue
        chart_title = str(record.get("chart_title", "")).strip()
        insight = str(record.get("insight", "")).strip()
        if not chart_title or not insight:
            continue
        segments.append(f"Grafik {idx}: {chart_title}\n{insight}\n")
    if not segments:
        segments.append(
            "Grafik 1: Tren Rata-rata Daya Aktif Harian\nTidak ada insight tersedia.\n"
        )
    return ENERGY_CONCLUSION_PROMPT.format(analysis_context="\n".join(segments))
