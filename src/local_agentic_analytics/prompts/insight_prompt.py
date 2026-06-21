"""Prompt template for chart insight generation."""

from __future__ import annotations

import json
from typing import Any


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
