"""Deterministic grounded-narrative synthesizer for the fake teacher client.

This stands in for a real teacher LLM during offline runs and tests. It reads a
parsed statistics block and emits a short Indonesian narrative that, by
construction, satisfies the validator: every number traces to the block (or to a
whitelisted derived value), units sit next to numbers, no forbidden tokens
appear, and at least two concept-bank terms are used. Light template variety is
driven by an injected RNG so the corpus is not perfectly uniform.
"""

from __future__ import annotations

import random
from typing import Any

from local_agentic_analytics.finetune.value_sampler import parse_stats_block


_MONTHS_ID = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]


def _num(raw: str) -> float:
    cleaned = raw.strip()
    if "," in cleaned and "." in cleaned:
        # Last separator is the decimal point.
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    return float(cleaned)


def _fmt(value: float, digits: int = 2) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    # Indonesian comma decimal, matching the statistics block.
    return f"{value:.{digits}f}".rstrip("0").rstrip(".").replace(".", ",")


def _fmt_full(value: float) -> str:
    """Format preserving up to 4 decimals (for raw block values)."""
    return _fmt(value, 4)


def _id_date(iso: str) -> str:
    parts = iso.split("-")
    if len(parts) != 3:
        return iso
    year, month, day = parts
    try:
        return f"{int(day)} {_MONTHS_ID[int(month) - 1]} {int(year)}"
    except (ValueError, IndexError):
        return iso


def _daily(f: dict[str, str], rng: random.Random) -> str:
    mean = _num(f["mean_daily_avg_kw"])
    minimum = _num(f["min_daily_avg_kw"])
    maximum = _num(f["max_daily_avg_kw"])
    day_count = f["day_count"]
    span = _fmt(maximum - minimum)
    load_factor = _fmt(mean / maximum) if maximum else "0"
    first = _id_date(f["first_date"])
    last = _id_date(f["last_date"])
    max_date = _id_date(f["max_daily_avg_date"])
    min_date = _id_date(f["min_daily_avg_date"])

    openings = [
        "Tren rata-rata daya aktif harian",
        "Kurva beban harian daya aktif",
    ]
    return (
        f"{rng.choice(openings)} memperlihatkan beban rata-rata sebesar "
        f"{_fmt_full(mean)} kW selama {day_count} hari pengamatan "
        f"({first} hingga {last}). Beban puncak harian tercatat "
        f"{_fmt_full(maximum)} kW pada {max_date}, sementara base load terendah "
        f"{_fmt_full(minimum)} kW pada {min_date}, sehingga rentang konsumsi "
        f"mencapai {span} kW. Pola ini konsisten dengan kurva beban rumah tangga "
        f"dengan load factor sekitar {load_factor}."
    )


def _hourly(f: dict[str, str], rng: random.Random) -> str:
    mean = _num(f["mean_hourly_avg_kw"])
    minimum = _num(f["min_hourly_avg_kw"])
    maximum = _num(f["max_hourly_avg_kw"])
    max_hour = int(_num(f["max_avg_hour"]))
    min_hour = int(_num(f["min_avg_hour"]))
    load_factor = _fmt(mean / maximum) if maximum else "0"

    return (
        f"Pola konsumsi daya per jam membentuk kurva beban harian dengan "
        f"konsumsi rata-rata {_fmt_full(mean)} kW. Beban puncak terjadi pada "
        f"pukul {max_hour}.00 dengan rata-rata {_fmt_full(maximum)} kW, "
        f"sedangkan base load terendah {_fmt_full(minimum)} kW pada pukul "
        f"{min_hour}.00. Disparitas puncak-lembah ini berpotensi mendukung "
        f"load shifting, dengan load factor sekitar {load_factor}."
    )


def _power(f: dict[str, str], rng: random.Random) -> str:
    avg = _num(f["avg_global_active_power_kw"])
    stddev = _num(f["stddev_global_active_power_kw"])
    minimum = _num(f["min_global_active_power_kw"])
    maximum = _num(f["max_global_active_power_kw"])
    mode_bin = _num(f["mode_bin_kw"])
    cv_pct = _fmt((stddev / avg) * 100.0) if avg else "0"

    return (
        f"Distribusi daya aktif global terbentuk dari {f['record_count']} catatan "
        f"dengan nilai rata-rata {_fmt_full(avg)} kW dan simpangan baku "
        f"{_fmt_full(stddev)} kW (koefisien variasi sekitar {cv_pct}%). Nilai "
        f"berkisar dari {_fmt_full(minimum)} kW hingga {_fmt_full(maximum)} kW, "
        f"dengan modus pada bin {_fmt_full(mode_bin)} kW yang memuat "
        f"{f['mode_bin_count']} catatan. Sebaran yang relatif lebar ini "
        f"mengindikasikan keberadaan outlier pada beban tinggi."
    )


def _voltage(f: dict[str, str], rng: random.Random) -> str:
    avg = _num(f["avg_voltage_v"])
    stddev = _num(f["stddev_voltage_v"])
    minimum = _num(f["min_voltage_v"])
    maximum = _num(f["max_voltage_v"])
    mode_bin = _num(f["mode_bin_v"])
    cv_pct = _fmt((stddev / avg) * 100.0) if avg else "0"

    return (
        f"Distribusi tegangan menunjukkan nilai rata-rata {_fmt_full(avg)} Volt "
        f"dengan simpangan baku {_fmt_full(stddev)} Volt (koefisien variasi "
        f"sekitar {cv_pct}%), mengindikasikan stabilitas tegangan yang relatif "
        f"baik. Tegangan berkisar antara {_fmt_full(minimum)} Volt dan "
        f"{_fmt_full(maximum)} Volt, dengan modus pada bin {_fmt_full(mode_bin)} "
        f"Volt."
    )


def _correlation(f: dict[str, str], rng: random.Random) -> str:
    pos_r = _num(f["strongest_positive_r"])
    neg_r = _num(f["strongest_negative_r"])
    pos_pair = f["strongest_positive_pair"]
    neg_pair = f["strongest_negative_pair"]

    return (
        f"Heatmap korelasi mengungkap hubungan linear terkuat antara {pos_pair} "
        f"dengan koefisien Pearson r = {_fmt_full(pos_r)}, menandakan keterkaitan "
        f"erat antara daya aktif dan intensitas arus. Sebaliknya, korelasi "
        f"negatif terkuat muncul pada {neg_pair} dengan r = {_fmt_full(neg_r)}, "
        f"mengindikasikan hubungan terbalik antara tegangan dan intensitas arus. "
        f"Pola korelasi lainnya tergolong lemah."
    )


def _sub_metering(f: dict[str, str], rng: random.Random) -> str:
    avg1 = _num(f["avg_sub_metering_1_wh"])
    avg2 = _num(f["avg_sub_metering_2_wh"])
    avg3 = _num(f["avg_sub_metering_3_wh"])
    total3 = _num(f["total_sub_metering_3_wh"])

    return (
        f"Perbandingan sub-metering menunjukkan konsumsi spesifik per zona yang "
        f"timpang. Sub_metering_3 mencatat rata-rata tertinggi sebesar "
        f"{_fmt_full(avg3)} Wh, jauh di atas Sub_metering_1 ({_fmt_full(avg1)} Wh) "
        f"dan Sub_metering_2 ({_fmt_full(avg2)} Wh). Secara kumulatif, total "
        f"energi Sub_metering_3 mencapai {_fmt_full(total3)} Wh, mengindikasikan "
        f"distribusi energi antar zona yang didominasi satu kanal."
    )


_SYNTH_REGISTRY = {
    "daily_active_power_trend": _daily,
    "hourly_consumption_pattern": _hourly,
    "power_distribution": _power,
    "voltage_distribution": _voltage,
    "correlation_heatmap": _correlation,
    "sub_metering_comparison": _sub_metering,
}


def synthesize_from_block(block: str, rng: random.Random | None = None) -> str:
    """Synthesize a grounded narrative from a rendered statistics block."""
    fields = parse_stats_block(block)
    chart_id = fields.get("chart_id", "")
    synth = _SYNTH_REGISTRY.get(chart_id)
    if synth is None:
        raise KeyError(f"No synthesizer registered for chart id: {chart_id!r}")
    if rng is None:
        rng = random.Random(hash(block) & 0xFFFFFFFF)
    return synth(fields, rng)


def synthesize(variant: dict[str, Any], rng: random.Random | None = None) -> str:
    """Synthesize a narrative directly from a variant dict (test convenience)."""
    from local_agentic_analytics.finetune.value_sampler import format_stats_block

    return synthesize_from_block(format_stats_block(variant), rng)
