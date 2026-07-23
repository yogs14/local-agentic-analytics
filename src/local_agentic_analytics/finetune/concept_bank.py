"""Concept bank loading and term detection.

The concept bank markdown (``data/finetune/concept_bank_kelistrikan.md``) is the
source of truth that gets injected into the teacher prompt. For *validation* we
do not parse the markdown (its tags use emoji that survive poorly through some
encodings); instead we keep a curated vocabulary here that mirrors the bank's
AMAN / HATI-HATI terms. Keeping the list in code makes term detection
deterministic and unit-testable.
"""

from __future__ import annotations

from pathlib import Path

from local_agentic_analytics.core.config import PROJECT_ROOT


# Default location. The file the user placed is ``concept_bank_kelistrikan.md``;
# we also accept a plain ``concept_bank.md`` sibling so the path in the spec keeps
# working if the file is ever renamed.
DEFAULT_CONCEPT_BANK_CANDIDATES = (
    PROJECT_ROOT / "data" / "finetune" / "concept_bank_kelistrikan.md",
    PROJECT_ROOT / "data" / "finetune" / "concept_bank.md",
)


# Curated vocabulary mirroring sections A-C of the concept bank. Each term maps
# to the lowercase substrings that signal its presence in a narrative. Only
# AMAN / HATI-HATI terms live here; HINDARI terms are policed by the validator's
# forbidden-token rules instead.
CONCEPT_TERMS: dict[str, tuple[str, ...]] = {
    # A. Statistics
    "rata-rata": ("rata-rata", "rerata", "mean"),
    "median": ("median",),
    "simpangan baku": ("simpangan baku", "standar deviasi", "deviasi standar"),
    "koefisien variasi": ("koefisien variasi", "coefficient of variation"),
    "distribusi": ("distribusi",),
    "modus": ("modus", "mode bin"),
    "outlier": ("outlier", "pencilan"),
    "korelasi": ("korelasi", "pearson"),
    "tren": ("tren", "trend"),
    "rentang": ("rentang", "range"),
    # B. Load & consumption
    "daya aktif": ("daya aktif",),
    "daya reaktif": ("daya reaktif",),
    "energi": ("energi",),
    "tegangan": ("tegangan", "voltase"),
    "intensitas arus": ("intensitas arus", "arus listrik"),
    "beban puncak": ("beban puncak", "peak load"),
    "base load": ("base load", "beban dasar"),
    "load factor": ("load factor", "faktor beban"),
    "kurva beban": ("kurva beban", "load curve"),
    "profil residensial": ("profil beban", "residensial", "rumah tangga"),
    "disparitas puncak-lembah": (
        "disparitas",
        "puncak-lembah",
        "puncak dan lembah",
    ),
    "power factor": ("power factor", "faktor daya"),
    "sub-metering": ("sub-metering", "sub metering", "sub_metering", "submeter"),
    "konsumsi spesifik per zona": (
        "konsumsi spesifik",
        "per zona",
        "antar zona",
    ),
    # C. Management & implications (qualitative use only)
    "demand response": ("demand response",),
    "load shifting": ("load shifting",),
    "peak shaving": ("peak shaving",),
    "efisiensi energi": ("efisiensi energi",),
    "anomali konsumsi": ("anomali",),
    "stabilitas tegangan": ("stabilitas tegangan",),
}


# Section D: which terms are most relevant per chart. Used by the fake teacher to
# pick chart-appropriate vocabulary; the validator threshold uses the full bank.
CHART_RELEVANT_TERMS: dict[str, tuple[str, ...]] = {
    "daily_active_power_trend": (
        "tren",
        "beban puncak",
        "base load",
        "kurva beban",
        "rentang",
        "rata-rata",
    ),
    "hourly_consumption_pattern": (
        "kurva beban",
        "load factor",
        "base load",
        "profil residensial",
        "disparitas puncak-lembah",
        "load shifting",
    ),
    "power_distribution": (
        "distribusi",
        "rata-rata",
        "simpangan baku",
        "koefisien variasi",
        "modus",
        "outlier",
    ),
    "voltage_distribution": (
        "distribusi",
        "stabilitas tegangan",
        "koefisien variasi",
        "simpangan baku",
        "modus",
    ),
    "correlation_heatmap": (
        "korelasi",
        "daya aktif",
        "intensitas arus",
        "tegangan",
    ),
    "sub_metering_comparison": (
        "sub-metering",
        "konsumsi spesifik per zona",
        "distribusi",
        "energi",
    ),
}


def resolve_concept_bank_path(path: str | Path | None = None) -> Path | None:
    """Return an existing concept bank path, or ``None`` if none is found."""
    if path is not None:
        candidate = Path(path)
        return candidate if candidate.exists() else None
    for candidate in DEFAULT_CONCEPT_BANK_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_concept_bank_text(path: str | Path | None = None) -> str:
    """Load the concept bank markdown for prompt injection.

    Returns an empty string when no file is found so callers (and offline tests)
    never hard-fail on a missing bank.
    """
    resolved = resolve_concept_bank_path(path)
    if resolved is None:
        return ""
    return resolved.read_text(encoding="utf-8", errors="replace")


def find_concept_terms(text: str, chart_id: str | None = None) -> set[str]:
    """Return the set of concept-bank term names present in ``text``.

    When ``chart_id`` is given, only the chart's relevant terms are considered.
    """
    lowered = text.lower()
    if chart_id is not None and chart_id in CHART_RELEVANT_TERMS:
        candidate_terms = CHART_RELEVANT_TERMS[chart_id]
    else:
        candidate_terms = tuple(CONCEPT_TERMS)

    found: set[str] = set()
    for term in candidate_terms:
        for keyword in CONCEPT_TERMS[term]:
            if keyword in lowered:
                found.add(term)
                break
    return found


def count_concept_terms(text: str, chart_id: str | None = None) -> int:
    """Count distinct concept-bank terms present in ``text``."""
    return len(find_concept_terms(text, chart_id))
