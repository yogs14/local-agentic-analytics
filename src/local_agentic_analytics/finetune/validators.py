"""Automatic validation that rejects ungrounded or unsafe narratives.

A narrative is accepted only when every rule passes:

1. Numeric grounding -- each number in the narrative matches a value present in
   the statistics block, or a whitelisted derived value (load factor = mean/max,
   coefficient of variation = SD/mean, range = max - min), within a rounding
   tolerance.
2. No forbidden tokens -- external standards (IEEE/SPLN/PLN/benchmark), untested
   statistics (p-value/ANOVA/Jarque-Bera), or quantitative future predictions.
3. Unit presence -- at least one number carries an electrical unit (skipped for
   the dimensionless correlation chart).
4. Concept coverage -- at least two concept-bank terms appear.

``validate_narrative`` collects *all* failing reasons (it does not short-circuit)
so ``rejected.csv`` can capture every problem for manual review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from local_agentic_analytics.finetune.concept_bank import count_concept_terms
from local_agentic_analytics.finetune.value_sampler import parse_stats_block


# A standalone number, optionally negative, that is not glued to an identifier
# (so the ``3`` in ``Sub_metering_3`` is ignored). Multiple separator groups are
# captured as one token so grouped thousands like ``2.070.363`` stay intact. The
# sign accepts both ASCII ``-`` and the Unicode minus ``−`` (U+2212) that LLMs
# frequently emit for negative correlation coefficients.
_NUMBER_RE = re.compile(r"(?<![\w.,])[-−]?\d+(?:[.,]\d+)*")

# A number immediately followed by an electrical unit (longest units first).
_UNIT_NEAR_NUMBER_RE = re.compile(
    r"\d[\d.,]*\s*(?:kWh|kVAR|Volt|Ampere|Wh|kW|V|A)(?![A-Za-z])"
)

_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("external_standard", re.compile(r"\b(?:ieee|spln|pln|benchmark)\b", re.IGNORECASE)),
    (
        "statistical_test",
        re.compile(
            r"\b(?:p-value|p\s*<|anova|jarque[\s-]?bera|chi-?square|"
            r"uji\s+hipotesis)\b",
            re.IGNORECASE,
        ),
    ),
)

_FORECAST_VERB_RE = re.compile(
    r"(?:prediksi|proyeksi|forecast|diperkirakan|diprediksi|memprediksi|"
    r"akan\s+(?:naik|turun|meningkat|menurun|berkurang|bertambah))",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")
_FORECAST_WINDOW = 60

_ABS_TOL = 0.02
_REL_TOL = 0.01

# A complete narrative ends on sentence-terminal punctuation. Reasoning models
# that exhaust the token budget leave the answer cut off mid-clause.
_SENTENCE_TERMINATORS = ".!?"
_TRAILING_CLOSERS = "\"'’”»)]} …"


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    concept_terms: int = 0


def _canonical_value(raw: str) -> float | None:
    """Best-effort interpretation of a numeric token (Indonesian locale).

    Comma is the decimal separator; a dot is a thousands separator only when it
    groups three digits or repeats. A single ``X.YYY`` token is read as a decimal
    (matching the comma-decimal block convention); its thousands reading is added
    separately as a grounding candidate.
    """
    raw = raw.replace("−", "-")  # normalize Unicode minus (U+2212) to ASCII
    if "," in raw:
        norm = raw.replace(".", "").replace(",", ".")
    elif raw.count(".") > 1:
        norm = raw.replace(".", "")
    else:
        norm = raw
    try:
        return float(norm)
    except ValueError:
        return None


def _candidate_values(raw: str) -> set[float]:
    """Plausible numeric readings of a token, for tolerant grounding."""
    raw = raw.replace("−", "-")  # normalize Unicode minus (U+2212) to ASCII
    candidates: set[float] = set()
    canonical = _canonical_value(raw)
    if canonical is not None:
        candidates.add(canonical)
    # Ambiguous single-dot, three-digit fraction: could be thousands grouping.
    if "," not in raw and raw.count(".") == 1:
        _, _, fraction = raw.partition(".")
        if len(fraction) == 3:
            try:
                candidates.add(float(raw.replace(".", "")))
            except ValueError:
                pass
    return candidates


def extract_numbers(text: str) -> list[float]:
    """Extract standalone numeric literals (negatives allowed) from ``text``."""
    numbers: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        value = _canonical_value(match.group())
        if value is not None:
            numbers.append(value)
    return numbers


def _numeric_field_map(stats_block: str) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, raw in parse_stats_block(stats_block).items():
        value = _canonical_value(raw)
        if value is not None:
            numeric[key] = value
    return numeric


def derived_values(stats_block: str) -> set[float]:
    """Whitelisted derived quantities computable from the block."""
    numeric = _numeric_field_map(stats_block)
    means = [v for k, v in numeric.items() if k.startswith("avg") or "mean" in k]
    maxima = [v for k, v in numeric.items() if k.startswith("max")]
    minima = [v for k, v in numeric.items() if k.startswith("min")]
    stds = [v for k, v in numeric.items() if "stddev" in k or k.startswith("std")]

    derived: set[float] = set()
    for mean in means:
        if mean == 0:
            continue
        for maximum in maxima:
            if maximum:
                derived.add(mean / maximum)          # load factor
                derived.add(mean / maximum * 100.0)  # load factor (%)
        for std in stds:
            derived.add(std / mean)            # coefficient of variation
            derived.add(std / mean * 100.0)    # CV (%)
    for maximum in maxima:
        for minimum in minima:
            derived.add(maximum - minimum)  # range
    return derived


def build_allowed_values(stats_block: str) -> set[float]:
    """All numbers a faithful narrative may use for this block."""
    allowed = set(extract_numbers(stats_block))
    allowed |= derived_values(stats_block)
    # Meter indices written in prose ("sub-metering 3") are labels, not fabricated
    # quantities — allow 1-3 when the block is about sub-metering.
    if "metering" in stats_block.lower():
        allowed |= {1.0, 2.0, 3.0}
    return allowed


def is_truncated(text: str) -> bool:
    """True when the narrative is cut off (no sentence-terminal punctuation)."""
    core = text.rstrip().rstrip(_TRAILING_CLOSERS)
    if not core:
        return True
    return core[-1] not in _SENTENCE_TERMINATORS


def is_grounded(value: float, allowed: set[float]) -> bool:
    for candidate in allowed:
        if abs(value - candidate) <= max(_ABS_TOL, _REL_TOL * abs(candidate)):
            return True
    return False


def find_forbidden_tokens(text: str) -> list[str]:
    """Return reason strings for any forbidden token found."""
    reasons: list[str] = []
    for label, pattern in _FORBIDDEN_PATTERNS:
        match = pattern.search(text)
        if match:
            reasons.append(f"forbidden:{label}:{match.group().strip()}")

    verb_spans = [m.start() for m in _FORECAST_VERB_RE.finditer(text)]
    pct_spans = [m.start() for m in _PERCENT_RE.finditer(text)]
    for verb_pos in verb_spans:
        for pct_pos in pct_spans:
            if abs(verb_pos - pct_pos) <= _FORECAST_WINDOW:
                reasons.append("forbidden:future_prediction")
                return reasons
    return reasons


def has_unit_near_number(text: str) -> bool:
    return _UNIT_NEAR_NUMBER_RE.search(text) is not None


def validate_narrative(
    narrative: str,
    stats_block: str,
    chart_id: str | None = None,
    *,
    require_unit: bool = True,
    min_concept_terms: int = 2,
) -> ValidationResult:
    """Validate one narrative against its statistics block."""
    text = narrative.strip()
    if not text:
        return ValidationResult(ok=False, reasons=["empty"])

    reasons: list[str] = []

    if is_truncated(text):
        reasons.append("truncated")

    allowed = build_allowed_values(stats_block)
    ungrounded: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        candidates = _candidate_values(match.group())
        if not candidates:
            continue
        if not any(is_grounded(candidate, allowed) for candidate in candidates):
            canonical = _canonical_value(match.group())
            token = f"{canonical:g}" if canonical is not None else match.group()
            if token not in ungrounded:
                ungrounded.append(token)
    reasons.extend(f"ungrounded_number:{token}" for token in ungrounded)

    reasons.extend(find_forbidden_tokens(text))

    if require_unit and not has_unit_near_number(text):
        reasons.append("missing_unit")

    term_count = count_concept_terms(text)
    if term_count < min_concept_terms:
        reasons.append(f"few_concept_terms:{term_count}")

    return ValidationResult(ok=not reasons, reasons=reasons, concept_terms=term_count)
