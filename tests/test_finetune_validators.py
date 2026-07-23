from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.finetune.narrative_synth import synthesize
from local_agentic_analytics.finetune.validators import (
    build_allowed_values,
    extract_numbers,
    has_unit_near_number,
    is_grounded,
    validate_narrative,
)
from local_agentic_analytics.finetune.value_sampler import (
    format_stats_block,
    sample_variant,
)


def _has_reason(reasons, prefix):
    return any(reason.startswith(prefix) for reason in reasons)


# --- Number extraction ---------------------------------------------------------


def test_extract_numbers_ignores_identifier_digits():
    text = "Sub_metering_3 mencatat rata-rata 6.45 Wh."
    assert extract_numbers(text) == [6.45]


def test_extract_numbers_handles_negative_and_comma_decimal():
    assert extract_numbers("r = -0.47 dan nilai 1,09 kW") == [-0.47, 1.09]


def test_unicode_minus_is_treated_as_negative():
    # LLMs often emit U+2212 instead of ASCII '-' for negative correlations.
    assert extract_numbers("korelasi sebesar −0,4557") == [-0.4557]


def test_correlation_with_unicode_minus_is_grounded():
    block = (
        "chart_id: correlation_heatmap\npair_count: 21\n"
        "strongest_positive_r: 0,9721\nstrongest_negative_r: -0,4557\n"
        "unit_note: tanpa satuan"
    )
    narrative = (
        "Korelasi daya aktif dan intensitas arus 0,9721; tegangan vs intensitas "
        "arus −0,4557 menandakan stabilitas tegangan menurun saat arus naik."
    )
    result = validate_narrative(
        narrative, block, "correlation_heatmap", require_unit=False
    )
    assert result.ok, result.reasons


def test_dot_thousands_in_narrative_grounds_against_integer_block():
    block = (
        "chart_id: power_distribution\n"
        "record_count: 2070363\n"
        "avg_global_active_power_kw: 1,09\n"
        "unit: kW"
    )
    # Real teacher may render the count with Indonesian thousands dots.
    narrative = "Distribusi rata-rata 1,09 kW terbentuk dari 2.070.363 catatan."
    result = validate_narrative(narrative, block, "power_distribution")
    assert result.ok, result.reasons


def test_dot_decimal_grounds_when_block_uses_comma_decimal():
    block = (
        "chart_id: power_distribution\n"
        "max_global_active_power_kw: 9,964\n"
        "avg_global_active_power_kw: 1,09\n"
        "unit: kW"
    )
    # Teacher ignores the comma rule and writes a dot decimal; still grounded.
    narrative = "Distribusi rata-rata 1,09 kW dengan beban puncak 9.964 kW."
    result = validate_narrative(narrative, block, "power_distribution")
    assert result.ok, result.reasons


# --- Happy path ----------------------------------------------------------------


def test_synthesized_narratives_pass_validation():
    for chart_id in (
        "daily_active_power_trend",
        "hourly_consumption_pattern",
        "power_distribution",
        "voltage_distribution",
        "sub_metering_comparison",
    ):
        variant = sample_variant(chart_id, random.Random(42))
        block = format_stats_block(variant)
        narrative = synthesize(variant, random.Random(1))
        result = validate_narrative(narrative, block, chart_id)
        assert result.ok, (chart_id, result.reasons)
        assert result.concept_terms >= 2


def test_correlation_narrative_does_not_require_units():
    variant = sample_variant("correlation_heatmap", random.Random(42))
    block = format_stats_block(variant)
    narrative = synthesize(variant, random.Random(1))

    assert not has_unit_near_number(narrative)
    result = validate_narrative(
        narrative, block, "correlation_heatmap", require_unit=False
    )
    assert result.ok, result.reasons


# --- Rejection: each bad category ---------------------------------------------

_BLOCK = (
    "chart_id: power_distribution\n"
    "avg_global_active_power_kw: 1.09\n"
    "min_global_active_power_kw: 0.1\n"
    "max_global_active_power_kw: 9.5\n"
    "unit: kW"
)


def test_rejects_ungrounded_number():
    narrative = "Distribusi menunjukkan rata-rata 1.09 kW namun memuncak di 35 kW."
    result = validate_narrative(narrative, _BLOCK, "power_distribution")
    assert not result.ok
    assert _has_reason(result.reasons, "ungrounded_number:35")


def test_rejects_external_standard():
    narrative = "Mengacu pada standar IEEE, rata-rata distribusi 1.09 kW."
    result = validate_narrative(narrative, _BLOCK, "power_distribution")
    assert not result.ok
    assert _has_reason(result.reasons, "forbidden:external_standard")


def test_rejects_statistical_test():
    narrative = "Hasil uji ANOVA pada distribusi rata-rata 1.09 kW signifikan."
    result = validate_narrative(narrative, _BLOCK, "power_distribution")
    assert not result.ok
    assert _has_reason(result.reasons, "forbidden:statistical_test")


def test_rejects_future_quantitative_prediction():
    narrative = (
        "Distribusi rata-rata 1.09 kW dan konsumsi diprediksi akan turun 15% tahun depan."
    )
    result = validate_narrative(narrative, _BLOCK, "power_distribution")
    assert not result.ok
    assert _has_reason(result.reasons, "forbidden:future_prediction")


def test_rejects_missing_unit():
    narrative = "Distribusi menunjukkan rata-rata 1.09 tanpa menyebut satuan apa pun."
    result = validate_narrative(narrative, _BLOCK, "power_distribution")
    assert not result.ok
    assert "missing_unit" in result.reasons


def test_rejects_too_few_concept_terms():
    narrative = "Nilai tercatat 1.09 kW pada periode tersebut."
    result = validate_narrative(narrative, _BLOCK, "power_distribution")
    assert not result.ok
    assert _has_reason(result.reasons, "few_concept_terms")


# --- Derived-value whitelist ---------------------------------------------------


def test_accepts_whitelisted_derived_load_factor():
    block = (
        "chart_id: daily_active_power_trend\n"
        "mean_daily_avg_kw: 1.0\n"
        "min_daily_avg_kw: 0.2\n"
        "max_daily_avg_kw: 4.0\n"
        "unit: kW"
    )
    # load factor = mean / max = 1.0 / 4.0 = 0.25
    narrative = (
        "Tren rata-rata daya aktif 1.0 kW dengan beban puncak 4.0 kW, "
        "menghasilkan load factor sekitar 0.25."
    )
    result = validate_narrative(narrative, block, "daily_active_power_trend")
    assert result.ok, result.reasons


def test_rejects_truncated_narrative():
    # Ends mid-clause with no terminal punctuation (reasoning model ran out of budget).
    narrative = "Distribusi tegangan rata-rata 240 Volt dan rentang beban yang lebar ini menghasilkan"
    block = "chart_id: voltage_distribution\navg_voltage_v: 240\nunit: Volt"
    result = validate_narrative(narrative, block, "voltage_distribution")
    assert not result.ok
    assert "truncated" in result.reasons


def test_complete_narrative_not_flagged_truncated():
    narrative = "Distribusi tegangan menunjukkan rata-rata 240 Volt yang stabil."
    block = "chart_id: voltage_distribution\navg_voltage_v: 240\nunit: Volt"
    result = validate_narrative(narrative, block, "voltage_distribution")
    assert "truncated" not in result.reasons


def test_sub_metering_index_in_prose_is_not_ungrounded():
    block = (
        "chart_id: sub_metering_comparison\n"
        "avg_sub_metering_3_wh: 6.45\n"
        "unit: Wh"
    )
    # "sub-metering 3" (with a space) — the 3 is a meter label, not a fabricated number.
    narrative = (
        "Analisis sub-metering menunjukkan konsumsi energi sub-metering 3 sebesar "
        "6.45 Wh, mengindikasikan distribusi energi antar zona yang timpang."
    )
    result = validate_narrative(narrative, block, "sub_metering_comparison")
    assert result.ok, result.reasons


def test_allowed_values_include_derived_quantities():
    block = (
        "chart_id: x\nmean_x: 1.0\nmax_x: 4.0\nstddev_x: 0.5\nmin_x: 0.2\nunit: kW"
    )
    allowed = build_allowed_values(block)
    assert is_grounded(0.25, allowed)  # load factor 1/4
    assert is_grounded(0.5, allowed)   # CV 0.5/1.0
    assert is_grounded(3.8, allowed)   # range 4.0-0.2
