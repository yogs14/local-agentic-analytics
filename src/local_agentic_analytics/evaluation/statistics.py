"""Statistical validity helpers for benchmark rates.

- Bootstrap 95% confidence intervals for success rates (percentile method
  over resampled binary outcomes).
- Exact McNemar test for PAIRED comparisons between two systems evaluated on
  the same questions (uses only the discordant pairs, exact binomial).

Used by ``scripts/summarize_model_benchmark.py`` to attach CI columns and
p-values versus a baseline model. Never fabricates data: everything is
computed from per-question outcomes read from the run CSVs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42


def bootstrap_rate_ci(
    outcomes: Sequence[bool],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Percentile bootstrap CI for the success rate of binary outcomes.

    Returns ``{"n", "rate", "ci_low", "ci_high"}``; the CI bounds are ``None``
    when there are no outcomes.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    values = np.asarray([bool(outcome) for outcome in outcomes], dtype=float)
    n = int(values.size)
    if n == 0:
        return {"n": 0, "rate": None, "ci_low": None, "ci_high": None}

    rate = float(values.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    resampled_rates = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(resampled_rates, [alpha, 1.0 - alpha])
    return {
        "n": n,
        "rate": rate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def mcnemar_exact(only_a: int, only_b: int) -> dict[str, Any]:
    """Exact McNemar test from the two discordant counts.

    ``only_a`` = questions system A got right and B got wrong; ``only_b`` the
    reverse. Concordant pairs carry no information for McNemar. Returns the
    two-sided exact binomial p-value (1.0 when there are no discordant pairs).
    """
    if only_a < 0 or only_b < 0:
        raise ValueError("discordant counts must be non-negative")

    discordant = only_a + only_b
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(min(only_a, only_b), discordant, 0.5,
                      alternative="two-sided").pvalue
        )
    return {
        "only_a": only_a,
        "only_b": only_b,
        "n_discordant": discordant,
        "p_value": p_value,
        "method": "mcnemar_exact_binomial",
    }


def paired_discordant_counts(
    outcomes_a: Mapping[str, bool],
    outcomes_b: Mapping[str, bool],
) -> tuple[int, int, int]:
    """Count (only_a, only_b, n_common) over the shared question ids."""
    common = set(outcomes_a) & set(outcomes_b)
    only_a = sum(
        1 for qid in common if outcomes_a[qid] and not outcomes_b[qid]
    )
    only_b = sum(
        1 for qid in common if outcomes_b[qid] and not outcomes_a[qid]
    )
    return only_a, only_b, len(common)


def mcnemar_vs_baseline(
    outcomes: Mapping[str, bool],
    baseline_outcomes: Mapping[str, bool],
) -> dict[str, Any]:
    """Exact McNemar test of a system against the baseline (paired by id)."""
    only_system, only_baseline, n_common = paired_discordant_counts(
        outcomes, baseline_outcomes
    )
    result = mcnemar_exact(only_system, only_baseline)
    result["n_common"] = n_common
    return result


def majority_vote(outcome_runs: Sequence[Mapping[str, bool]]) -> dict[str, bool]:
    """Combine per-run outcome maps into one map by majority vote per id.

    Ties (possible with an even number of runs) count as failure, the
    conservative choice.
    """
    counts: dict[str, list[int]] = {}
    for run in outcome_runs:
        for qid, outcome in run.items():
            entry = counts.setdefault(str(qid), [0, 0])
            entry[0] += 1 if outcome else 0
            entry[1] += 1
    return {
        qid: successes * 2 > total
        for qid, (successes, total) in counts.items()
    }
