from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.evaluation.statistics import (
    bootstrap_rate_ci,
    majority_vote,
    mcnemar_exact,
    mcnemar_vs_baseline,
    paired_discordant_counts,
)


def test_bootstrap_ci_degenerate_all_true():
    result = bootstrap_rate_ci([True] * 20)
    assert result["n"] == 20
    assert result["rate"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0


def test_bootstrap_ci_empty():
    result = bootstrap_rate_ci([])
    assert result == {"n": 0, "rate": None, "ci_low": None, "ci_high": None}


def test_bootstrap_ci_brackets_rate_and_is_deterministic():
    outcomes = [True] * 10 + [False] * 26
    first = bootstrap_rate_ci(outcomes, seed=42)
    second = bootstrap_rate_ci(outcomes, seed=42)

    assert first == second
    assert first["rate"] == pytest.approx(10 / 36)
    assert first["ci_low"] <= first["rate"] <= first["ci_high"]
    assert first["ci_low"] > 0.0
    assert first["ci_high"] < 1.0


def test_bootstrap_ci_validates_arguments():
    with pytest.raises(ValueError):
        bootstrap_rate_ci([True], confidence=1.5)
    with pytest.raises(ValueError):
        bootstrap_rate_ci([True], n_boot=0)


def test_mcnemar_exact_no_discordant_pairs():
    result = mcnemar_exact(0, 0)
    assert result["p_value"] == 1.0
    assert result["n_discordant"] == 0


def test_mcnemar_exact_one_sided_discordance():
    # 5 vs 0 discordant pairs: exact two-sided p = 2 * 0.5^5 = 0.0625.
    result = mcnemar_exact(5, 0)
    assert result["p_value"] == pytest.approx(0.0625)
    assert mcnemar_exact(0, 5)["p_value"] == pytest.approx(0.0625)


def test_mcnemar_exact_rejects_negative():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_paired_discordant_counts_uses_common_ids_only():
    a = {"q1": True, "q2": False, "q3": True, "extra_a": True}
    b = {"q1": False, "q2": False, "q3": True, "extra_b": True}

    only_a, only_b, n_common = paired_discordant_counts(a, b)

    assert only_a == 1  # q1
    assert only_b == 0
    assert n_common == 3


def test_mcnemar_vs_baseline():
    system = {"q1": True, "q2": True, "q3": False}
    baseline = {"q1": False, "q2": True, "q3": False}

    result = mcnemar_vs_baseline(system, baseline)

    assert result["only_a"] == 1
    assert result["only_b"] == 0
    assert result["n_common"] == 3
    assert 0.0 < result["p_value"] <= 1.0


def test_majority_vote():
    runs = [
        {"q1": True, "q2": False},
        {"q1": True, "q2": True},
        {"q1": False, "q2": True},
    ]
    assert majority_vote(runs) == {"q1": True, "q2": True}

    # Even number of runs: a tie counts as failure (conservative).
    tie_runs = [{"q1": True}, {"q1": False}]
    assert majority_vote(tie_runs) == {"q1": False}
