from __future__ import annotations

from mcmc_ref.compare import compare_stats


def test_compare_stats_passes() -> None:
    ref = {"mu": {"mean": 1.0, "std": 1.0}}
    actual = {"mu": {"mean": 1.05, "std": 0.95}}

    result = compare_stats(ref, actual, tolerance=0.1, metrics=["mean", "std"])

    assert result.passed is True
    assert result.failures == []
    assert result.details["mu"]["mean"].passed is True


def test_compare_stats_fails() -> None:
    ref = {"mu": {"mean": 1.0, "std": 1.0}}
    actual = {"mu": {"mean": 2.0, "std": 1.0}}

    result = compare_stats(ref, actual, tolerance=0.1, metrics=["mean", "std"])

    assert result.passed is False
    assert result.failures
