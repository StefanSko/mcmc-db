from __future__ import annotations

from mcmc_ref.diagnostics import ess_bulk, split_rhat


def test_split_rhat_identical_chains() -> None:
    chains = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    rhat = split_rhat(chains)
    assert 0.99 <= rhat <= 1.01


def test_ess_positive() -> None:
    chains = [
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 2.1, 3.1, 4.1],
    ]
    ess = ess_bulk(chains)
    assert ess > 0


def test_split_rhat_detects_scale_diff() -> None:
    chains = [
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
    ]
    rhat = split_rhat(chains)
    assert rhat > 1.1
