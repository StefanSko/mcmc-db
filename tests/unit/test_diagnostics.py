from __future__ import annotations

import math

import pytest

from mcmc_ref.diagnostics import ess_bulk, split_rhat


def test_split_rhat_identical_chains() -> None:
    chains = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    rhat = split_rhat(chains)
    assert 0.99 <= rhat <= 1.01


def test_ess_positive() -> None:
    chains = [
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 2.1, 3.1, 4.1],
        [0.9, 1.9, 2.9, 3.9],
        [1.05, 2.05, 3.05, 4.05],
    ]
    ess = ess_bulk(chains)
    assert ess > 0


def test_split_rhat_detects_scale_diff() -> None:
    chains = [
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
    ]
    rhat = split_rhat(chains)
    assert rhat > 1.1


def test_split_rhat_requires_four_chains_by_default() -> None:
    chains = [[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1]]
    with pytest.raises(ValueError, match="at least 4 chains"):
        split_rhat(chains)


def test_split_rhat_allows_single_chain_when_explicitly_overridden() -> None:
    chains = [[1.0, 2.0, 3.0, 4.0]]
    rhat = split_rhat(chains, min_chains=1)
    assert math.isnan(rhat)


def test_ess_bulk_requires_four_chains_by_default() -> None:
    chains = [[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1]]
    with pytest.raises(ValueError, match="at least 4 chains"):
        ess_bulk(chains)
