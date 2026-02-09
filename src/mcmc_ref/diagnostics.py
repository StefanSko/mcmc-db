"""Minimal diagnostics: rank-normalized split R-hat and ESS.

Dependency-light versions based on standard recommendations, using only stdlib.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist, median


def split_rhat(
    chains: Sequence[Sequence[float]],
    *,
    min_chains: int = 4,
) -> float:
    """Rank-normalized split R-hat with folded variant (returns max of both).

    By default diagnostics are only valid for at least 4 independent chains.
    Set ``min_chains=1`` to bypass this guard and return
    ``nan`` for single-chain input.
    """
    _validate_min_chains(min_chains)
    if len(chains) < min_chains:
        raise ValueError(
            f"R-hat diagnostics require at least {min_chains} chains; got {len(chains)} chain(s)"
        )
    if len(chains) < 2:
        return float("nan")

    z = _rank_normalize(chains)
    split = _split_chains(z)
    rhat_bulk = _rhat(split)

    folded = _fold_chains(chains)
    z_folded = _rank_normalize(folded)
    split_folded = _split_chains(z_folded)
    rhat_tail = _rhat(split_folded)
    return max(rhat_bulk, rhat_tail)


def ess_bulk(
    chains: Sequence[Sequence[float]],
    *,
    min_chains: int = 4,
) -> float:
    _validate_min_chains(min_chains)
    if len(chains) < min_chains:
        raise ValueError(
            f"ESS diagnostics require at least {min_chains} chains; got {len(chains)} chain(s)"
        )
    if len(chains) < 2:
        return float("nan")
    z = _rank_normalize(chains)
    return _ess(z)


def ess_tail(
    chains: Sequence[Sequence[float]],
    *,
    min_chains: int = 4,
) -> float:
    _validate_min_chains(min_chains)
    if len(chains) < min_chains:
        raise ValueError(
            f"ESS diagnostics require at least {min_chains} chains; got {len(chains)} chain(s)"
        )
    if len(chains) < 2:
        return float("nan")
    folded = _fold_chains(chains)
    z_folded = _rank_normalize(folded)
    return _ess(z_folded)


def _split_chains(chains: Sequence[Sequence[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for chain in chains:
        n = len(chain)
        half = n // 2
        if half == 0:
            continue
        out.append(list(chain[:half]))
        out.append(list(chain[half : half * 2]))
    return out


def _validate_min_chains(min_chains: int) -> None:
    if min_chains < 1:
        raise ValueError(f"min_chains must be >= 1; got {min_chains}")


def _fold_chains(chains: Sequence[Sequence[float]]) -> list[list[float]]:
    flat = [v for chain in chains for v in chain]
    if not flat:
        return []
    med = median(flat)
    return [[abs(v - med) for v in chain] for chain in chains]


def _rank_normalize(chains: Sequence[Sequence[float]]) -> list[list[float]]:
    flat: list[tuple[float, int, int]] = []
    for chain_idx, chain in enumerate(chains):
        for draw_idx, value in enumerate(chain):
            flat.append((float(value), chain_idx, draw_idx))
    n = len(flat)
    if n == 0:
        return []

    flat_sorted = sorted(flat, key=lambda x: x[0])
    ranks: dict[tuple[int, int], float] = {}

    i = 0
    while i < n:
        j = i + 1
        while j < n and flat_sorted[j][0] == flat_sorted[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            _, chain_idx, draw_idx = flat_sorted[k]
            ranks[(chain_idx, draw_idx)] = avg_rank
        i = j

    norm = NormalDist()
    out: list[list[float]] = []
    for chain_idx, chain in enumerate(chains):
        z_chain: list[float] = []
        for draw_idx in range(len(chain)):
            rank = ranks[(chain_idx, draw_idx)]
            p = (rank - 0.5) / n
            z_chain.append(norm.inv_cdf(p))
        out.append(z_chain)
    return out


def _rhat(chains: Sequence[Sequence[float]]) -> float:
    m = len(chains)
    if m < 2:
        return float("nan")
    n = min(len(c) for c in chains)
    if n < 2:
        return float("nan")

    means = [sum(c[:n]) / n for c in chains]
    mean_total = sum(means) / m
    var_between = n * sum((mu - mean_total) ** 2 for mu in means) / (m - 1)
    var_within = sum(_variance(c[:n]) for c in chains) / m
    var_hat = (n - 1) / n * var_within + var_between / n
    if var_within == 0:
        return 1.0 if var_between == 0 else float("inf")
    return math.sqrt(var_hat / var_within)


def _ess(chains: Sequence[Sequence[float]]) -> float:
    m = len(chains)
    if m == 0:
        return float("nan")
    n = min(len(c) for c in chains)
    if n < 2:
        return float("nan")

    chains = [list(c[:n]) for c in chains]
    means = [sum(c) / n for c in chains]
    mean_total = sum(means) / m
    var_between = n * sum((mu - mean_total) ** 2 for mu in means) / (m - 1) if m > 1 else 0.0
    var_within = sum(_variance(c) for c in chains) / m
    var_hat = (n - 1) / n * var_within + var_between / n
    if var_hat == 0:
        return float(m * n)

    rho_sum = 0.0
    for lag in range(1, n):
        rho = _autocorr(chains, lag, var_hat)
        if rho < 0:
            break
        rho_sum += rho
    return m * n / (1 + 2 * rho_sum)


def _autocorr(chains: Sequence[Sequence[float]], lag: int, var_hat: float) -> float:
    m = len(chains)
    n = min(len(c) for c in chains)
    if var_hat == 0:
        return 0.0
    cov_sum = 0.0
    for chain in chains:
        mean = sum(chain[:n]) / n
        cov = 0.0
        for i in range(n - lag):
            cov += (chain[i] - mean) * (chain[i + lag] - mean)
        cov /= n - lag
        cov_sum += cov
    return cov_sum / (m * var_hat)


def _variance(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)
