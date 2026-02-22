from __future__ import annotations

import pyarrow as pa
import pytest


def _make_table() -> pa.Table:
    return pa.table(
        {
            "chain": pa.array([0] * 100, type=pa.int32()),
            "draw": pa.array(list(range(100)), type=pa.int32()),
            "mu": pa.array([float(i) * 0.1 for i in range(100)], type=pa.float64()),
        }
    )


def test_arrow_and_numpy_backends_produce_same_std() -> None:
    """Both backends must agree on stddev (ddof=0)."""
    pytest.importorskip("numpy")
    from mcmc_ref.backends_arrow import ArrowBackend
    from mcmc_ref.backends_numpy import NumpyBackend

    table = _make_table()
    arrow_stats = ArrowBackend().stats(table, ["mu"])
    numpy_stats = NumpyBackend().stats(table, ["mu"])

    assert arrow_stats["mu"]["mean"] == pytest.approx(numpy_stats["mu"]["mean"], rel=1e-10)
    assert arrow_stats["mu"]["std"] == pytest.approx(numpy_stats["mu"]["std"], rel=1e-10)
