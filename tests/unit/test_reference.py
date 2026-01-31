from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mcmc_ref import reference
from mcmc_ref.draws import Draws
from mcmc_ref.store import DataStore


def _write_parquet(path: Path) -> None:
    table = pa.table(
        {
            "chain": pa.array([0, 0, 1, 1], type=pa.int32()),
            "draw": pa.array([0, 1, 0, 1], type=pa.int32()),
            "mu": pa.array([1.0, 2.0, 1.5, 2.5], type=pa.float64()),
        }
    )
    pq.write_table(table, path)


def test_reference_draws_returns_wrapper(tmp_path: Path) -> None:
    draws_dir = tmp_path / "draws"
    meta_dir = tmp_path / "meta"
    draws_dir.mkdir()
    meta_dir.mkdir()
    _write_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    store = DataStore(local_root=tmp_path)
    result = reference.draws("example", return_="draws", store=store)

    assert isinstance(result, Draws)


def test_reference_draws_return_arrow(tmp_path: Path) -> None:
    draws_dir = tmp_path / "draws"
    meta_dir = tmp_path / "meta"
    draws_dir.mkdir()
    meta_dir.mkdir()
    _write_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    store = DataStore(local_root=tmp_path)
    result = reference.draws("example", return_="arrow", store=store)

    assert hasattr(result, "schema")


def test_reference_draws_return_list(tmp_path: Path) -> None:
    draws_dir = tmp_path / "draws"
    meta_dir = tmp_path / "meta"
    draws_dir.mkdir()
    meta_dir.mkdir()
    _write_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    store = DataStore(local_root=tmp_path)
    result = reference.draws("example", return_="list", store=store)

    assert isinstance(result, list)
    assert result
