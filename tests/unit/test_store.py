from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

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


def test_datastore_list_models(tmp_path: Path) -> None:
    draws_dir = tmp_path / "draws"
    meta_dir = tmp_path / "meta"
    draws_dir.mkdir()
    meta_dir.mkdir()

    _write_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    store = DataStore(local_root=tmp_path)
    models = store.list_models()

    assert models == ["example"]


def test_datastore_open_draws_params(tmp_path: Path) -> None:
    draws_dir = tmp_path / "draws"
    meta_dir = tmp_path / "meta"
    draws_dir.mkdir()
    meta_dir.mkdir()

    _write_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    store = DataStore(local_root=tmp_path)
    reader = store.open_draws("example", params=["mu"], chains=[1])
    table = reader.read_all()

    assert table.column_names == ["chain", "draw", "mu"]
    assert table.num_rows == 2
