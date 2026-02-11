from __future__ import annotations

import importlib.resources
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mcmc_ref import store as store_mod
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

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path)
    models = store.list_models()

    assert "example" in models


def test_datastore_open_draws_params(tmp_path: Path) -> None:
    draws_dir = tmp_path / "draws"
    meta_dir = tmp_path / "meta"
    draws_dir.mkdir()
    meta_dir.mkdir()

    _write_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path)
    reader = store.open_draws("example", params=["mu"], chains=[1])
    table = reader.read_all()

    assert table.column_names == ["chain", "draw", "mu"]
    assert table.num_rows == 2


def test_default_packaged_root_falls_back_to_data_package(tmp_path: Path, monkeypatch) -> None:
    core_pkg = tmp_path / "core_pkg"
    data_pkg = tmp_path / "data_pkg"
    (core_pkg / "data").mkdir(parents=True)
    (data_pkg / "data" / "draws").mkdir(parents=True)
    (data_pkg / "data" / "meta").mkdir(parents=True)

    def fake_files(package_name: str):
        mapping = {
            "mcmc_ref": core_pkg,
            "mcmc_ref_data": data_pkg,
        }
        return mapping[package_name]

    monkeypatch.setattr(importlib.resources, "files", fake_files)

    root = store_mod._default_packaged_root()

    assert root == data_pkg / "data"


def test_default_packaged_root_prefers_core_package_when_present(
    tmp_path: Path, monkeypatch
) -> None:
    core_pkg = tmp_path / "core_pkg"
    data_pkg = tmp_path / "data_pkg"
    (core_pkg / "data" / "draws").mkdir(parents=True)
    (core_pkg / "data" / "meta").mkdir(parents=True)
    (data_pkg / "data" / "draws").mkdir(parents=True)
    (data_pkg / "data" / "meta").mkdir(parents=True)

    def fake_files(package_name: str):
        mapping = {
            "mcmc_ref": core_pkg,
            "mcmc_ref_data": data_pkg,
        }
        return mapping[package_name]

    monkeypatch.setattr(importlib.resources, "files", fake_files)

    root = store_mod._default_packaged_root()

    assert root == core_pkg / "data"
