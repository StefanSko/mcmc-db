from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mcmc_ref.store import DataStore


def _write_draws_parquet(path: Path) -> None:
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

    _write_draws_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path / "nonexistent")
    models = store.list_models()

    assert models == ["example"]


def test_datastore_open_draws_params(tmp_path: Path) -> None:
    draws_dir = tmp_path / "draws"
    meta_dir = tmp_path / "meta"
    draws_dir.mkdir()
    meta_dir.mkdir()

    _write_draws_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    store = DataStore(local_root=tmp_path)
    reader = store.open_draws("example", params=["mu"], chains=[1])
    table = reader.read_all()

    assert table.column_names == ["chain", "draw", "mu"]
    assert table.num_rows == 2


def test_resolve_stan_data_path(tmp_path: Path) -> None:
    stan_data_dir = tmp_path / "stan_data"
    stan_data_dir.mkdir()
    data_file = stan_data_dir / "example.data.json"
    data_file.write_text('{"N": 10}')

    store = DataStore(local_root=tmp_path)
    path = store.resolve_stan_data_path("example")
    assert path == data_file


def test_resolve_stan_data_path_not_found(tmp_path: Path) -> None:
    stan_data_dir = tmp_path / "stan_data"
    stan_data_dir.mkdir()

    store = DataStore(local_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="stan_data not found"):
        store.resolve_stan_data_path("missing_model")


def test_resolve_stan_code_path(tmp_path: Path) -> None:
    stan_code_dir = tmp_path / "stan_code"
    stan_code_dir.mkdir()
    code_file = stan_code_dir / "example.stan"
    code_file.write_text("data { int N; }")

    store = DataStore(local_root=tmp_path)
    path = store.resolve_stan_code_path("example")
    assert path == code_file


def test_resolve_stan_code_path_not_found(tmp_path: Path) -> None:
    stan_code_dir = tmp_path / "stan_code"
    stan_code_dir.mkdir()

    store = DataStore(local_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="stan_code not found"):
        store.resolve_stan_code_path("missing_model")


def test_read_stan_data(tmp_path: Path) -> None:
    stan_data_dir = tmp_path / "stan_data"
    stan_data_dir.mkdir()
    (stan_data_dir / "example.data.json").write_text('{"N": 10, "x": [1, 2, 3]}')

    store = DataStore(local_root=tmp_path)
    data = store.read_stan_data("example")
    assert data == {"N": 10, "x": [1, 2, 3]}


def test_read_stan_code(tmp_path: Path) -> None:
    stan_code_dir = tmp_path / "stan_code"
    stan_code_dir.mkdir()
    (stan_code_dir / "example.stan").write_text("data { int N; }\nparameters { real mu; }")

    store = DataStore(local_root=tmp_path)
    code = store.read_stan_code("example")
    assert code == "data { int N; }\nparameters { real mu; }"


def test_datastore_init_with_only_stan_data(tmp_path: Path) -> None:
    """DataStore should initialize when only stan_data/ exists (no draws/meta)."""
    stan_data_dir = tmp_path / "stan_data"
    stan_data_dir.mkdir()
    (stan_data_dir / "example.data.json").write_text('{"N": 10}')

    store = DataStore(local_root=tmp_path)
    data = store.read_stan_data("example")
    assert data == {"N": 10}
