from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from click.testing import CliRunner

from mcmc_ref.cli import main


def _write_draws_parquet(path: Path) -> None:
    table = pa.table(
        {
            "chain": pa.array([0, 0, 1, 1, 2, 2, 3, 3], type=pa.int32()),
            "draw": pa.array([0, 1, 0, 1, 0, 1, 0, 1], type=pa.int32()),
            "mu": pa.array([1.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.5, 2.5], type=pa.float64()),
        }
    )
    pq.write_table(table, path)


def test_cli_list(tmp_path: Path, monkeypatch) -> None:
    draws_dir = tmp_path / "draws"
    meta_dir = tmp_path / "meta"
    draws_dir.mkdir()
    meta_dir.mkdir()
    _write_draws_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    # Force CLI to use a temp local store
    monkeypatch.setenv("MCMC_REF_LOCAL_ROOT", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(main, ["list", "--format", "json"])

    assert result.exit_code == 0
    assert "example" in result.output


def test_cli_stats_include_diagnostics(tmp_path: Path, monkeypatch) -> None:
    draws_dir = tmp_path / "draws"
    meta_dir = tmp_path / "meta"
    draws_dir.mkdir()
    meta_dir.mkdir()
    _write_draws_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    monkeypatch.setenv("MCMC_REF_LOCAL_ROOT", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "stats",
            "example",
            "--format",
            "json",
            "--include-diagnostics",
            "--quantile-mode",
            "exact",
        ],
    )

    assert result.exit_code == 0
    assert "rhat" in result.output


def test_cli_compare_passes(tmp_path: Path, monkeypatch) -> None:
    draws_dir = tmp_path / "draws"
    meta_dir = tmp_path / "meta"
    draws_dir.mkdir()
    meta_dir.mkdir()
    _write_draws_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    actual_csv = tmp_path / "actual.csv"
    actual_csv.write_text("chain,draw,mu\n0,0,1.0\n0,1,2.0\n1,0,1.5\n1,1,2.5\n")

    monkeypatch.setenv("MCMC_REF_LOCAL_ROOT", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(main, ["compare", "example", "--actual", str(actual_csv)])

    assert result.exit_code == 0
    assert "passed" in result.output


def test_cli_data(tmp_path: Path, monkeypatch) -> None:
    stan_data_dir = tmp_path / "stan_data"
    stan_data_dir.mkdir()
    (stan_data_dir / "example.data.json").write_text('{"N": 10, "x": [1, 2, 3]}')

    monkeypatch.setenv("MCMC_REF_LOCAL_ROOT", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(main, ["data", "example"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {"N": 10, "x": [1, 2, 3]}


def test_cli_model_code(tmp_path: Path, monkeypatch) -> None:
    stan_code_dir = tmp_path / "stan_code"
    stan_code_dir.mkdir()
    (stan_code_dir / "example.stan").write_text("data { int N; }")

    monkeypatch.setenv("MCMC_REF_LOCAL_ROOT", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(main, ["model-code", "example"])

    assert result.exit_code == 0
    assert "data { int N; }" in result.output
