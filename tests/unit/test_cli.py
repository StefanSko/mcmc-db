from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from click.testing import CliRunner

from mcmc_ref.cli import main


def _write_parquet(path: Path) -> None:
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
    _write_parquet(draws_dir / "example.draws.parquet")
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
    _write_parquet(draws_dir / "example.draws.parquet")
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
    _write_parquet(draws_dir / "example.draws.parquet")
    (meta_dir / "example.meta.json").write_text("{}")

    actual_csv = tmp_path / "actual.csv"
    actual_csv.write_text("chain,draw,mu\n0,0,1.0\n0,1,2.0\n1,0,1.5\n1,1,2.5\n")

    monkeypatch.setenv("MCMC_REF_LOCAL_ROOT", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(main, ["compare", "example", "--actual", str(actual_csv)])

    assert result.exit_code == 0
    assert "passed" in result.output


def test_cli_provenance_scaffold(tmp_path: Path) -> None:
    out_root = tmp_path / "provenance"
    runner = CliRunner()
    result = runner.invoke(main, ["provenance-scaffold", "--output-root", str(out_root)])

    assert result.exit_code == 0
    assert (out_root / "provenance_manifest.json").exists()


def test_cli_provenance_generate_and_publish(tmp_path: Path) -> None:
    scaffold_root = tmp_path / "scaffold"
    generated_root = tmp_path / "generated"
    package_root = tmp_path / "package_data"

    runner = CliRunner()
    scaffold_result = runner.invoke(
        main, ["provenance-scaffold", "--output-root", str(scaffold_root)]
    )
    assert scaffold_result.exit_code == 0

    generate_result = runner.invoke(
        main,
        [
            "provenance-generate",
            "--scaffold-root",
            str(scaffold_root),
            "--output-root",
            str(generated_root),
            "--models",
            "dugongs",
            "--fake-runner",
            "--force",
        ],
    )
    assert generate_result.exit_code == 0
    assert (generated_root / "draws" / "dugongs.draws.parquet").exists()

    publish_result = runner.invoke(
        main,
        [
            "provenance-publish",
            "--source-root",
            str(generated_root),
            "--scaffold-root",
            str(scaffold_root),
            "--package-root",
            str(package_root),
        ],
    )
    assert publish_result.exit_code == 0
    assert (package_root / "draws" / "dugongs.draws.parquet").exists()
    assert (package_root / "pairs" / "neals_funnel" / "pair.json").exists()
