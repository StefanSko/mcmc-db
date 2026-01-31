from __future__ import annotations

from pathlib import Path

from mcmc_ref import convert


def test_convert_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "draws.csv"
    csv_path.write_text("chain,draw,mu,tau\n0,0,1.0,2.0\n0,1,1.1,2.1\n1,0,0.9,2.2\n1,1,1.2,2.3\n")

    out_draws = tmp_path / "draws"
    out_meta = tmp_path / "meta"
    out_draws.mkdir()
    out_meta.mkdir()

    convert.convert_file(
        csv_path, name="example", out_draws_dir=out_draws, out_meta_dir=out_meta, force=True
    )

    assert (out_draws / "example.draws.parquet").exists()
    assert (out_meta / "example.meta.json").exists()
