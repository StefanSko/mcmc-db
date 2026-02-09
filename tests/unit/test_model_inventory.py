from __future__ import annotations

from pathlib import Path

from mcmc_ref.model_inventory import reference_models_from_draws, split_informed_models


def test_reference_models_from_draws(tmp_path: Path) -> None:
    draws_dir = tmp_path / "draws"
    draws_dir.mkdir()
    (draws_dir / "z_model.draws.parquet").write_bytes(b"")
    (draws_dir / "a_model.draws.parquet").write_bytes(b"")
    (draws_dir / "ignore.txt").write_text("x")

    models = reference_models_from_draws(draws_dir)

    assert models == ["a_model", "z_model"]


def test_split_informed_models() -> None:
    models = ["foo", "bar_informed", "baz", "qux_informed"]
    standard, informed = split_informed_models(models)

    assert standard == ["foo", "baz"]
    assert informed == ["bar_informed", "qux_informed"]
