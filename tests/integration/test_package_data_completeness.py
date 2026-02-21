from __future__ import annotations

from pathlib import Path

from mcmc_ref import pairs, reference
from mcmc_ref.store import DataStore


def test_package_data_contains_provenance_manifest_pairs_and_issue_models() -> None:
    package_root = Path("packages/mcmc-ref-data/src/mcmc_ref_data/data")

    assert (package_root / "provenance_manifest.json").exists()
    assert (package_root / "pairs" / "neals_funnel" / "pair.json").exists()
    assert (package_root / "draws" / "dugongs.draws.parquet").exists()
    assert (package_root / "draws" / "radon_pooled.draws.parquet").exists()
    assert (package_root / "meta" / "dugongs.meta.json").exists()
    assert (package_root / "meta" / "radon_pooled.meta.json").exists()
    assert (package_root / "stan_models" / "dugongs.stan").exists()
    assert (package_root / "stan_models" / "radon_pooled.stan").exists()
    assert (package_root / "stan_data" / "dugongs.data.json").exists()
    assert (package_root / "stan_data" / "radon_pooled.data.json").exists()

    store = DataStore(local_root=None, packaged_root=package_root)
    assert len(pairs.list_pairs(store=store)) == 5
    models = reference.list_models(store=store)
    assert "dugongs" in models
    assert "radon_pooled" in models

    draws_dir = package_root / "draws"
    meta_dir = package_root / "meta"
    stan_models_dir = package_root / "stan_models"
    stan_data_dir = package_root / "stan_data"

    draws = {p.name.replace(".draws.parquet", "") for p in draws_dir.glob("*.draws.parquet")}
    meta = {p.name.replace(".meta.json", "") for p in meta_dir.glob("*.meta.json")}
    stan_models = {p.name.replace(".stan", "") for p in stan_models_dir.glob("*.stan")}
    stan_data = {p.name.replace(".data.json", "") for p in stan_data_dir.glob("*.data.json")}

    assert draws - meta == set()
    assert draws - stan_models == set()
    assert draws - stan_data == set()
