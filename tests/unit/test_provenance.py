from __future__ import annotations

import json
from pathlib import Path

from mcmc_ref import provenance


def test_model_recipe_registry_contains_issue_models() -> None:
    names = {recipe.name for recipe in provenance.list_model_recipes()}
    assert names == {
        "bangladesh_contraceptive-noncentered",
        "blr",
        "blr_informed",
        "dugongs",
        "earn_height",
        "eight_schools-noncentered",
        "eight_schools_noncentered",
        "glm_binomial",
        "glm_poisson",
        "gp_regression",
        "hierarchical_lr-noncentered",
        "hmm_example",
        "irt_2pl",
        "kidscore_interaction",
        "kidscore_momhs",
        "kidscore_momiq",
        "kidscore_momiq_informed",
        "logearn_height",
        "logearn_height_informed",
        "mesquite_logmesquite",
        "mesquite_logvolume",
        "mesquite_logvolume_informed",
        "neals_funnel-noncentered",
        "radon_hierarchical_intercept_noncentered",
        "radon_pooled",
        "radon_pooled_informed",
        "varying_slopes-noncentered",
        "wells_dist",
    }


def test_pair_recipe_registry_contains_geometry_pairs() -> None:
    names = {recipe.name for recipe in provenance.list_pair_recipes()}
    assert names == {
        "bangladesh_contraceptive",
        "eight_schools",
        "hierarchical_lr",
        "neals_funnel",
        "varying_slopes",
    }


def test_materialize_scaffold_writes_expected_tree(tmp_path: Path) -> None:
    out_root = tmp_path / "scaffold"
    manifest_path = provenance.materialize_scaffold(out_root)

    assert manifest_path.exists()
    assert (out_root / "stan_models" / "dugongs.stan").exists()
    assert (out_root / "stan_models" / "radon_pooled.stan").exists()
    assert (out_root / "stan_models" / "radon_pooled_informed.stan").exists()
    assert (out_root / "stan_data" / "dugongs.json").exists()
    assert (out_root / "stan_data" / "radon_pooled.json").exists()

    pair_root = out_root / "pairs" / "neals_funnel"
    assert (pair_root / "pair.json").exists()
    assert (pair_root / "centered" / "model.stan").exists()
    assert (pair_root / "centered" / "data.json").exists()
    assert (pair_root / "centered" / "model_spec.json").exists()
    assert (pair_root / "noncentered" / "model.stan").exists()
    assert (pair_root / "noncentered" / "data.json").exists()
    assert (pair_root / "noncentered" / "model_spec.json").exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["cmdstan"]["chains"] == 10
    assert manifest["cmdstan"]["iter_sampling"] == 10_000
    assert "stan_models/dugongs.stan" in manifest["files"]
    assert "stan_models/blr_informed.stan" in manifest["files"]
    assert "stan_models/neals_funnel-noncentered.stan" in manifest["files"]


def test_materialize_scaffold_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    manifest_a = provenance.materialize_scaffold(out_a)
    manifest_b = provenance.materialize_scaffold(out_b)

    body_a = json.loads(manifest_a.read_text())
    body_b = json.loads(manifest_b.read_text())
    assert body_a == body_b
