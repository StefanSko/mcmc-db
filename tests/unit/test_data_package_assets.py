from __future__ import annotations

from pathlib import Path

MODELS = [
    "wells_data-wells_dist",
    "GLM_Binomial_data-GLM_Binomial_model",
    "GLM_Poisson_Data-GLM_Poisson_model",
    "radon_mn-radon_hierarchical_intercept_noncentered",
    "irt_2pl-irt_2pl",
]


def test_packaged_corpus_contains_required_models() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "packages"
        / "mcmc-ref-data"
        / "src"
        / "mcmc_ref_data"
        / "data"
    )

    for model in MODELS:
        assert (root / "draws" / f"{model}.draws.parquet").exists()
        assert (root / "meta" / f"{model}.meta.json").exists()
        assert (root / "stan_data" / f"{model}.data.json").exists()
        assert (root / "stan_models" / f"{model}.stan").exists()
