"""Integration tests verifying all 5 models have complete bundled assets."""

from __future__ import annotations

import pytest

from mcmc_ref import reference
from mcmc_ref.store import DataStore

MODELS = [
    "wells_data-wells_dist",
    "GLM_Binomial_data-GLM_Binomial_model",
    "GLM_Poisson_Data-GLM_Poisson_model",
    "radon_mn-radon_hierarchical_intercept_noncentered",
    "irt_2pl-irt_2pl",
]


@pytest.fixture
def store() -> DataStore:
    return DataStore()


@pytest.mark.parametrize("model", MODELS)
def test_model_has_draws(store: DataStore, model: str) -> None:
    path = store.resolve_draws_path(model)
    assert path.exists()


@pytest.mark.parametrize("model", MODELS)
def test_model_has_meta(store: DataStore, model: str) -> None:
    meta = store.read_meta(model)
    assert meta["n_chains"] == 10
    assert meta["n_draws_per_chain"] == 1000
    assert meta["checks"]["rhat_below_1_01"] is True
    assert meta["checks"]["ess_above_400"] is True


@pytest.mark.parametrize("model", MODELS)
def test_model_has_stan_data(store: DataStore, model: str) -> None:
    data = reference.stan_data(model, store=store)
    assert isinstance(data, dict)
    assert len(data) > 0


@pytest.mark.parametrize("model", MODELS)
def test_model_has_stan_code(store: DataStore, model: str) -> None:
    code = reference.model_code(model, store=store)
    assert isinstance(code, str)
    assert "data" in code or "parameters" in code


@pytest.mark.parametrize("model", MODELS)
def test_model_stats_computable(store: DataStore, model: str) -> None:
    stats = reference.stats(model, store=store)
    assert isinstance(stats, dict)
    assert len(stats) > 0
    for _param, metrics in stats.items():
        assert "mean" in metrics
        assert "std" in metrics


@pytest.mark.parametrize("model", MODELS)
def test_model_diagnostics_healthy(store: DataStore, model: str) -> None:
    diag = reference.diagnostics_for_model(model, store=store)
    for param, metrics in diag.items():
        assert metrics["rhat"] < 1.01, f"{model}/{param}: rhat={metrics['rhat']}"
        assert metrics["ess_bulk"] > 400, f"{model}/{param}: ess_bulk={metrics['ess_bulk']}"
