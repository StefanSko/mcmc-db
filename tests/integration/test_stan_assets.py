"""Integration tests verifying all 5 models have complete bundled assets."""

from __future__ import annotations

import math
from pathlib import Path

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
    package_root = (
        Path(__file__).resolve().parents[2]
        / "packages"
        / "mcmc-ref-data"
        / "src"
        / "mcmc_ref_data"
        / "data"
    )
    disabled_local_root = Path(__file__).resolve().parent / "._package_only_no_local_store"
    return DataStore(local_root=disabled_local_root, packaged_root=package_root)


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


def _zscore(values: list[float], *, mean: float, std: float) -> list[float]:
    return [(value - mean) / std for value in values]


def _population_mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def _population_std(values: list[float]) -> float:
    mean = _population_mean(values)
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return math.sqrt(variance)


def test_radon_informed_standardization_metadata_matches_packaged_data(
    store: DataStore,
) -> None:
    raw = reference.stan_data("radon_pooled", store=store)
    meta = store.read_meta("radon_pooled_informed")
    info = meta["informed_reference_info"]["standardization"]

    floor_raw = [float(value) for value in raw["floor_measure"]]
    log_radon_raw = [float(value) for value in raw["log_radon"]]

    floor_std = _zscore(
        floor_raw,
        mean=float(info["floor_measure"]["mean"]),
        std=float(info["floor_measure"]["std"]),
    )
    log_radon_std = _zscore(
        log_radon_raw,
        mean=float(info["log_radon"]["mean"]),
        std=float(info["log_radon"]["std"]),
    )

    assert abs(_population_mean(floor_std)) < 1e-6
    assert abs(_population_mean(log_radon_std)) < 1e-6
    assert abs(_population_std(floor_std) - 1.0) < 1e-6
    assert abs(_population_std(log_radon_std) - 1.0) < 1e-6


def test_radon_informed_stan_data_is_prestandardized(store: DataStore) -> None:
    data = reference.stan_data("radon_pooled_informed", store=store)
    assert set(data.keys()) == {"N", "floor_measure_std", "log_radon_std"}

    n = int(data["N"])
    floor = [float(value) for value in data["floor_measure_std"]]
    log_radon = [float(value) for value in data["log_radon_std"]]
    assert len(floor) == n
    assert len(log_radon) == n

    assert abs(_population_mean(floor)) < 1e-6
    assert abs(_population_mean(log_radon)) < 1e-6
    assert abs(_population_std(floor) - 1.0) < 1e-6
    assert abs(_population_std(log_radon) - 1.0) < 1e-6
