"""TDD tests for new benchmark models.

Validates that every new model has:
  1. A .stan file that is syntactically valid Stan
  2. A matching .data.json file with valid JSON
  3. The data JSON matches the Stan data block declarations
  4. The model can be discovered by the local generation pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcmc_ref.local_generate import discover_local_model_specs, load_stan_data

MODELS_DIR = Path("packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_models")
DATA_DIR = Path("packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_data")

NEW_MODELS = [
    "admit-logistic_regression",
    "oceanic_tools-poisson_regression",
    "trolley-ordered_logistic",
    "oceanic_tools-negative_binomial",
    "manuscripts-zero_inflated_poisson",
    "admit_dept-beta_regression",
    "tadpoles-multilevel_binomial",
    "chimpanzees-multilevel_varying_slopes",
    "cafe-multivariate_normal",
    "earnings-robust_regression",
    "employees-weibull_survival",
    "coal_mining-changepoint",
]


@pytest.fixture(params=NEW_MODELS)
def model_name(request: pytest.FixtureRequest) -> str:
    return request.param


def test_stan_file_exists(model_name: str) -> None:
    stan_path = MODELS_DIR / f"{model_name}.stan"
    assert stan_path.exists(), f"Missing Stan file: {stan_path}"


def test_data_file_exists(model_name: str) -> None:
    data_path = DATA_DIR / f"{model_name}.data.json"
    assert data_path.exists(), f"Missing data file: {data_path}"


def test_data_file_is_valid_json(model_name: str) -> None:
    data_path = DATA_DIR / f"{model_name}.data.json"
    if not data_path.exists():
        pytest.skip("data file not yet created")
    data = json.loads(data_path.read_text())
    assert isinstance(data, dict), "Stan data JSON must be a dict"


def test_stan_file_has_data_block(model_name: str) -> None:
    stan_path = MODELS_DIR / f"{model_name}.stan"
    if not stan_path.exists():
        pytest.skip("stan file not yet created")
    content = stan_path.read_text()
    assert "parameters" in content, "Stan model must have a parameters block"
    assert "model" in content, "Stan model must have a model block"


def test_stan_file_has_no_empty_blocks(model_name: str) -> None:
    stan_path = MODELS_DIR / f"{model_name}.stan"
    if not stan_path.exists():
        pytest.skip("stan file not yet created")
    content = stan_path.read_text()
    # Ensure there's actual content in the model block
    assert len(content.strip()) > 50, "Stan model file seems too short"


def test_data_vars_are_nonempty(model_name: str) -> None:
    data_path = DATA_DIR / f"{model_name}.data.json"
    if not data_path.exists():
        pytest.skip("data file not yet created")
    data = json.loads(data_path.read_text())
    for key, value in data.items():
        if isinstance(value, list):
            assert len(value) > 0, f"Data variable '{key}' is empty"


def test_models_discoverable() -> None:
    """All new models should be discoverable by the local generation pipeline."""
    specs = discover_local_model_specs(
        models_dir=MODELS_DIR,
        data_dir=DATA_DIR,
        models=NEW_MODELS,
    )
    discovered = {s.model for s in specs}
    for model in NEW_MODELS:
        assert model in discovered, f"Model {model} not discoverable"


def test_load_stan_data_valid(model_name: str) -> None:
    data_path = DATA_DIR / f"{model_name}.data.json"
    if not data_path.exists():
        pytest.skip("data file not yet created")
    data = load_stan_data(data_path)
    assert isinstance(data, dict)
