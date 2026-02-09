"""Tests for generated reference draws of new benchmark models.

Validates that reference draws:
  1. Exist as parquet files with correct structure
  2. Have valid metadata files with passing quality checks
  3. Contain expected parameters for each model
  4. Have physically plausible posterior moments
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import pytest

DRAWS_DIR = Path("packages/mcmc-ref-data/src/mcmc_ref_data/data/draws")
META_DIR = Path("packages/mcmc-ref-data/src/mcmc_ref_data/data/meta")

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

EXPECTED_PARAMS: dict[str, list[str]] = {
    "admit-logistic_regression": ["alpha", "beta"],
    "oceanic_tools-poisson_regression": [
        "alpha",
        "beta_pop",
        "beta_contact",
        "beta_interaction",
    ],
    "trolley-ordered_logistic": [
        "beta_action",
        "beta_intention",
        "beta_contact",
        "cutpoints[1]",
        "cutpoints[2]",
        "cutpoints[3]",
        "cutpoints[4]",
        "cutpoints[5]",
    ],
    "oceanic_tools-negative_binomial": [
        "alpha",
        "beta_pop",
        "beta_contact",
        "phi",
    ],
    "manuscripts-zero_inflated_poisson": ["alpha", "theta"],
    "admit_dept-beta_regression": ["alpha", "beta", "kappa"],
    "tadpoles-multilevel_binomial": ["alpha_bar", "sigma"],
    "chimpanzees-multilevel_varying_slopes": [
        "alpha_bar",
        "beta_bar",
        "sigma_alpha",
        "sigma_beta",
        "sigma_gamma",
    ],
    "cafe-multivariate_normal": [
        "alpha_bar",
        "beta_bar",
        "sigma",
        "sigma_cafe[1]",
        "sigma_cafe[2]",
    ],
    "earnings-robust_regression": ["alpha", "beta", "sigma", "nu"],
    "employees-weibull_survival": ["alpha", "beta", "k"],
    "coal_mining-changepoint": ["early_rate", "late_rate", "switchpoint"],
}


@pytest.fixture(params=NEW_MODELS)
def model_name(request: pytest.FixtureRequest) -> str:
    return request.param


def test_draws_parquet_exists(model_name: str) -> None:
    path = DRAWS_DIR / f"{model_name}.draws.parquet"
    assert path.exists(), f"Missing draws parquet: {path}"


def test_meta_json_exists(model_name: str) -> None:
    path = META_DIR / f"{model_name}.meta.json"
    assert path.exists(), f"Missing metadata: {path}"


def test_parquet_has_chain_draw_columns(model_name: str) -> None:
    path = DRAWS_DIR / f"{model_name}.draws.parquet"
    if not path.exists():
        pytest.skip("parquet not yet generated")
    table = pq.read_table(path)
    assert "chain" in table.column_names
    assert "draw" in table.column_names


def test_parquet_has_10k_draws(model_name: str) -> None:
    path = DRAWS_DIR / f"{model_name}.draws.parquet"
    if not path.exists():
        pytest.skip("parquet not yet generated")
    table = pq.read_table(path)
    assert table.num_rows == 10_000, f"Expected 10000 rows, got {table.num_rows}"


def test_parquet_has_10_chains(model_name: str) -> None:
    path = DRAWS_DIR / f"{model_name}.draws.parquet"
    if not path.exists():
        pytest.skip("parquet not yet generated")
    table = pq.read_table(path)
    chains = set(table.column("chain").to_pylist())
    assert len(chains) == 10, f"Expected 10 chains, got {len(chains)}"


def test_parquet_contains_expected_params(model_name: str) -> None:
    path = DRAWS_DIR / f"{model_name}.draws.parquet"
    if not path.exists():
        pytest.skip("parquet not yet generated")
    table = pq.read_table(path)
    expected = EXPECTED_PARAMS.get(model_name, [])
    actual = set(table.column_names)
    for param in expected:
        assert param in actual, f"Missing parameter '{param}' in {model_name}"


def test_meta_quality_checks_pass(model_name: str) -> None:
    path = META_DIR / f"{model_name}.meta.json"
    if not path.exists():
        pytest.skip("metadata not yet generated")
    meta = json.loads(path.read_text())
    checks = meta.get("checks", {})
    for check_name, passed in checks.items():
        assert passed, f"Quality check '{check_name}' failed for {model_name}"


def test_meta_rhat_below_threshold(model_name: str) -> None:
    path = META_DIR / f"{model_name}.meta.json"
    if not path.exists():
        pytest.skip("metadata not yet generated")
    meta = json.loads(path.read_text())
    diag = meta.get("diagnostics", {})
    for param, values in diag.items():
        rhat = values.get("rhat", float("nan"))
        assert not math.isnan(rhat), f"Rhat is NaN for {param} in {model_name}"
        assert rhat < 1.01, f"Rhat {rhat:.4f} >= 1.01 for {param} in {model_name}"


def test_meta_ess_above_threshold(model_name: str) -> None:
    path = META_DIR / f"{model_name}.meta.json"
    if not path.exists():
        pytest.skip("metadata not yet generated")
    meta = json.loads(path.read_text())
    diag = meta.get("diagnostics", {})
    for param, values in diag.items():
        ess_bulk = values.get("ess_bulk", 0.0)
        assert ess_bulk > 400, f"ESS_bulk {ess_bulk:.1f} <= 400 for {param} in {model_name}"


def _mean_of_param(model_name: str, param: str) -> float:
    path = DRAWS_DIR / f"{model_name}.draws.parquet"
    table = pq.read_table(path)
    values = table.column(param).to_pylist()
    return sum(values) / len(values)


class TestLogisticRegressionPosterior:
    """Validate logistic regression posterior makes sense."""

    def test_intercept_plausible(self) -> None:
        path = DRAWS_DIR / "admit-logistic_regression.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_alpha = _mean_of_param("admit-logistic_regression", "alpha")
        # Intercept should be positive (majority admit at x=0 roughly)
        assert -5 < mean_alpha < 5

    def test_slope_negative(self) -> None:
        path = DRAWS_DIR / "admit-logistic_regression.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_beta = _mean_of_param("admit-logistic_regression", "beta")
        # Slope should be negative (higher x -> less admission)
        assert mean_beta < 0


class TestPoissonRegressionPosterior:
    """Validate Poisson regression posterior makes sense."""

    def test_population_effect_positive(self) -> None:
        path = DRAWS_DIR / "oceanic_tools-poisson_regression.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_beta_pop = _mean_of_param("oceanic_tools-poisson_regression", "beta_pop")
        # Larger populations -> more tools
        assert mean_beta_pop > 0


class TestOrderedLogisticPosterior:
    """Validate ordered logistic posterior makes sense."""

    def test_cutpoints_ordered(self) -> None:
        path = DRAWS_DIR / "trolley-ordered_logistic.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        means = [_mean_of_param("trolley-ordered_logistic", f"cutpoints[{i}]") for i in range(1, 6)]
        # Cutpoints should be ordered
        for i in range(len(means) - 1):
            assert means[i] < means[i + 1], (
                f"Cutpoints not ordered: {means[i]:.3f} >= {means[i + 1]:.3f}"
            )


class TestZeroInflatedPoissonPosterior:
    """Validate zero-inflated Poisson posterior."""

    def test_theta_between_0_and_1(self) -> None:
        path = DRAWS_DIR / "manuscripts-zero_inflated_poisson.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_theta = _mean_of_param("manuscripts-zero_inflated_poisson", "theta")
        assert 0 < mean_theta < 1

    def test_zero_inflation_substantial(self) -> None:
        path = DRAWS_DIR / "manuscripts-zero_inflated_poisson.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_theta = _mean_of_param("manuscripts-zero_inflated_poisson", "theta")
        # Data has lots of zeros, so theta should be noticeable
        assert mean_theta > 0.05


class TestMultilevelBinomialPosterior:
    """Validate multilevel binomial posterior."""

    def test_sigma_positive(self) -> None:
        path = DRAWS_DIR / "tadpoles-multilevel_binomial.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_sigma = _mean_of_param("tadpoles-multilevel_binomial", "sigma")
        assert mean_sigma > 0


class TestChangepointPosterior:
    """Validate changepoint posterior."""

    def test_switchpoint_in_range(self) -> None:
        path = DRAWS_DIR / "coal_mining-changepoint.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_sp = _mean_of_param("coal_mining-changepoint", "switchpoint")
        assert 1851 < mean_sp < 1962

    def test_early_rate_higher_than_late(self) -> None:
        path = DRAWS_DIR / "coal_mining-changepoint.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        early = _mean_of_param("coal_mining-changepoint", "early_rate")
        late = _mean_of_param("coal_mining-changepoint", "late_rate")
        # Coal mining disasters decreased over time
        assert early > late


class TestWeibullSurvivalPosterior:
    """Validate Weibull survival posterior."""

    def test_shape_positive(self) -> None:
        path = DRAWS_DIR / "employees-weibull_survival.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_k = _mean_of_param("employees-weibull_survival", "k")
        assert mean_k > 0


class TestRobustRegressionPosterior:
    """Validate Student-t robust regression posterior."""

    def test_nu_moderate(self) -> None:
        path = DRAWS_DIR / "earnings-robust_regression.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_nu = _mean_of_param("earnings-robust_regression", "nu")
        # With outliers in the data, nu should be smallish (heavy tails)
        assert 1 < mean_nu < 100

    def test_beta_positive(self) -> None:
        path = DRAWS_DIR / "earnings-robust_regression.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_beta = _mean_of_param("earnings-robust_regression", "beta")
        # Taller -> higher earnings
        assert mean_beta > 0


class TestMultivariateNormalPosterior:
    """Validate cafe multivariate normal posterior."""

    def test_afternoon_effect_negative(self) -> None:
        path = DRAWS_DIR / "cafe-multivariate_normal.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_beta_bar = _mean_of_param("cafe-multivariate_normal", "beta_bar")
        # Afternoon wait times should be shorter
        assert mean_beta_bar < 0

    def test_correlation_bounded(self) -> None:
        path = DRAWS_DIR / "cafe-multivariate_normal.draws.parquet"
        if not path.exists():
            pytest.skip("draws not generated")
        mean_rho = _mean_of_param("cafe-multivariate_normal", "Rho[1,2]")
        assert -1 < mean_rho < 1
