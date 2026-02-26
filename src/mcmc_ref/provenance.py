"""Deterministic provenance recipes for rebuilding reference assets."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CmdStanConfig:
    chains: int = 10
    iter_sampling: int = 10_000
    iter_warmup: int = 10_000
    thin: int = 10
    seed: int = 4711


@dataclass(frozen=True)
class ModelRecipe:
    name: str
    description: str
    stan_code: str
    stan_data: dict[str, Any]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class PairVariantRecipe:
    name: str
    stan_code: str
    stan_data: dict[str, Any]
    model_spec: dict[str, Any]


@dataclass(frozen=True)
class PairRecipe:
    name: str
    description: str
    bad_variant: str
    good_variant: str
    reference_model: str
    expected_pathologies: tuple[str, ...]
    difficulty: str
    variants: dict[str, PairVariantRecipe]


DEFAULT_CMDSTAN = CmdStanConfig()


def list_model_recipes() -> list[ModelRecipe]:
    """Return model recipes that can be materialized into Stan source + data."""
    return sorted(
        [
            _blr_recipe(),
            _blr_informed_recipe(),
            _dugongs_recipe(),
            _earn_height_recipe(),
            _eight_schools_noncentered_recipe(),
            _glm_binomial_recipe(),
            _glm_poisson_recipe(),
            _gp_regression_recipe(),
            _hmm_example_recipe(),
            _irt_2pl_recipe(),
            _kidscore_interaction_recipe(),
            _kidscore_momhs_recipe(),
            _kidscore_momiq_recipe(),
            _kidscore_momiq_informed_recipe(),
            _logearn_height_recipe(),
            _logearn_height_informed_recipe(),
            _mesquite_logmesquite_recipe(),
            _mesquite_logvolume_recipe(),
            _mesquite_logvolume_informed_recipe(),
            _radon_hierarchical_intercept_noncentered_recipe(),
            _radon_pooled_recipe(),
            _radon_pooled_informed_recipe(),
            _wells_dist_recipe(),
            *_pair_reference_model_recipes(),
        ],
        key=lambda recipe: recipe.name,
    )


def list_pair_recipes() -> list[PairRecipe]:
    """Return geometry pair recipes."""
    return [
        _pair_eight_schools(),
        _pair_neals_funnel(),
        _pair_hierarchical_lr(),
        _pair_varying_slopes(),
        _pair_bangladesh_contraceptive(),
    ]


def materialize_scaffold(output_root: Path) -> Path:
    """Write deterministic provenance scaffold and return manifest path."""
    output_root = Path(output_root)
    stan_models = output_root / "stan_models"
    stan_data = output_root / "stan_data"
    pairs_root = output_root / "pairs"
    stan_models.mkdir(parents=True, exist_ok=True)
    stan_data.mkdir(parents=True, exist_ok=True)
    pairs_root.mkdir(parents=True, exist_ok=True)

    for recipe in list_model_recipes():
        _write_text(stan_models / f"{recipe.name}.stan", recipe.stan_code)
        _write_json(stan_data / f"{recipe.name}.json", recipe.stan_data)

    for recipe in list_pair_recipes():
        pair_root = pairs_root / recipe.name
        pair_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            pair_root / "pair.json",
            {
                "name": recipe.name,
                "description": recipe.description,
                "bad_variant": recipe.bad_variant,
                "good_variant": recipe.good_variant,
                "reference_model": recipe.reference_model,
                "expected_pathologies": list(recipe.expected_pathologies),
                "difficulty": recipe.difficulty,
            },
        )
        for variant_name, variant in recipe.variants.items():
            variant_root = pair_root / variant_name
            variant_root.mkdir(parents=True, exist_ok=True)
            _write_text(variant_root / "model.stan", variant.stan_code)
            _write_json(variant_root / "data.json", variant.stan_data)
            _write_json(variant_root / "model_spec.json", variant.model_spec)

    manifest_path = output_root / "provenance_manifest.json"
    manifest_body = _build_manifest(output_root)
    _write_json(manifest_path, manifest_body)
    return manifest_path


def _build_manifest(output_root: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_root).as_posix()
        if rel == "provenance_manifest.json":
            continue
        files[rel] = _sha256(path.read_bytes())

    return {
        "schema_version": 1,
        "generator": {
            "name": "mcmc-ref",
            "version": _generator_version(),
            "source_commit": _source_commit(),
        },
        "cmdstan": asdict(DEFAULT_CMDSTAN),
        "models": [recipe.name for recipe in list_model_recipes()],
        "pairs": [recipe.name for recipe in list_pair_recipes()],
        "files": files,
    }


def _generator_version() -> str:
    try:
        return importlib.metadata.version("mcmc-ref")
    except Exception:
        return "unknown"


def _source_commit() -> str:
    env_commit = os.environ.get("MCMC_REF_SOURCE_COMMIT")
    if env_commit:
        return env_commit

    try:
        repo_root = Path(__file__).resolve().parents[2]
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"

    commit = proc.stdout.strip()
    return commit or "unknown"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_text(path: Path, body: str) -> None:
    path.write_text(body.rstrip() + "\n")


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


def _dugongs_recipe() -> ModelRecipe:
    n = 27
    x = [1.0 + 0.6 * idx for idx in range(n)]
    y = [4.5 + 6.0 * (1.0 - math.exp(-0.11 * value)) for value in x]
    return ModelRecipe(
        name="dugongs",
        description="Nonlinear growth curve model.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real x;
  array[N] real y;
}
parameters {
  real<lower=0> U3;
  real alpha;
  real beta;
  real<lower=0> lambda;
  real<lower=0> sigma;
  real tau;
}
transformed parameters {
  array[N] real mu;
  for (n in 1:N) {
    mu[n] = U3 - alpha * exp(-lambda * x[n]) + beta;
  }
}
model {
  U3 ~ lognormal(2, 0.3);
  alpha ~ normal(3.0, 1.0);
  beta ~ normal(0.0, 1.0);
  lambda ~ lognormal(-2.0, 0.4);
  tau ~ normal(alpha, 0.5);
  sigma ~ lognormal(-2.0, 0.3);
  y ~ normal(mu, sigma + 0.05 * abs(tau));
}
""",
        stan_data={"N": n, "x": x, "y": y},
        tags=("core", "issue-12"),
    )


def _radon_pooled_data() -> dict[str, Any]:
    n = 12_573
    n_county = 386
    floor_measure = [idx % 2 for idx in range(n)]
    county = [(idx % n_county) + 1 for idx in range(n)]
    log_radon = [
        1.1
        - 0.28 * float(floor_measure[idx])
        + 0.06 * ((county[idx] - 1) / float(n_county))
        + 0.01 * float(idx % 11)
        for idx in range(n)
    ]
    return {
        "N": n,
        "N_county": n_county,
        "county": county,
        "floor_measure": floor_measure,
        "log_radon": log_radon,
    }


def _zscore_population(values: list[float]) -> list[float]:
    n = len(values)
    mean = sum(values) / float(n)
    variance = sum((value - mean) ** 2 for value in values) / float(n)
    std = math.sqrt(variance)
    if std == 0.0:
        raise ValueError("Cannot standardize constant vector")
    return [(value - mean) / std for value in values]


def _radon_pooled_recipe() -> ModelRecipe:
    return ModelRecipe(
        name="radon_pooled",
        description="Pooled radon regression with floor_measure.",
        stan_code="""
data {
  int<lower=1> N;
  int<lower=1> N_county;
  array[N] int<lower=1, upper=N_county> county;
  array[N] int<lower=0, upper=1> floor_measure;
  array[N] real log_radon;
}
parameters {
  real beta_0;
  real beta_1;
  real<lower=0> sigma;
}
model {
  beta_0 ~ normal(0, 2);
  beta_1 ~ normal(0, 1);
  sigma ~ lognormal(-1, 0.5);
  for (n in 1:N) {
    log_radon[n] ~ normal(beta_0 + beta_1 * floor_measure[n], sigma);
  }
}
""",
        stan_data=_radon_pooled_data(),
        tags=("core", "issue-12"),
    )


def _radon_pooled_informed_recipe() -> ModelRecipe:
    raw = _radon_pooled_data()
    floor_measure_std = _zscore_population([float(v) for v in raw["floor_measure"]])
    log_radon_std = _zscore_population([float(v) for v in raw["log_radon"]])
    return ModelRecipe(
        name="radon_pooled_informed",
        description="Informed-prior pooled radon regression.",
        stan_code="""
data {
  int<lower=0> N;
  vector[N] floor_measure_std;
  vector[N] log_radon_std;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma_y;
}
model {
  // Weakly informative priors (after standardization)
  alpha ~ normal(0, 2.5);
  beta ~ normal(0, 2.5);
  sigma_y ~ normal(0, 1);  // half-normal via constraint

  // Likelihood
  log_radon_std ~ normal(alpha + beta * floor_measure_std, sigma_y);
}
""",
        stan_data={
            "N": int(raw["N"]),
            "floor_measure_std": floor_measure_std,
            "log_radon_std": log_radon_std,
        },
        tags=("informed", "issue-12"),
    )


def _pair_reference_model_recipes() -> list[ModelRecipe]:
    recipes: list[ModelRecipe] = []
    for pair in list_pair_recipes():
        variant = pair.variants[pair.good_variant]
        recipes.append(
            ModelRecipe(
                name=pair.reference_model,
                description=f"Geometry reference ({pair.name}, {pair.good_variant}).",
                stan_code=variant.stan_code,
                stan_data=variant.stan_data,
                tags=("geometry", "pair-reference"),
            )
        )
    return recipes


def _eight_schools_noncentered_recipe() -> ModelRecipe:
    pair = _pair_eight_schools()
    variant = pair.variants["noncentered"]
    return ModelRecipe(
        name="eight_schools_noncentered",
        description="Eight Schools non-centered model.",
        stan_code=variant.stan_code,
        stan_data=variant.stan_data,
        tags=("core",),
    )


def _kidscore_momiq_recipe() -> ModelRecipe:
    n = 120
    mom_iq = [80.0 + float(idx % 40) for idx in range(n)]
    kid_score = [35.0 + 0.5 * value + 0.1 * float(idx % 7) for idx, value in enumerate(mom_iq)]
    return ModelRecipe(
        name="kidscore_momiq",
        description="Kid score regression with maternal IQ.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real mom_iq;
  array[N] real kid_score;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 2);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    kid_score[n] ~ normal(beta[1] + beta[2] * mom_iq[n], sigma);
  }
}
""",
        stan_data={"N": n, "mom_iq": mom_iq, "kid_score": kid_score},
        tags=("core",),
    )


def _kidscore_interaction_recipe() -> ModelRecipe:
    n = 160
    mom_hs = [idx % 2 for idx in range(n)]
    mom_iq = [82.0 + float((idx * 3) % 45) for idx in range(n)]
    kid_score = [
        20.0
        + 0.6 * mom_iq[idx]
        + 2.0 * float(mom_hs[idx])
        - 0.03 * mom_iq[idx] * float(mom_hs[idx])
        for idx in range(n)
    ]
    return ModelRecipe(
        name="kidscore_interaction",
        description="Kid score regression with interaction.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real mom_hs;
  array[N] real mom_iq;
  array[N] real kid_score;
}
parameters {
  vector[4] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 2);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    kid_score[n] ~ normal(
      beta[1] + beta[2] * mom_hs[n] + beta[3] * mom_iq[n] + beta[4] * mom_hs[n] * mom_iq[n],
      sigma
    );
  }
}
""",
        stan_data={"N": n, "mom_hs": mom_hs, "mom_iq": mom_iq, "kid_score": kid_score},
        tags=("core",),
    )


def _kidscore_momhs_recipe() -> ModelRecipe:
    n = 120
    mom_hs = [idx % 2 for idx in range(n)]
    kid_score = [
        45.0 + 4.0 * float(flag) + 0.2 * float(idx % 11) for idx, flag in enumerate(mom_hs)
    ]
    return ModelRecipe(
        name="kidscore_momhs",
        description="Kid score regression with maternal high-school indicator.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real mom_hs;
  array[N] real kid_score;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 2);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    kid_score[n] ~ normal(beta[1] + beta[2] * mom_hs[n], sigma);
  }
}
""",
        stan_data={"N": n, "mom_hs": mom_hs, "kid_score": kid_score},
        tags=("core",),
    )


def _earn_height_recipe() -> ModelRecipe:
    n = 150
    height = [145.0 + float(idx % 40) for idx in range(n)]
    earn = [12.0 + 0.35 * value + 0.2 * float(idx % 8) for idx, value in enumerate(height)]
    return ModelRecipe(
        name="earn_height",
        description="Earnings regression on height.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real height;
  array[N] real earn;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 2);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    earn[n] ~ normal(beta[1] + beta[2] * height[n], sigma);
  }
}
""",
        stan_data={"N": n, "height": height, "earn": earn},
        tags=("core",),
    )


def _logearn_height_recipe() -> ModelRecipe:
    n = 150
    height = [145.0 + float(idx % 40) for idx in range(n)]
    log_earn = [2.0 + 0.01 * value + 0.015 * float(idx % 9) for idx, value in enumerate(height)]
    return ModelRecipe(
        name="logearn_height",
        description="Log earnings regression on height.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real height;
  array[N] real log_earn;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    log_earn[n] ~ normal(beta[1] + beta[2] * height[n], sigma);
  }
}
""",
        stan_data={"N": n, "height": height, "log_earn": log_earn},
        tags=("core",),
    )


def _wells_dist_recipe() -> ModelRecipe:
    n = 256
    dist = [0.2 * float(idx % 30) for idx in range(n)]
    switched = [1 if value < 2.8 else 0 for value in dist]
    return ModelRecipe(
        name="wells_dist",
        description="Logistic regression for well switching.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real dist;
  array[N] int<lower=0, upper=1> switched;
}
parameters {
  vector[2] beta;
}
model {
  beta ~ normal(0, 2);
  for (n in 1:N) {
    switched[n] ~ bernoulli_logit(beta[1] + beta[2] * dist[n]);
  }
}
""",
        stan_data={"N": n, "dist": dist, "switched": switched},
        tags=("core",),
    )


def _glm_binomial_recipe() -> ModelRecipe:
    n = 120
    year = [1950.0 + float(idx % 70) for idx in range(n)]
    year_squared = [value * value for value in year]
    trials = [20 + (idx % 5) for idx in range(n)]
    counts = [max(0, min(trials[idx], int(4 + 0.01 * (year[idx] - 1950.0)))) for idx in range(n)]
    return ModelRecipe(
        name="glm_binomial",
        description="Quadratic binomial GLM.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real year;
  array[N] real year_squared;
  array[N] int<lower=0> N_trials;
  array[N] int<lower=0> C;
}
parameters {
  real alpha;
  real beta1;
  real beta2;
}
model {
  alpha ~ normal(0, 2);
  beta1 ~ normal(0, 1);
  beta2 ~ normal(0, 1);
  for (n in 1:N) {
    C[n] ~ binomial_logit(N_trials[n], alpha + beta1 * year[n] + beta2 * year_squared[n]);
  }
}
""",
        stan_data={
            "N": n,
            "year": year,
            "year_squared": year_squared,
            "N_trials": trials,
            "C": counts,
        },
        tags=("core",),
    )


def _glm_poisson_recipe() -> ModelRecipe:
    n = 120
    year = [1950.0 + float(idx % 70) for idx in range(n)]
    counts = [int(6 + 0.03 * (value - 1950.0) + 0.0002 * (value - 1950.0) ** 2) for value in year]
    return ModelRecipe(
        name="glm_poisson",
        description="Cubic Poisson GLM.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real year;
  array[N] int<lower=0> C;
}
parameters {
  real alpha;
  real beta1;
  real beta2;
  real beta3;
}
model {
  alpha ~ normal(0, 2);
  beta1 ~ normal(0, 1);
  beta2 ~ normal(0, 1);
  beta3 ~ normal(0, 1);
  for (n in 1:N) {
    C[n] ~ poisson_log(alpha + beta1 * year[n] + beta2 * square(year[n]) + beta3 * cube(year[n]));
  }
}
""",
        stan_data={"N": n, "year": year, "C": counts},
        tags=("core",),
    )


def _radon_hierarchical_intercept_noncentered_recipe() -> ModelRecipe:
    j = 30
    n = 900
    county_idx = [(idx % j) + 1 for idx in range(n)]
    log_uppm = [0.2 + 0.01 * float(idx % 13) for idx in range(n)]
    floor_measure = [idx % 2 for idx in range(n)]
    log_radon = [
        1.0
        + 0.1 * float(county_idx[idx] / j)
        + 0.2 * log_uppm[idx]
        - 0.3 * float(floor_measure[idx])
        for idx in range(n)
    ]
    return ModelRecipe(
        name="radon_hierarchical_intercept_noncentered",
        description="Hierarchical radon model with non-centered intercepts.",
        stan_code="""
data {
  int<lower=1> J;
  int<lower=1> N;
  array[N] int<lower=1, upper=J> county_idx;
  array[N] real log_uppm;
  array[N] real floor_measure;
  array[N] real log_radon;
}
parameters {
  real mu_alpha;
  real<lower=0> sigma_alpha;
  vector[J] alpha_raw;
  vector[2] beta;
  real<lower=0> sigma_y;
}
transformed parameters {
  vector[J] alpha = mu_alpha + sigma_alpha * alpha_raw;
}
model {
  mu_alpha ~ normal(0, 1);
  sigma_alpha ~ normal(0, 1);
  alpha_raw ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma_y ~ normal(0, 1);
  for (n in 1:N) {
    log_radon[n] ~ normal(
      alpha[county_idx[n]] + beta[1] * log_uppm[n] + beta[2] * floor_measure[n],
      sigma_y
    );
  }
}
""",
        stan_data={
            "J": j,
            "N": n,
            "county_idx": county_idx,
            "log_uppm": log_uppm,
            "floor_measure": floor_measure,
            "log_radon": log_radon,
        },
        tags=("core",),
    )


def _blr_recipe() -> ModelRecipe:
    n = 120
    d = 5
    x = [[1.0 + 0.1 * float((row + col) % 7) for col in range(d)] for row in range(n)]
    y = [
        sum(0.2 * x[row][col] * float(col + 1) for col in range(d)) + 0.05 * float(row % 5)
        for row in range(n)
    ]
    return ModelRecipe(
        name="blr",
        description="Bayesian linear regression.",
        stan_code="""
data {
  int<lower=1> N;
  int<lower=1> D;
  matrix[N, D] X;
  vector[N] y;
}
parameters {
  vector[D] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  y ~ normal(X * beta, sigma);
}
""",
        stan_data={"N": n, "D": d, "X": x, "y": y},
        tags=("core",),
    )


def _hmm_example_recipe() -> ModelRecipe:
    k = 3
    n = 180
    y = [(-1.5 + 1.5 * float(idx % k)) + 0.1 * float((idx * 7) % 5) for idx in range(n)]
    return ModelRecipe(
        name="hmm_example",
        description="Simple latent-state Gaussian emissions example.",
        stan_code="""
data {
  int<lower=1> K;
  int<lower=1> N;
  array[N] real y;
}
parameters {
  simplex[K] theta;
  vector[K] mu;
  vector<lower=0>[K] sigma;
}
model {
  theta ~ dirichlet(rep_vector(1.0, K));
  mu ~ normal(0, 2);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    int s = 1 + ((n - 1) % K);
    y[n] ~ normal(mu[s], sigma[s]);
  }
}
""",
        stan_data={"K": k, "N": n, "y": y},
        tags=("core",),
    )


def _gp_regression_recipe() -> ModelRecipe:
    n = 80
    x = [float(idx) / 10.0 for idx in range(n)]
    y = [0.7 * value + 0.4 * math.sin(value) for value in x]
    return ModelRecipe(
        name="gp_regression",
        description="Lightweight GP-style regression benchmark.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real x;
  array[N] real y;
}
parameters {
  real<lower=0> rho;
  real alpha;
  real<lower=0> sigma;
}
model {
  rho ~ normal(0, 1);
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    y[n] ~ normal(alpha + rho * x[n], sigma);
  }
}
""",
        stan_data={"N": n, "x": x, "y": y},
        tags=("core",),
    )


def _mesquite_logvolume_recipe() -> ModelRecipe:
    n = 140
    x = [[1.0 + 0.2 * float(idx % 9), 0.5 + 0.1 * float((idx * 3) % 11)] for idx in range(n)]
    y = [0.4 + 0.7 * row[0] + 0.3 * row[1] for row in x]
    return ModelRecipe(
        name="mesquite_logvolume",
        description="Mesquite regression with 2 predictors.",
        stan_code="""
data {
  int<lower=1> N;
  matrix[N, 2] X;
  vector[N] y;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  y ~ normal(X * beta, sigma);
}
""",
        stan_data={"N": n, "X": x, "y": y},
        tags=("core",),
    )


def _mesquite_logmesquite_recipe() -> ModelRecipe:
    n = 140
    k = 7
    x = [[0.8 + 0.05 * float((idx + col) % 13) for col in range(k)] for idx in range(n)]
    y = [sum(0.1 * float(col + 1) * row[col] for col in range(k)) for row in x]
    return ModelRecipe(
        name="mesquite_logmesquite",
        description="Mesquite regression with 7 predictors.",
        stan_code="""
data {
  int<lower=1> N;
  matrix[N, 7] X;
  vector[N] y;
}
parameters {
  vector[7] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  y ~ normal(X * beta, sigma);
}
""",
        stan_data={"N": n, "X": x, "y": y},
        tags=("core",),
    )


def _irt_2pl_recipe() -> ModelRecipe:
    i = 12
    j = 40
    y = [[1 if (student + item) % 5 > 1 else 0 for student in range(j)] for item in range(i)]
    return ModelRecipe(
        name="irt_2pl",
        description="2PL item-response model.",
        stan_code="""
data {
  int<lower=1> I;
  int<lower=1> J;
  array[I, J] int<lower=0, upper=1> y;
}
parameters {
  real<lower=0> sigma_theta;
  real<lower=0> sigma_a;
  real mu_b;
  real<lower=0> sigma_b;
  vector[J] theta;
  vector[I] a_raw;
  vector[I] b;
}
transformed parameters {
  vector<lower=0>[I] a;
  for (i in 1:I) {
    a[i] = exp(a_raw[i] * sigma_a);
  }
}
model {
  sigma_theta ~ normal(0, 1);
  sigma_a ~ normal(0, 1);
  mu_b ~ normal(0, 1);
  sigma_b ~ normal(0, 1);
  theta ~ normal(0, sigma_theta);
  a_raw ~ normal(0, 1);
  b ~ normal(mu_b, sigma_b);
  for (i in 1:I) {
    for (j in 1:J) {
      y[i, j] ~ bernoulli_logit(a[i] * (theta[j] - b[i]));
    }
  }
}
""",
        stan_data={"I": i, "J": j, "y": y},
        tags=("core",),
    )


def _logearn_height_informed_recipe() -> ModelRecipe:
    n = 150
    height_std = [(-2.0 + 4.0 * float(idx) / float(n - 1)) for idx in range(n)]
    log_earn_std = [
        0.2 + 0.4 * value + 0.05 * float(idx % 5) for idx, value in enumerate(height_std)
    ]
    return ModelRecipe(
        name="logearn_height_informed",
        description="Informed log-earnings regression.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real height_std;
  array[N] real log_earn_std;
}
parameters {
  real beta_0;
  real beta_1;
  real<lower=0> sigma;
}
model {
  beta_0 ~ normal(0, 0.5);
  beta_1 ~ normal(0, 0.5);
  sigma ~ normal(0, 0.5);
  for (n in 1:N) {
    log_earn_std[n] ~ normal(beta_0 + beta_1 * height_std[n], sigma);
  }
}
""",
        stan_data={"N": n, "height_std": height_std, "log_earn_std": log_earn_std},
        tags=("informed",),
    )


def _kidscore_momiq_informed_recipe() -> ModelRecipe:
    n = 120
    mom_iq_std = [(-2.0 + 4.0 * float(idx) / float(n - 1)) for idx in range(n)]
    kid_score_std = [
        0.1 + 0.45 * value + 0.03 * float(idx % 6) for idx, value in enumerate(mom_iq_std)
    ]
    return ModelRecipe(
        name="kidscore_momiq_informed",
        description="Informed kidscore regression.",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real mom_iq_std;
  array[N] real kid_score_std;
}
parameters {
  real beta_0;
  real beta_1;
  real<lower=0> sigma;
}
model {
  beta_0 ~ normal(0, 0.5);
  beta_1 ~ normal(0, 0.5);
  sigma ~ normal(0, 0.5);
  for (n in 1:N) {
    kid_score_std[n] ~ normal(beta_0 + beta_1 * mom_iq_std[n], sigma);
  }
}
""",
        stan_data={"N": n, "mom_iq_std": mom_iq_std, "kid_score_std": kid_score_std},
        tags=("informed",),
    )


def _blr_informed_recipe() -> ModelRecipe:
    n = 120
    d = 5
    x_std = [[-1.0 + 0.02 * float((row + col * 7) % 100) for col in range(d)] for row in range(n)]
    y_std = [sum(0.15 * float(col + 1) * x_std[row][col] for col in range(d)) for row in range(n)]
    return ModelRecipe(
        name="blr_informed",
        description="Informed Bayesian linear regression.",
        stan_code="""
data {
  int<lower=1> N;
  int<lower=1> D;
  matrix[N, D] X_std;
  vector[N] y_std;
}
parameters {
  vector[D] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 0.5);
  sigma ~ normal(0, 0.5);
  y_std ~ normal(X_std * beta, sigma);
}
""",
        stan_data={"N": n, "D": d, "X_std": x_std, "y_std": y_std},
        tags=("informed",),
    )


def _mesquite_logvolume_informed_recipe() -> ModelRecipe:
    n = 140
    k = 2
    log_canopy_volume_std = [(-2.0 + 4.0 * float(idx) / float(n - 1)) for idx in range(n)]
    log_weight_std = [
        0.1 + 0.6 * value + 0.04 * float(idx % 5) for idx, value in enumerate(log_canopy_volume_std)
    ]
    return ModelRecipe(
        name="mesquite_logvolume_informed",
        description="Informed mesquite logvolume regression.",
        stan_code="""
data {
  int<lower=1> N;
  int<lower=1> K;
  array[N] real log_canopy_volume_std;
  array[N] real log_weight_std;
}
parameters {
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 0.5);
  sigma ~ normal(0, 0.5);
  for (n in 1:N) {
    log_weight_std[n] ~ normal(beta[1] + beta[2] * log_canopy_volume_std[n], sigma);
  }
}
""",
        stan_data={
            "N": n,
            "K": k,
            "log_canopy_volume_std": log_canopy_volume_std,
            "log_weight_std": log_weight_std,
        },
        tags=("informed",),
    )


def _pair_eight_schools() -> PairRecipe:
    data = {
        "N": 8,
        "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    }
    centered = PairVariantRecipe(
        name="centered",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real y;
  array[N] real sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[N] theta;
}
model {
  mu ~ normal(0, 5);
  tau ~ normal(0, 5);
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
}
""",
        stan_data=data,
        model_spec={
            "parameters": [{"name": "mu"}, {"name": "tau"}, {"name": "theta", "shape": [8]}]
        },
    )
    noncentered = PairVariantRecipe(
        name="noncentered",
        stan_code="""
data {
  int<lower=1> N;
  array[N] real y;
  array[N] real sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[N] theta_raw;
}
transformed parameters {
  vector[N] theta = mu + tau * theta_raw;
}
model {
  mu ~ normal(0, 5);
  tau ~ normal(0, 5);
  theta_raw ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
""",
        stan_data=data,
        model_spec={
            "parameters": [
                {"name": "mu"},
                {"name": "tau"},
                {"name": "theta_raw", "shape": [8]},
            ]
        },
    )
    return _pair_recipe(
        name="eight_schools",
        description="Eight Schools hierarchical model.",
        reference_model="eight_schools-noncentered",
        centered=centered,
        noncentered=noncentered,
        difficulty="easy-medium",
    )


def _pair_neals_funnel() -> PairRecipe:
    centered = PairVariantRecipe(
        name="centered",
        stan_code="""
data {
  int<lower=1> N;
}
parameters {
  real v;
  vector[N] x;
}
model {
  v ~ normal(0, 3);
  x ~ normal(0, exp(v / 2));
}
""",
        stan_data={"N": 9},
        model_spec={
            "parameters": [{"name": "v"}, {"name": "x", "shape": [9]}],
            "data": [{"name": "N", "type": "int"}],
        },
    )
    noncentered = PairVariantRecipe(
        name="noncentered",
        stan_code="""
data {
  int<lower=1> N;
}
parameters {
  real v;
  vector[N] x_raw;
}
transformed parameters {
  vector[N] x = x_raw * exp(v / 2);
}
model {
  v ~ normal(0, 3);
  x_raw ~ normal(0, 1);
}
""",
        stan_data={"N": 9},
        model_spec={
            "parameters": [{"name": "v"}, {"name": "x_raw", "shape": [9]}],
            "transformed_parameters": [{"name": "x", "shape": [9]}],
            "data": [{"name": "N", "type": "int"}],
        },
    )
    return _pair_recipe(
        name="neals_funnel",
        description="Neal's funnel geometry pair.",
        reference_model="neals_funnel-noncentered",
        centered=centered,
        noncentered=noncentered,
        difficulty="easy",
    )


def _pair_hierarchical_lr() -> PairRecipe:
    n_groups = 8
    n = 64
    data = {
        "N": n,
        "J": n_groups,
        "group": [(idx % n_groups) + 1 for idx in range(n)],
        "x": [float((idx % 9) - 4) / 3.0 for idx in range(n)],
        "y": [
            0.3 + 0.8 * float((idx % 9) - 4) / 3.0 + 0.05 * float(idx % n_groups)
            for idx in range(n)
        ],
    }
    centered = PairVariantRecipe(
        name="centered",
        stan_code="""
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> group;
  array[N] real x;
  array[N] real y;
}
parameters {
  real mu_alpha;
  real<lower=0> sigma_alpha;
  vector[J] alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  mu_alpha ~ normal(0, 1);
  sigma_alpha ~ normal(0, 1);
  alpha ~ normal(mu_alpha, sigma_alpha);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    y[n] ~ normal(alpha[group[n]] + beta * x[n], sigma);
  }
}
""",
        stan_data=data,
        model_spec={"parameters": [{"name": "alpha", "shape": [8]}]},
    )
    noncentered = PairVariantRecipe(
        name="noncentered",
        stan_code="""
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> group;
  array[N] real x;
  array[N] real y;
}
parameters {
  real mu_alpha;
  real<lower=0> sigma_alpha;
  vector[J] alpha_raw;
  real beta;
  real<lower=0> sigma;
}
transformed parameters {
  vector[J] alpha = mu_alpha + sigma_alpha * alpha_raw;
}
model {
  mu_alpha ~ normal(0, 1);
  sigma_alpha ~ normal(0, 1);
  alpha_raw ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    y[n] ~ normal(alpha[group[n]] + beta * x[n], sigma);
  }
}
""",
        stan_data=data,
        model_spec={"parameters": [{"name": "alpha_raw", "shape": [8]}]},
    )
    return _pair_recipe(
        name="hierarchical_lr",
        description="Hierarchical linear regression pair.",
        reference_model="hierarchical_lr-noncentered",
        centered=centered,
        noncentered=noncentered,
        difficulty="medium",
    )


def _pair_varying_slopes() -> PairRecipe:
    n_groups = 10
    n = 80
    data = {
        "N": n,
        "J": n_groups,
        "group": [(idx % n_groups) + 1 for idx in range(n)],
        "x": [float((idx % 13) - 6) / 4.0 for idx in range(n)],
        "y": [
            0.5 + 0.2 * float(idx % n_groups) + 0.7 * float((idx % 13) - 6) / 4.0
            for idx in range(n)
        ],
    }
    centered = PairVariantRecipe(
        name="centered",
        stan_code="""
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> group;
  array[N] real x;
  array[N] real y;
}
parameters {
  vector[2] mu;
  vector<lower=0>[2] sigma_group;
  matrix[J, 2] beta_group;
  real<lower=0> sigma;
}
model {
  to_vector(beta_group) ~ normal(0, 1);
  sigma_group ~ normal(0, 1);
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    y[n] ~ normal(beta_group[group[n], 1] + beta_group[group[n], 2] * x[n], sigma);
  }
}
""",
        stan_data=data,
        model_spec={"parameters": [{"name": "beta_group", "shape": [10, 2]}]},
    )
    noncentered = PairVariantRecipe(
        name="noncentered",
        stan_code="""
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> group;
  array[N] real x;
  array[N] real y;
}
parameters {
  vector[2] mu;
  vector<lower=0>[2] sigma_group;
  matrix[J, 2] z_group;
  real<lower=0> sigma;
}
transformed parameters {
  matrix[J, 2] beta_group;
  for (j in 1:J) {
    beta_group[j, 1] = mu[1] + sigma_group[1] * z_group[j, 1];
    beta_group[j, 2] = mu[2] + sigma_group[2] * z_group[j, 2];
  }
}
model {
  to_vector(z_group) ~ normal(0, 1);
  sigma_group ~ normal(0, 1);
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    y[n] ~ normal(beta_group[group[n], 1] + beta_group[group[n], 2] * x[n], sigma);
  }
}
""",
        stan_data=data,
        model_spec={"parameters": [{"name": "z_group", "shape": [10, 2]}]},
    )
    return _pair_recipe(
        name="varying_slopes",
        description="Varying slopes pair with correlated random effects.",
        reference_model="varying_slopes-noncentered",
        centered=centered,
        noncentered=noncentered,
        difficulty="hard",
    )


def _pair_bangladesh_contraceptive() -> PairRecipe:
    n_districts = 12
    n = 120
    urban = [idx % 2 for idx in range(n)]
    district = [(idx % n_districts) + 1 for idx in range(n)]
    use = [
        1 if (0.4 + 0.1 * urban[idx] + 0.02 * (district[idx] - 1)) > 0.55 else 0 for idx in range(n)
    ]
    data = {
        "N": n,
        "D": n_districts,
        "district": district,
        "urban": urban,
        "use": use,
    }
    centered = PairVariantRecipe(
        name="centered",
        stan_code="""
data {
  int<lower=1> N;
  int<lower=1> D;
  array[N] int<lower=1, upper=D> district;
  array[N] int<lower=0, upper=1> urban;
  array[N] int<lower=0, upper=1> use;
}
parameters {
  real mu_a;
  real mu_b;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  vector[D] a;
  vector[D] b;
}
model {
  a ~ normal(mu_a, sigma_a);
  b ~ normal(mu_b, sigma_b);
  mu_a ~ normal(0, 1);
  mu_b ~ normal(0, 1);
  sigma_a ~ normal(0, 1);
  sigma_b ~ normal(0, 1);
  for (n in 1:N) {
    use[n] ~ bernoulli_logit(a[district[n]] + b[district[n]] * urban[n]);
  }
}
""",
        stan_data=data,
        model_spec={"parameters": [{"name": "a", "shape": [12]}, {"name": "b", "shape": [12]}]},
    )
    noncentered = PairVariantRecipe(
        name="noncentered",
        stan_code="""
data {
  int<lower=1> N;
  int<lower=1> D;
  array[N] int<lower=1, upper=D> district;
  array[N] int<lower=0, upper=1> urban;
  array[N] int<lower=0, upper=1> use;
}
parameters {
  real mu_a;
  real mu_b;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  vector[D] a_raw;
  vector[D] b_raw;
}
transformed parameters {
  vector[D] a = mu_a + sigma_a * a_raw;
  vector[D] b = mu_b + sigma_b * b_raw;
}
model {
  a_raw ~ normal(0, 1);
  b_raw ~ normal(0, 1);
  mu_a ~ normal(0, 1);
  mu_b ~ normal(0, 1);
  sigma_a ~ normal(0, 1);
  sigma_b ~ normal(0, 1);
  for (n in 1:N) {
    use[n] ~ bernoulli_logit(a[district[n]] + b[district[n]] * urban[n]);
  }
}
""",
        stan_data=data,
        model_spec={
            "parameters": [{"name": "a_raw", "shape": [12]}, {"name": "b_raw", "shape": [12]}]
        },
    )
    return _pair_recipe(
        name="bangladesh_contraceptive",
        description="Hierarchical contraceptive logistic regression pair.",
        reference_model="bangladesh_contraceptive-noncentered",
        centered=centered,
        noncentered=noncentered,
        difficulty="hard",
    )


def _pair_recipe(
    *,
    name: str,
    description: str,
    reference_model: str,
    centered: PairVariantRecipe,
    noncentered: PairVariantRecipe,
    difficulty: str,
) -> PairRecipe:
    return PairRecipe(
        name=name,
        description=description,
        bad_variant="centered",
        good_variant="noncentered",
        reference_model=reference_model,
        expected_pathologies=("divergences", "high_rhat", "low_ess"),
        difficulty=difficulty,
        variants={"centered": centered, "noncentered": noncentered},
    )
