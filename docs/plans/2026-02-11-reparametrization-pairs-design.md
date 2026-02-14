# Reparametrization Test Pairs — Design

## Goal

Add a reparametrization test suite to mcmc-ref: paired models (centered + non-centered) where one parametrization is known-bad and the other samples cleanly. Intended as integration tests for validating probabilistic sampling libraries.

## Design Decisions

- **No reference draws for bad variants.** Only store model definitions (Stan code + spec) for both. Reference draws only for the good variant. The bad variant is a known-failure fixture.
- **Naming convention + pair metadata.** Models named `{pair}/centered` and `{pair}/noncentered`, with `pair.json` at the pair level.
- **Stan code + model_spec.json.** Stan code as canonical reference, JSON spec as machine-readable structure for sampler consumption.
- **Python API.** `mcmc_ref.list_pairs()` and `mcmc_ref.pair(name)` returning a `Pair` dataclass.

## Storage Layout

```
src/mcmc_ref/data/pairs/
  neals_funnel/
    pair.json
    centered/
      model.stan
      model_spec.json
      data.json
    noncentered/
      model.stan
      model_spec.json
      data.json
  eight_schools/
    ...
  hierarchical_lr/
    ...
  varying_slopes/
    ...
```

Reference draws for good variants live in the existing `draws/` + `meta/` store (no duplication).

## pair.json Schema

```json
{
  "name": "neals_funnel",
  "description": "Neal's funnel: pure funnel geometry",
  "bad_variant": "centered",
  "good_variant": "noncentered",
  "reference_model": "neals_funnel-noncentered",
  "expected_pathologies": ["divergences", "high_rhat", "low_ess"],
  "difficulty": "easy"
}
```

`reference_model` points to the model name in the existing `draws/` + `meta/` store.

## model_spec.json Schema

```json
{
  "parameters": [
    {"name": "v", "shape": [], "constraint": "real"},
    {"name": "x", "shape": [9], "constraint": "real"}
  ],
  "data": [
    {"name": "N", "type": "int"}
  ],
  "log_density_components": [
    {"type": "normal", "variate": "v", "mean": 0, "sd": 3},
    {"type": "normal", "variate": "x", "mean": 0, "sd": "exp(v/2)"}
  ]
}
```

## Python API

```python
import mcmc_ref

# List all reparametrization pairs
mcmc_ref.list_pairs()
# -> ["neals_funnel", "eight_schools", "hierarchical_lr", "varying_slopes"]

# Get a specific pair
pair = mcmc_ref.pair("neals_funnel")
#   pair.name            -> "neals_funnel"
#   pair.bad_variant     -> "centered"
#   pair.good_variant    -> "noncentered"
#   pair.bad_spec        -> dict (parsed model_spec.json for centered)
#   pair.good_spec       -> dict (parsed model_spec.json for noncentered)
#   pair.data            -> dict (parsed data.json, shared across variants)
#   pair.reference_draws -> Draws object (from existing store)
#   pair.reference_stats -> dict (from existing store)
#   pair.expected_pathologies -> ["divergences", "high_rhat", "low_ess"]
```

`Pair` is a dataclass. `reference_draws` and `reference_stats` delegate to the existing `mcmc_ref.draws()` and `mcmc_ref.stats()` using `reference_model` from `pair.json`.

### Typical test usage

```python
pair = mcmc_ref.pair("neals_funnel")

# Build model from spec, run sampler on good parametrization
good_model = my_sampler.from_spec(pair.good_spec, pair.data)
my_draws = good_model.sample(chains=4, draws=1000)
result = mcmc_ref.compare("neals_funnel-noncentered", my_draws)
assert result.passed

# Confirm bad parametrization shows expected problems
bad_model = my_sampler.from_spec(pair.bad_spec, pair.data)
bad_draws = bad_model.sample(chains=4, draws=1000)
assert bad_draws.divergences > 0
```

## The Four Model Pairs

### 1. Neal's Funnel (difficulty: easy)

The simplest possible funnel. No data — pure prior sampling.

**Centered (bad):**
```
v ~ Normal(0, 3)
x[1..9] ~ Normal(0, exp(v/2))
```
When v is large negative (e.g. -6), exp(v/2) ~ 0.05 — the x dimensions squeeze into a tiny region. HMC can't find a single step size that works across this geometry. Expect: divergences, low ESS on v.

**Non-centered (good):**
```
v ~ Normal(0, 3)
x_raw[1..9] ~ Normal(0, 1)
x = x_raw * exp(v/2)       // deterministic transform
```
The sampler explores (v, x_raw) which is standard normal in all dimensions. Clean sampling.

**Data:** None. N = 9 (dimensionality only).
**Parameters:** 10 — v (scalar) + x (9-vector).

If your sampler can't handle this, nothing else will work. First test to pass.

### 2. Eight Schools (difficulty: easy-medium)

The canonical hierarchical model from Rubin (1981).

**Centered (bad):**
```
mu ~ Normal(0, 5)
tau ~ HalfCauchy(0, 5)
theta[1..8] ~ Normal(mu, tau)
y[j] ~ Normal(theta[j], sigma[j])    // sigma known
```
When tau approaches zero, theta values are forced near mu but the likelihood pulls them toward school-specific estimates. Funnel in (tau, theta) space. Expect: divergences near tau ~ 0, poor R-hat on tau.

**Non-centered (good):**
```
mu ~ Normal(0, 5)
tau ~ HalfCauchy(0, 5)
theta_raw[1..8] ~ Normal(0, 1)
theta = mu + tau * theta_raw          // deterministic
y[j] ~ Normal(theta[j], sigma[j])
```

**Data:**
```json
{"J": 8, "y": [28, 8, -3, 7, -1, 1, 18, 12], "sigma": [15, 10, 16, 11, 9, 11, 10, 18]}
```

**Parameters:** 10 — mu + tau + theta[8].

Existing non-centered reference draws can be reused. Only need to add centered variant definition and pair metadata.

### 3. Hierarchical Linear Regression with Sparse Groups (difficulty: medium)

Varying-intercepts regression with deliberately imbalanced group sizes.

**Centered (bad):**
```
mu_alpha ~ Normal(0, 10)
sigma_alpha ~ HalfNormal(0, 5)
alpha[1..K] ~ Normal(mu_alpha, sigma_alpha)
beta ~ Normal(0, 5)
sigma_y ~ HalfNormal(0, 5)
y[i] ~ Normal(alpha[group[i]] + beta * x[i], sigma_y)
```
Groups with 1-2 observations can't constrain their alpha. The posterior collapses into a funnel with sigma_alpha. Expect: divergences, poor R-hat on sigma_alpha and sparse-group alphas.

**Non-centered (good):**
```
mu_alpha ~ Normal(0, 10)
sigma_alpha ~ HalfNormal(0, 5)
alpha_raw[1..K] ~ Normal(0, 1)
alpha = mu_alpha + sigma_alpha * alpha_raw    // deterministic
beta ~ Normal(0, 5)
sigma_y ~ HalfNormal(0, 5)
y[i] ~ Normal(alpha[group[i]] + beta * x[i], sigma_y)
```

**Data:** Synthetic, ~6 groups, deliberately imbalanced sizes (e.g. [50, 40, 30, 3, 2, 1]). Continuous predictor x, continuous outcome y. Generated and frozen.

**Parameters:** K + 4 — mu_alpha, sigma_alpha, alpha[1..K], beta, sigma_y.

### 4. Varying Slopes with Correlated Random Effects (difficulty: hard)

The "full luxury" multilevel model. Groups have both varying intercepts and varying slopes with a covariance matrix.

**Centered (bad):**
```
mu[1..2] ~ Normal(0, 5)
sigma[1..2] ~ HalfNormal(0, 5)
Rho ~ LKJcorr(2, eta=2)
Sigma = diag(sigma) * Rho * diag(sigma)
z[1..K] ~ MVN(mu, Sigma)              // z[k] = (alpha_k, beta_k)
sigma_y ~ HalfNormal(0, 5)
y[i] ~ Normal(z[group[i], 1] + z[group[i], 2] * x[i], sigma_y)
```
When sigma[1] or sigma[2] is small, corresponding random effects collapse and Rho becomes poorly identified. Higher-dimensional funnel with correlation structure. Expect: divergences, poor mixing on sigma and Rho.

**Non-centered (good):**
```
mu[1..2] ~ Normal(0, 5)
sigma[1..2] ~ HalfNormal(0, 5)
Rho ~ LKJcorr(2, eta=2)
L = cholesky(diag(sigma) * Rho * diag(sigma))
z_raw[1..K] ~ Normal(0, 1)            // K x 2 matrix of raw offsets
z = mu + (L * z_raw')'                // deterministic transform
sigma_y ~ HalfNormal(0, 5)
y[i] ~ Normal(z[group[i], 1] + z[group[i], 2] * x[i], sigma_y)
```

**Data:** Synthetic, ~8 groups with imbalanced sizes. True correlation between intercept and slope ~ -0.5 (groups with high intercepts tend to have flatter slopes).

**Parameters:** 2K + 6 — mu[2], sigma[2], Rho (1 free param for 2x2), z[K,2], sigma_y.

The final boss. If your sampler handles this cleanly, it handles hierarchical models well.

## Implementation Steps

1. Create directory structure under `src/mcmc_ref/data/pairs/`
2. Write Stan code for all 8 variants (4 pairs x 2 parametrizations)
3. Write `model_spec.json` for all 8 variants
4. Generate and freeze synthetic data for hierarchical_lr and varying_slopes
5. Generate reference draws for 3 new good variants via CmdStan (eight_schools non-centered already exists)
6. Write `pair.json` for all 4 pairs
7. Implement `Pair` dataclass and `list_pairs()` / `pair()` API
8. Add CLI support (e.g. `mcmc-ref pairs`, `mcmc-ref pair neals_funnel`)
9. Tests for pair discovery, loading, and API
10. Bundle everything into the package

## Release & Integration Notification

After merging the reparametrization pairs feature:

1. **Bump version** in `pyproject.toml` (minor version bump — new feature, backwards compatible).

2. **Post GitHub issue to downstream consumers** notifying them of the new version and encouraging integration. Use `gh` CLI:
   ```bash
   gh issue create --repo <org>/jaxstanv3 \
     --title "mcmc-ref v<X.Y.0>: new reparametrization test pairs available" \
     --body "mcmc-ref v<X.Y.0> adds reparametrization test pairs (Neal's funnel, Eight Schools, hierarchical LR, varying slopes). Each pair provides centered (known-bad) and non-centered (known-good) parametrizations with reference draws for the good variant. Useful for validating sampler correctness on funnel geometries. See: <link-to-changelog-or-readme>"
   ```

3. **Repeat for other downstream repos** that depend on mcmc-ref as a test dependency.

4. **Update README** with a section on reparametrization pairs and example test code.
