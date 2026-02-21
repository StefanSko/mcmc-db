# mcmc-ref Integration Guide (uv)

This guide defines the canonical integration and data-provenance workflow for `mcmc-ref`.

## Install

Use `uv`.

```bash
# Core API
uv add mcmc-ref

# Core API + packaged reference corpus
uv add "mcmc-ref[data]"
```

## Consumer Integration (canonical)

Downstream libraries (for example `jaxstanv3`) should consume only `mcmc-ref` + `mcmc-ref-data`.
Do not check in local copies of reference draws/meta/pairs.

### API flow

1. Choose a reference model id.
2. Run your sampler with deterministic settings (fixed seed, >=4 chains).
3. Convert framework samples to `dict[param_name, list[float]]`.
4. Compare with `mcmc_ref.reference.compare`.

```python
from mcmc_ref import reference

actual = {
    "mu": mu_draws.reshape(-1).tolist(),
    "tau": tau_draws.reshape(-1).tolist(),
}

result = reference.compare(
    "eight_schools-eight_schools_noncentered",
    actual=actual,
    tolerance=0.15,
    metrics=("mean", "std"),
)
assert result.passed, result.failures
```

### Data root resolution order

`DataStore` resolves in this order:

1. `MCMC_REF_LOCAL_ROOT` (optional local override)
2. installed package data (`mcmc-ref-data`, fallback `mcmc-ref`)

Use `MCMC_REF_LOCAL_ROOT` only for local experimentation.

## Maintainer Provenance Pipeline (canonical)

All curated references should flow through:

1. `mcmc-ref provenance-scaffold`
2. `mcmc-ref provenance-generate`
3. `mcmc-ref provenance-publish`

### 1) Scaffold deterministic recipes

```bash
mcmc-ref provenance-scaffold --output-root /tmp/mcmc-db/provenance
```

Writes:
- `stan_models/*.stan`
- `stan_data/*.json`
- `pairs/**`
- `provenance_manifest.json`

### 2) Generate draws/meta via CmdStan

```bash
# one-time dependency + CmdStan install
uv add --dev "mcmc-ref[provenance]"
uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

mcmc-ref provenance-generate \
  --scaffold-root /tmp/mcmc-db/provenance \
  --output-root /tmp/mcmc-db/generated
```

Writes:
- `archives/*.json.zip`
- `draws/*.draws.parquet`
- `meta/*.meta.json`

### 3) Publish into package data layout

```bash
mcmc-ref provenance-publish \
  --source-root /tmp/mcmc-db/generated \
  --scaffold-root /tmp/mcmc-db/provenance \
  --package-root packages/mcmc-ref-data/src/mcmc_ref_data/data
```

Copies:
- generated `draws/*`
- generated `meta/*`
- scaffolded `pairs/*`
- `provenance_manifest.json`

## CI/Release Gates

Run before release:

```bash
uv run ruff check .
uv run ty check .
uv run pytest
```

Recommended smoke gate:
- scaffold -> generate (small subset) -> publish -> consumer smoke test.
