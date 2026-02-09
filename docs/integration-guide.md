# mcmc-ref Integration Guide (uv)

This guide is for developers implementing or validating a Bayesian
probabilistic framework against `mcmc-ref` reference draws.

## Install

Use `uv` only.

```bash
# Core library only
uv add mcmc-ref

# Core library + packaged reference corpus
uv add "mcmc-ref[data]"
```

In this repository, `mcmc-ref[data]` resolves `mcmc-ref-data` from
`packages/mcmc-ref-data` via `tool.uv.sources`.

## What You Validate

Your framework sampler output is compared against reference posterior summaries.
Default diagnostics require at least 4 independent chains.

## Framework Integration Flow

1. Pick one or more reference models.
2. Run your sampler with deterministic settings:
   - fixed seed
   - at least 4 chains
   - enough warmup/samples for stable mean/std
3. Convert your samples to `dict[param_name, list[float]]`.
4. Compare with `mcmc_ref.reference.compare`.

### Python Example

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

### CLI Example

```bash
mcmc-ref compare eight_schools-eight_schools_noncentered \
  --actual my_draws.csv \
  --tolerance 0.15
```

CSV should include columns:
- `chain`
- `draw`
- parameter columns (for example `mu`, `tau`, `theta[1]`)

## Parameter Naming Rules

- Names must match reference names exactly.
- Vector parameters use Stan-style indexing, e.g. `theta[1]`, `theta[2]`.
- Flatten your framework output to scalar parameter series before comparing.

## CI Pattern

Use two layers:

1. PR/fast CI:
   - small model subset
   - compare on mean/std with moderate tolerance
2. Nightly/full CI:
   - larger model matrix
   - optional tighter tolerance/model-specific checks

## Using Custom Corpora

If you maintain a private or custom corpus:

```bash
export MCMC_REF_LOCAL_ROOT=/path/to/reference-corpus
```

Corpus layout:
- `draws/*.draws.parquet`
- `meta/*.meta.json`

## Build Canonical References (Maintainers)

Generate local corpus from posteriordb draw archives:

```bash
uv run --extra dev python scripts/build_references.py
```

Useful options:

```bash
uv run --extra dev python scripts/build_references.py \
  --models eight_schools-eight_schools_noncentered,diamonds-diamonds

uv run --extra dev python scripts/build_references.py \
  --source-dir ~/.posteriordb/posterior_database/reference_posteriors/draws/draws \
  --output-root /path/to/reference-corpus
```

## Sync and Publish Data Package (Maintainers)

Sync generated local corpus into package data:

```bash
uv run --extra dev python scripts/sync_data_package.py
```

Reinstall data package in local env after sync:

```bash
uv sync --extra data --reinstall-package mcmc-ref-data
```

Publish data package:

```bash
cd packages/mcmc-ref-data
uv build
uv publish
```

Then publish `mcmc-ref` with a matching version so `mcmc-ref[data]` stays aligned.

## Quality Notes

- Diagnostics use rank-normalized split R-hat and ESS.
- Defaults enforce 4-chain diagnostics.
- `convert --force` allows bypassing strict checks for non-canonical workflows.
