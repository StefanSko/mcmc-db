# mcmc-ref

Reference posterior validation tool for Bayesian inference libraries.

## Development

- Use `uv` for envs and tool execution.
- Use `ruff` for lint/format.
- Use `ty` for type checking (strict).
- Use red -> green TDD for changes.

### Commands

```bash
uv run ruff check .
uv run ruff format .
uv run ty check .
uv run pytest
```

## Bundled Models

mcmc-ref ships complete assets (draws, metadata, Stan data, Stan code) for these models:

| Model | Type | Params |
|-------|------|--------|
| `wells_data-wells_dist` | Logistic regression | 2 |
| `GLM_Binomial_data-GLM_Binomial_model` | Binomial GLM | 3 + generated quantities |
| `GLM_Poisson_Data-GLM_Poisson_model` | Poisson GLM | 4 + generated quantities |
| `radon_mn-radon_hierarchical_intercept_noncentered` | Hierarchical | ~90 |
| `irt_2pl-irt_2pl` | IRT 2PL | ~125 |

All reference draws: 10 chains x 1,000 draws, R-hat < 1.01, ESS > 400.

## CLI

```bash
# List all available models
mcmc-ref list

# Get Stan input data as JSON
mcmc-ref data wells_data-wells_dist

# Get Stan model code
mcmc-ref model-code wells_data-wells_dist

# Summary statistics
mcmc-ref stats wells_data-wells_dist --include-diagnostics --format json

# Diagnostics (R-hat, ESS)
mcmc-ref diagnostics wells_data-wells_dist

# Compare sampler output against reference
mcmc-ref compare wells_data-wells_dist --actual my_results.csv --tolerance 0.15
```

## Python API

```python
from mcmc_ref import reference

# Stan input data (dict)
data = reference.stan_data("wells_data-wells_dist")

# Stan model code (str)
code = reference.model_code("wells_data-wells_dist")

# Reference draws (Arrow table)
table = reference.draws("wells_data-wells_dist", return_="arrow")

# Summary statistics
stats = reference.stats("wells_data-wells_dist")

# Diagnostics
diag = reference.diagnostics_for_model("wells_data-wells_dist")

# Compare actual draws against reference
result = reference.compare("wells_data-wells_dist", actual={"beta[1]": [...], "beta[2]": [...]})
assert result.passed, result.failures
```

### Consumer Integration Guide

```python
from mcmc_ref import reference

# 1. Load Stan data and model code
data = reference.stan_data("wells_data-wells_dist")
code = reference.model_code("wells_data-wells_dist")

# 2. Compile and sample with your library
model = your_library.compile(code)
fit = model.sample(data, num_chains=4, num_samples=1000)

# 3. Compare against reference draws
result = reference.compare("wells_data-wells_dist", fit.as_dict(), tolerance=0.15)
assert result.passed, "\n".join(result.failures)
```

### Diagnostics

```python
from mcmc_ref import diagnostics

chains = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]
rhat = diagnostics.split_rhat(chains)
ess = diagnostics.ess_bulk(chains)
```

## Reference Draws

When generating reference draws, follow Stan's prior-choice recommendations:
avoid flat/super-vague priors, prefer weakly informative priors on a sensible
scale, and re-draw posteriordb models that use extremely flat priors so the
reference is stable and diagnostic-friendly.

Diagnostics use rank-normalized split R-hat and bulk/tail ESS with stdlib-only
implementations to keep dependencies light.
These diagnostics require at least 4 independent chains by default; use
explicit override paths (for example `convert --force`) only when you
intentionally want single-chain conversion with diagnostic metrics marked `nan`.

### Generate Reference Draws

To generate draws for a model that has bundled Stan data and code:

```bash
uv run --extra build mcmc-ref-generate wells_data-wells_dist
```

Settings: 10 chains, 10,000 warmup, 10,000 sampling, thin=10, seed=4711.

## Build Canonical References

Use this when you want a pre-built corpus other libraries can validate against.

```bash
# default source: ~/.posteriordb/posterior_database/reference_posteriors/draws/draws
# default output: ~/.mcmc-ref/{draws,meta}
uv run --extra dev python scripts/build_references.py
```

The builder converts every `*.json.zip` file and enforces quality checks
(`nchains_is_gte_4`, `rhat_below_1_01`, ESS and draw-count checks). If any model
fails, it exits non-zero and prints the failing models.

Useful options:

```bash
# convert only selected models
uv run --extra dev python scripts/build_references.py \
  --models eight_schools-eight_schools_noncentered,diamonds-diamonds

# override source and output paths
uv run --extra dev python scripts/build_references.py \
  --source-dir ~/.posteriordb/posterior_database/reference_posteriors/draws/draws \
  --output-root /path/to/reference-corpus
```

If you install `mcmc-ref` as a package, the equivalent command is
`mcmc-ref-build-references`.

## Validate From Another Library

Point your tests at the reference corpus and compare your sampler output.

1. Build or download a corpus containing `draws/*.draws.parquet` and `meta/*.meta.json`.
2. Set `MCMC_REF_LOCAL_ROOT` to that corpus root in your test environment.
3. Run comparisons against one or more models.

CLI flow:

```bash
export MCMC_REF_LOCAL_ROOT=/path/to/reference-corpus
mcmc-ref compare eight_schools-eight_schools_noncentered --actual my_draws.csv --tolerance 0.15
```

Python flow:

```python
import os
from mcmc_ref import reference

os.environ["MCMC_REF_LOCAL_ROOT"] = "/path/to/reference-corpus"
result = reference.compare(
    "eight_schools-eight_schools_noncentered",
    actual={"mu": mu_draws, "tau": tau_draws},
    tolerance=0.15,
)
assert result.passed, result.failures
```

`actual` can come from your sampler output after mapping parameter names to the
reference naming convention.

## Origin Story:

Brainstorm:
    -https://gisthost.github.io/?a47c363027aba4615cd3247ed5f88793

Implemntation:
    -https://gistpreview.github.io/?018116c7a98a7d75e363e77689ec7b26
