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

### CLI example

```bash
mcmc-ref stats eight_schools --include-diagnostics --quantile-mode exact
```

### Compare example

```bash
mcmc-ref compare eight_schools --actual my_results.csv --tolerance 0.15
```

### Python API example

```python
from mcmc_ref import reference

# Arrow by default
table = reference.draws("eight_schools", return_="arrow")

# Wrapper with conversion helpers
draws = reference.draws("eight_schools", return_="draws")
numpy_draws = draws.to_numpy()  # requires numpy
```

### Diagnostics example

```python
from mcmc_ref import diagnostics

chains = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]
rhat = diagnostics.split_rhat(chains)
ess = diagnostics.ess_bulk(chains)
```

## Reference draws

When generating reference draws, follow Stan's prior-choice recommendations:
avoid flat/super-vague priors, prefer weakly informative priors on a sensible
scale, and re-draw models that use extremely flat priors so the
reference is stable and diagnostic-friendly.

Diagnostics use rank-normalized split R-hat and bulk/tail ESS with stdlib-only
implementations to keep dependencies light.
These diagnostics require at least 4 independent chains by default; use
explicit override paths (for example `convert --force`) only when you
intentionally want single-chain conversion with diagnostic metrics marked `nan`.

For integration tests, you can optionally generate missing reference draws by
setting `MCMC_REF_GENERATE=1` and (optionally) `MCMC_REF_GENERATE_MODEL=...`.

## Build Canonical References

Use this when you want a pre-built corpus other libraries can validate against.

```bash
# default source: ~/.mcmc-db/reference_archives
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
  --source-dir ~/.mcmc-db/reference_archives \
  --output-root /path/to/reference-corpus
```

If you install `mcmc-ref` as a package, the equivalent command is
`mcmc-ref-build-references`.

## Recreate Provenance Assets

Generate deterministic Stan source/data scaffolding and geometry-pair definitions
directly from versioned recipes in this repo (no copied reference artifacts):

```bash
mcmc-ref provenance-scaffold --output-root /tmp/mcmc-db/provenance
# or
uv run --extra dev python scripts/materialize_provenance.py --output-root /tmp/mcmc-db/provenance
```

This writes:
- `stan_models/*.stan` and `stan_data/*.json` for all curated models (core, informed, geometry refs)
- `pairs/<name>/{pair.json,centered/*,noncentered/*}` for geometry benchmarks
- `provenance_manifest.json` with SHA256 hashes + CmdStan settings

Generate fresh reference archives/draws from those recipes:

```bash
# requires cmdstanpy + CmdStan
uv add --dev --optional provenance cmdstanpy
mcmc-ref provenance-generate \
  --scaffold-root /tmp/mcmc-db/provenance \
  --output-root /tmp/mcmc-db/generated
```

Publish generated references into the package data tree used by downstream consumers:

```bash
mcmc-ref provenance-publish \
  --source-root /tmp/mcmc-db/generated \
  --scaffold-root /tmp/mcmc-db/provenance \
  --package-root src/mcmc_ref/data
```

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
