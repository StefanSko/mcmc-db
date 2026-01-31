# mcmc-ref

Reference posterior validation tool for Bayesian inference libraries. It ships
pre-computed reference draws in a compact columnar format and provides a CLI and
Python API for comparing your sampler output against those references.

## Why

- Validate samplers against known posteriors with stable reference draws.
- Integrate in CI with deterministic, pre-computed baselines.
- Debug model issues with consistent diagnostics and summary stats.

## Quickstart

### Install

```bash
uv pip install -e .
```

### CLI (stats + compare)

```bash
# Summary stats
mcmc-ref stats eight_schools --include-diagnostics --quantile-mode exact

# Compare a CSV of your draws to the reference
mcmc-ref compare eight_schools --actual my_results.csv --tolerance 0.15
```

### Python API

```python
from mcmc_ref import reference

# List bundled + local models
models = reference.list_models()

# Summary stats
stats = reference.stats("eight_schools", params=["mu", "tau"])

# Draws (Arrow by default)
arrow_table = reference.draws("eight_schools", return_="arrow")

# Draws wrapper with conversion helpers
wrapper = reference.draws("eight_schools", return_="draws")
np_draws = wrapper.to_numpy()  # requires numpy

# Compare against your draws
result = reference.compare("eight_schools", actual=my_draws, tolerance=0.15)
assert result.passed
```

## Data store layout

Bundled data lives in the package. Users can extend/override by adding models
under `~/.mcmc-ref/`.

```
~/.mcmc-ref/
  draws/          # User-added models (.draws.parquet)
  meta/           # User-added metadata (.meta.json)

src/mcmc_ref/data/
  draws/          # Bundled .draws.parquet files
  meta/           # Bundled .meta.json files
```

Local models take precedence on name collisions.

## Data formats

### CSV (draws)

Required header:

- `chain` (int, 0-based)
- `draw` (int, 0-based)
- one column per parameter (float). Vector params are `theta[1]`, `theta[2]`, ...

Rules:
- Row order does not matter; `chain` + `draw` define identity.
- Missing values are rejected.
- Extra columns are rejected unless explicitly allowed.

### JSON-zip (draws)

`convert` supports posteriordb-style JSON-zip:

- File contains a single JSON blob.
- Top-level is a list of chains.
- Each chain is a dict mapping `param -> list[draws]`.

Example:

```json
[
  {"mu": [1.0, 1.1, ...], "tau": [0.2, 0.3, ...]},
  {"mu": [0.9, 1.0, ...], "tau": [0.4, 0.35, ...]}
]
```

### Parquet (draws)

Parquet uses the same schema as CSV: `chain`, `draw`, then one column per
parameter. This is the primary on-disk format.

## CLI reference

```text
mcmc-ref list [--format table|json]
mcmc-ref stats <model> [--params p1,p2] [--format table|csv|json] [--quantile-mode exact] [--include-diagnostics]
mcmc-ref draws <model> [--params p1,p2] [--chains 0,1] [--format csv|parquet]
mcmc-ref diagnostics <model> [--format table|csv|json]
mcmc-ref info <model>
mcmc-ref compare <model> --actual <file.csv> [--tolerance 0.15] [--format table|json]
mcmc-ref convert <file.json.zip|file.csv> --name <model_name> [--force]
```

Exit codes:
- `0` success
- `1` error (bad input, missing model)
- `2` comparison failed

## Python API reference

```python
from mcmc_ref import reference

reference.list_models() -> list[str]

reference.stats(
    model: str,
    params: list[str] | None = None,
    backend: str = "arrow",
    quantile_mode: str = "exact",
) -> dict[str, dict[str, float]]

reference.draws(
    model: str,
    params: list[str] | None = None,
    chains: list[int] | None = None,
    return_: str = "arrow",  # "arrow" | "draws" | "numpy" | "list"
)

reference.diagnostics_for_model(
    model: str,
    params: list[str] | None = None,
) -> dict[str, dict[str, float]]

reference.compare(
    model: str,
    actual: dict[str, list[float]],
    tolerance: float = 0.15,
    metrics: list[str] = ["mean", "std"],
) -> CompareResult
```

## Diagnostics

Diagnostics are computed via rank-normalized split R-hat (with folded variant)
plus bulk/tail ESS. These implementations use only the stdlib to keep
dependencies light.

```python
from mcmc_ref import diagnostics

chains = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]
rhat = diagnostics.split_rhat(chains)
ess = diagnostics.ess_bulk(chains)
```

## Workflow: adding a new model

1. Write or obtain a Stan model and data.
2. Check priors: if the model uses extremely flat priors, re-draw with
   weakly informative priors (see Stan prior recommendations).
3. Run CmdStan with posteriordb standard settings:
   - 10 chains, 10k warmup, 10k sampling, thin=10, seed=4711
4. Export draws as JSON-zip (posteriordb format) or CSV.
5. Convert to Parquet + metadata:

```bash
mcmc-ref convert my_model.json.zip --name my_model
```

6. Verify:

```bash
mcmc-ref stats my_model
```

## Integration tests (posteriordb)

Integration tests compare against existing posteriordb reference draws when
available. Search order:

1. `POSTERIORDB_REF_DRAWS` or `POSTERIORDB_DRAWS_DIR`
2. `POSTERIORDB_PATH` or `POSTERIORDB_ROOT` (auto-detect subpath)
3. `/tmp/jaxstanv3/tests/posteriordb/reference_draws`

If nothing is found, tests skip. You can opt-in to generating missing reference
 draws (heavy; CmdStan required):

```bash
MCMC_REF_GENERATE=1 MCMC_REF_GENERATE_MODEL=wells_data-wells_dist uv run pytest
```

## Development

Tooling (Astral stack):
- `uv` for envs + running tools
- `ruff` for lint/format
- `ty` for type checking (strict)

Commands:

```bash
uv run ruff check .
uv run ruff format .
uv run ty check .
uv run pytest
```

## Dependencies

Required:
- `pyarrow`
- `click`

Optional:
- `numpy` (faster stats, NumPy return type)

## Limitations / notes

- Quantile mode is currently `exact` only (approximate mode may be added later).
- Very large models may require streaming pipelines; Arrow-first paths keep
  memory in check but conversions to NumPy will materialize arrays.

## Contributing

- Follow red -> green TDD.
- Add tests for every behavior change.
- Keep type checking green (`ty`).
- When adding models, prefer weakly informative priors over flat priors.
