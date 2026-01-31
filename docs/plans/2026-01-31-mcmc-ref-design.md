# mcmc-ref: Reference Posterior Validation Tool

## Purpose

A standalone tool providing reference MCMC draws for validating Bayesian inference libraries. Ships pre-computed reference posteriors (from posteriordb + CmdStan) in an efficient columnar format, with both a CLI and Python API for querying, comparing, and debugging.

**Primary use case**: A Bayesian library (e.g., jaxstan, PyMC, numpyro) includes `mcmc-ref` as a test dependency, runs its sampler on known models, and validates results against the reference draws.

## Design Decisions

- **Parquet storage**: Reference draws stored as Parquet files (columnar, compressed, readable by any data tool). One file per model.
- **Dual interface**: CLI for shell/CI/UNIX composition, Python API for pytest integration.
- **Bundled data**: Pre-computed references ship with the package (~15-20MB). No setup needed.
- **Local store**: Users can add custom models to `~/.mcmc-ref/draws/`.
- **No CmdStan dependency at runtime**: Generation is external, documented. The tool is read-only over pre-computed data.
- **DuckDB is not a dependency**: Users can pipe CSV output to DuckDB CLI if they want SQL queries.

## Data Model

### Draws Parquet Schema

One file per model: `{model_name}.draws.parquet`

| Column    | Type    | Description                          |
|-----------|---------|--------------------------------------|
| chain     | int32   | Chain index (0-9)                    |
| draw      | int32   | Draw index within chain (0-999)      |
| {param}   | float64 | One column per parameter             |

- 10,000 rows per model (10 chains x 1,000 draws)
- Vector parameters expanded: `theta[1]`, `theta[2]`, etc.
- Columnar layout means reading one parameter doesn't load others

### Metadata Sidecar

One file per model: `{model_name}.meta.json`

```json
{
  "model": "eight_schools_data-eight_schools_centered",
  "parameters": ["mu", "tau", "theta[1]", "theta[2]", ...],
  "n_chains": 10,
  "n_draws_per_chain": 1000,
  "inference": {
    "method": "stan_sampling",
    "seed": 4711,
    "chains": 10,
    "iter_sampling": 10000,
    "iter_warmup": 10000,
    "thin": 10
  },
  "diagnostics": {
    "rhat": {"mu": 1.0004, "tau": 1.0006},
    "ess_bulk": {"mu": 9640, "tau": 8832},
    "ess_tail": {"mu": 9701, "tau": 8956},
    "divergences_per_chain": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  },
  "source": "posteriordb",
  "generated_date": "2025-12-17",
  "checks": {
    "ndraws_is_10k": true,
    "nchains_is_gte_4": true,
    "ess_above_400": true,
    "rhat_below_1_01": true,
    "no_divergences": true
  }
}
```

### Data Store Layout

```
~/.mcmc-ref/
  draws/          # User-added models (.draws.parquet)
  meta/           # User-added metadata (.meta.json)

Bundled in package:
  src/mcmc_ref/data/
    draws/        # Pre-computed .draws.parquet files
    meta/         # Pre-computed .meta.json files
```

Bundled models are read from the package. Local store augments bundled data (local takes precedence on name collision).

## Data Formats (Inputs + Outputs)

Clarify the wire formats used by `convert` and `compare --actual` to avoid guessing.

### CSV (Draws)

**Required header** with the following columns:

- `chain` (int, 0-based)
- `draw` (int, 0-based)
- One column per parameter (float). Vector params use Stan-style names: `theta[1]`, `theta[2]`, ...

**Rules**:
- Row order does not matter; `chain` + `draw` define identity.
- Extra columns are rejected unless `--ignore-extra-cols` is provided.
- Missing values are rejected.
- For `compare --actual`, if `chain` and `draw` are omitted, the file is treated as a single chain with sequential draws.

### JSON-zip (Reference Draws)

`convert` accepts a **zipped JSON** with a minimal, documented schema (recommended for local use), and also supports posteriordb JSON-zip via an adapter.

**mcmc-ref JSON-zip schema (recommended)**:

```json
{
  "schema_version": 1,
  "model": "eight_schools_data-eight_schools_centered",
  "parameters": ["mu", "tau", "theta[1]", "theta[2]"],
  "n_chains": 10,
  "n_draws_per_chain": 1000,
  "draws": [
    { "chain": 0, "draw": 0, "mu": 3.1, "tau": 1.2, "theta[1]": 2.9, "theta[2]": 3.5 },
    ...
  ]
}
```

**posteriordb JSON-zip**:
- Supported via a converter adapter in `convert.py`.
- The README should link to the posteriordb format and describe any required keys.

### Parquet (Draws Output)

The Parquet schema mirrors the CSV layout (column per parameter plus `chain` and `draw`), so round-tripping is lossless.

## CLI Interface

Package: `mcmc-ref` (PyPI), runnable via `uvx mcmc-ref`.

### Commands

```
mcmc-ref list [--format table|json]
mcmc-ref stats <model> [--params p1,p2] [--format table|csv|json] [--quantile-mode exact]
mcmc-ref draws <model> [--params p1,p2] [--chains 0,1] [--format csv|parquet]
mcmc-ref diagnostics <model> [--format table|csv|json]
mcmc-ref info <model>
mcmc-ref compare <model> --actual <file.csv> [--tolerance 0.15] [--format table|json]
mcmc-ref convert <file.json.zip|file.csv> --name <model_name>
```

### Output Format Defaults

- Terminal (isatty): `table`
- Piped: `csv`

### Exit Codes

- 0: Success
- 1: Error (model not found, bad input, etc.)
- 2: Validation failure (compare command, checks failed)

### Example Usage

```bash
# Quick stats check
mcmc-ref stats eight_schools --params mu,tau

# Pipe to DuckDB for ad-hoc analysis
mcmc-ref draws eight_schools --format csv | \
  duckdb -c "SELECT chain, avg(tau), stddev(tau) FROM read_csv('/dev/stdin') GROUP BY chain"

# CI validation
mcmc-ref compare eight_schools --actual my_results.csv --tolerance 0.15

# Zero-install
uvx mcmc-ref stats wells_data-wells_dist --format json
```

## Python API

```python
from mcmc_ref import reference
```

### Core Functions

```python
reference.list_models() -> list[str]

reference.stats(
    model: str,
    params: list[str] | None = None,
) -> dict[str, dict]
# Returns: {"mu": {"mean": 4.35, "std": 3.28, "q5": ..., "q50": ..., "q95": ..., "ess_bulk": ..., "rhat": ...}, ...}

reference.draws(
    model: str,
    params: list[str] | None = None,
    chains: list[int] | None = None,
) -> np.ndarray
# Returns: numpy array of shape (n_draws, n_params) or (n_chains, n_draws_per_chain, n_params)

reference.diagnostics(
    model: str,
) -> dict[str, dict]
# Returns: {"mu": {"ess_bulk": 9640, "ess_tail": 9701, "rhat": 1.0004}, ...}
```

### Lightweight Backend + Draws Wrapper (no heavy deps)

To avoid hard dependencies on NumPy/Pandas, keep Arrow as the core and make NumPy optional.

**Backend selection**

```python
reference.stats(..., backend="arrow")   # default, no NumPy needed
reference.stats(..., backend="numpy")   # optional acceleration if NumPy is installed
```

**Draws wrapper**

```python
class Draws:
    def __init__(self, reader, params, chains, meta): ...
    def to_arrow(self): ...
    def to_numpy(self): ...   # raises if NumPy is not installed
    def to_list(self): ...

reference.draws(
    model: str,
    params: list[str] | None = None,
    chains: list[int] | None = None,
    return_: str = "arrow",  # "arrow" | "numpy" | "list" | "draws"
) -> Draws | pyarrow.Table | np.ndarray | list
```

**Behavior**
- Default `return_="arrow"` uses `pyarrow.Table` or a `RecordBatchReader` internally.
- `return_="draws"` returns the lightweight wrapper with conversion helpers.
- `return_="numpy"` is allowed only if NumPy is installed.

This keeps the API elegant while remaining minimal for CLI and CI usage.

### Compare Helper

```python
@dataclass
class ParamResult:
    ref: float
    actual: float
    rel_error: float
    passed: bool

@dataclass
class CompareResult:
    passed: bool
    details: dict[str, dict[str, ParamResult]]  # param -> metric -> result
    failures: list[str]                           # human-readable failure messages

reference.compare(
    model: str,
    actual: dict[str, np.ndarray],   # param_name -> draws array
    tolerance: float = 0.15,
    metrics: list[str] = ["mean", "std"],
) -> CompareResult
```

### Usage in pytest

```python
def test_eight_schools():
    fit = my_library.sample("eight_schools", ...)
    result = reference.compare("eight_schools", fit.as_dict(), tolerance=0.15)
    assert result.passed, "\n".join(result.failures)
```

## Package Structure

```
mcmc-ref/
  pyproject.toml
  src/
    mcmc_ref/
      __init__.py
      reference.py        # Python API (stats, draws, diagnostics, compare)
      cli.py              # CLI entry point
      store.py            # Data store (bundled + local, parquet I/O)
      convert.py          # JSON-zip / CSV -> Parquet conversion
      compare.py          # Comparison logic (used by both API and CLI)
      data/
        draws/            # Bundled .draws.parquet files
        meta/             # Bundled .meta.json files
  tests/
  scripts/
    build_references.py   # One-time: convert jaxstanv3 reference draws to parquet
```

## Memory-Safe I/O Sketch (for store.py and CLI)

Goal: avoid materializing the full table when only a subset of columns/rows are needed.

**Read path (select columns, filter chains)**

```python
import pyarrow.dataset as ds

def read_draws(path, params=None, chains=None, batch_size=1024):
    columns = ["chain", "draw"] + (params or [])
    dataset = ds.dataset(path, format="parquet")
    filt = None if chains is None else ds.field("chain").isin(chains)
    scanner = dataset.scanner(columns=columns, filter=filt, batch_size=batch_size)
    return scanner.to_reader()  # RecordBatchReader
```

**CLI streaming (CSV output without Pandas)**

```python
import pyarrow.csv as pacsv

reader = read_draws(path, params=params, chains=chains)
pacsv.write_csv(reader, output_stream)
```

**Stats (per-column, bounded memory)**

Compute stats per parameter by iterating over batches and maintaining rolling summaries. For quantiles, either:
- Use `pyarrow.compute.quantile` on each column (loads a single column at a time), or
- Use an approximate streaming algorithm (e.g., t-digest) if exact quantiles are too costly.

### Dependencies

**Required**:
- `pyarrow` - Parquet read/write, Arrow compute, CSV streaming
- `click` - CLI framework

**Optional**:
- `numpy` - Faster stats + `np.ndarray` return type (API falls back to Arrow / Python lists if unavailable)
- `pandas` - Not required (prefer Arrow streaming for CLI output)

**No dependency on**: cmdstanpy, posteriordb, duckdb, jax, stan (at runtime).

### Lightweight Diagnostics Module

Provide an internal diagnostics module (or a tiny standalone package) that computes:
- Rank-normalized split R-hat (with folded variant)
- ESS (bulk + tail)
- Divergence counts (if present in input)

Implementation should use only the Python stdlib + `pyarrow` (optional `numpy` acceleration).
Expose a small API that other libraries can reuse:

```python
from mcmc_ref.diagnostics import summarize, rhat, ess_bulk, ess_tail
```

This avoids an ArviZ dependency while keeping the metrics available for quality checks and `stats`.

## Bundled Models

Initial set, converted from jaxstanv3 reference draws:

| Model | Parameters | Source |
|-------|-----------|--------|
| eight_schools_data-eight_schools_centered | mu, tau, theta[1-8] | posteriordb |
| wells_data-wells_dist | beta[1-2] | local reference |
| GLM_Poisson_Data-GLM_Poisson_model | ~43 params | local reference |
| GLM_Binomial_data-GLM_Binomial_model | multiple | local reference |
| irt_2pl-irt_2pl | many | local reference |
| radon_mn-radon_hierarchical_intercept_noncentered | multiple | local reference |
| dugongs_data-dugongs_model | multiple | local reference |
| blr_informed | beta[1-3], sigma | informed reference |
| kidscore_momiq_informed | beta[0-1], sigma | informed reference |
| logearn_height_informed | beta_0, beta_1, sigma | informed reference |
| mesquite_logvolume_informed | beta[1-4], sigma | informed reference |
| radon_pooled_informed | beta[0-1], sigma | informed reference |

Plus any models available directly from posteriordb.

## Workflow: Adding a New Model

Documented in README (not built into the tool as a complex command):

1. Write/obtain a Stan model and data
2. Check priors: if the model uses extremely flat priors, re-draw with informed priors per Stan's prior-choice recommendations.
3. Run CmdStan with posteriordb-standard settings (10 chains, 10k warmup, 10k sampling, thin 10, seed 4711)
4. Export draws as JSON-zip (posteriordb format) or CSV
5. Run `mcmc-ref convert <file> --name my_model`
6. Verify: `mcmc-ref stats my_model`
7. Optionally commit the parquet file to your repo or share it

## Quality Standards

All bundled references must pass:
- Exactly 10,000 draws (10 chains x 1,000)
- R-hat < 1.01 for all parameters
- ESS bulk > 400 for all parameters
- No divergent transitions
- At least 4 chains

The `convert` command enforces these checks and rejects draws that don't meet them (with `--force` override).
  --include-diagnostics   # add rhat/ess to stats output
