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
scale, and re-draw posteriordb models that use extremely flat priors so the
reference is stable and diagnostic-friendly.

Diagnostics use rank-normalized split R-hat and bulk/tail ESS with stdlib-only
implementations to keep dependencies light.

For integration tests, you can optionally generate missing reference draws by
setting `MCMC_REF_GENERATE=1` and (optionally) `MCMC_REF_GENERATE_MODEL=...`.

## Origin Story:

Brainstorm:
    -https://gisthost.github.io/?a47c363027aba4615cd3247ed5f88793

Implemntation: 
    -https://gistpreview.github.io/?018116c7a98a7d75e363e77689ec7b26
