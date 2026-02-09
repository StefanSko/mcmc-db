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

This builds the default/posteriordb corpus.

## Full Provenance Generation With CmdStan

If you want provenance generated in this repository (instead of importing zips
from another source), generate raw references directly from posteriordb with
CmdStan.

Install generation dependencies:

```bash
uv add --dev "mcmc-ref[generate,bootstrap]"
```

Install CmdStan (one-time):

```bash
uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Generate posteriors from posteriordb:

```bash
uv run --extra dev --extra generate python scripts/generate_posteriordb_references.py \
  --posteriordb-path ~/.posteriordb/posterior_database \
  --output-dir generated_references/posteriordb \
  --posterior eight_schools-eight_schools_noncentered
```

Outputs include:
- `generated_references/posteriordb/draws/{posterior}.json.zip`
- `generated_references/posteriordb/provenance/{posterior}.provenance.json`
- `generated_references/posteriordb/stan_models/{posterior}.stan`
- `generated_references/posteriordb/generation_manifest.json`

Then convert/import these references into `~/.mcmc-ref` via:
- `scripts/build_references.py` (for standard names)
- `scripts/import_informed_references.py` (for `_informed` variants)

## Posteriordb-Free Local Generation (Recommended)

For day-to-day generation, use local model+data files only:
- `data/stan_models/{model}.stan`
- `data/stan_data/{model}.data.json`

One-time bootstrap from posteriordb:

```bash
uv add --dev "mcmc-ref[bootstrap]"
uv run --extra dev --extra bootstrap python scripts/sync_stan_models.py \
  --draws-dir packages/mcmc-ref-data/src/mcmc_ref_data/data/draws \
  --target-dir packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_models \
  --posteriordb-path ~/.posteriordb/posterior_database \
  --informed-stan-dir /tmp/jaxstanv3/tests/posteriordb/informed_references/stan_models

uv run --extra dev --extra bootstrap python scripts/sync_stan_data.py \
  --draws-dir packages/mcmc-ref-data/src/mcmc_ref_data/data/draws \
  --target-dir packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_data \
  --posteriordb-path ~/.posteriordb/posterior_database
```

Generate references from local files (no posteriordb required):

```bash
uv run --extra dev --extra generate python scripts/generate_local_references.py \
  --models-dir packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_models \
  --data-dir packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_data \
  --output-dir generated_references/local \
  --model eight_schools-eight_schools_noncentered
```

Outputs:
- `generated_references/local/draws/{model}.json.zip`
- `generated_references/local/provenance/{model}.provenance.json`
- `generated_references/local/stan_models/{model}.stan`
- `generated_references/local/stan_data/{model}.data.json`
- `generated_references/local/generation_manifest.json`

## Add Informed-Prior References (Maintainers)

Import informed-prior references (for example the `jaxstanv3` informed set):

```bash
uv run --extra dev python scripts/import_informed_references.py \
  --source-dir /tmp/jaxstanv3/tests/posteriordb/informed_references/draws \
  --output-root ~/.mcmc-ref
```

Imported models are tagged in metadata with:
- `reference_variant: "informed_prior"`
- `informed_reference_info` (when the source `.info.json` exists)

Typical informed models currently available:
- `blr_informed`
- `kidscore_momiq_informed`
- `logearn_height_informed`
- `mesquite_logvolume_informed`
- `radon_pooled_informed`

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

## Add Stan Models Next To Packaged Draws

To bundle the actual Stan model code with packaged references:

```bash
uv run --extra dev --extra generate python scripts/sync_stan_models.py \
  --draws-dir packages/mcmc-ref-data/src/mcmc_ref_data/data/draws \
  --target-dir packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_models \
  --posteriordb-path ~/.posteriordb/posterior_database \
  --informed-stan-dir /tmp/jaxstanv3/tests/posteriordb/informed_references/stan_models
```

This writes:
- standard model `.stan` files from posteriordb
- informed `_informed.stan` files from the informed source directory

## Do You Still Need posteriordb?

- For using packaged references in framework tests: **no**.
- For generating new references from local model+data files: **no**.
- For bootstrapping new model/data from posteriordb by posterior name: **yes**.
- If you already have local Stan code + data and only need conversion/comparison:
  you can skip posteriordb and use `convert` directly.

## Quality Notes

- Diagnostics use rank-normalized split R-hat and ESS.
- Defaults enforce 4-chain diagnostics.
- `convert --force` allows bypassing strict checks for non-canonical workflows.
