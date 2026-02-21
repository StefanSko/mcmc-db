# mcmc-ref

Reference posterior validation toolkit for Bayesian inference frameworks.

For installation, integration, data build/publish workflow, and CI patterns, see:
- `docs/integration-guide.md`
- `docs/release-checklist.md`
- `docs/releases/0.1.1.md`
- `docs/releases/0.1.2.md`

Design notes:
- `docs/plans/2026-01-31-mcmc-ref-design.md`

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run ty check .
uv run pytest
```

## Build Canonical References

```bash
# default source: ~/.mcmc-db/reference_archives
# default output: ~/.mcmc-ref/{draws,meta}
uv run --extra dev python scripts/build_references.py
```

## Recreate Provenance Assets

Generate deterministic Stan scaffolding (models, data, geometry pairs):

```bash
mcmc-ref provenance-scaffold --output-root /tmp/mcmc-db/provenance
# or
uv run --extra dev python scripts/materialize_provenance.py --output-root /tmp/mcmc-db/provenance
```

Generate archives/parquet from recipes:

```bash
# requires cmdstanpy + CmdStan
uv add --dev --optional provenance cmdstanpy
mcmc-ref provenance-generate \
  --scaffold-root /tmp/mcmc-db/provenance \
  --output-root /tmp/mcmc-db/generated
```

Publish generated draws/meta/pairs into package data layout:

```bash
mcmc-ref provenance-publish \
  --source-root /tmp/mcmc-db/generated \
  --scaffold-root /tmp/mcmc-db/provenance \
  --package-root src/mcmc_ref/data
```

## Origin Story

Brainstorm:
- https://gisthost.github.io/?a47c363027aba4615cd3247ed5f88793

Implemntation:
- https://gistpreview.github.io/?018116c7a98a7d75e363e77689ec7b26
