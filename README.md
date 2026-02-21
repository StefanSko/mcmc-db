# mcmc-ref

Reference posterior validation toolkit for Bayesian inference frameworks.

For integration, provenance workflow, and release/CI guidance:
- `docs/integration-guide.md`
- `docs/release-checklist.md`
- `docs/releases/0.1.1.md`
- `docs/releases/0.1.2.md`
- `docs/releases/0.1.3.md`

Design notes:
- `docs/plans/2026-01-31-mcmc-ref-design.md`
- `docs/plans/2026-02-21-mcmc-ref-consolidation-plan.md`

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run ty check .
uv run pytest
```

## Canonical Provenance Flow

```bash
# 1) materialize deterministic scaffold
mcmc-ref provenance-scaffold --output-root /tmp/mcmc-db/provenance

# 2) generate draws/meta from recipes (requires cmdstanpy + CmdStan)
mcmc-ref provenance-generate \
  --scaffold-root /tmp/mcmc-db/provenance \
  --output-root /tmp/mcmc-db/generated

# 3) publish into packaged data layout
mcmc-ref provenance-publish \
  --source-root /tmp/mcmc-db/generated \
  --scaffold-root /tmp/mcmc-db/provenance \
  --package-root packages/mcmc-ref-data/src/mcmc_ref_data/data
```

Legacy conversion/import scripts remain available but are deprecated.

## Origin Story

Brainstorm:
- https://gisthost.github.io/?a47c363027aba4615cd3247ed5f88793

Implementation:
- https://gistpreview.github.io/?018116c7a98a7d75e363e77689ec7b26
