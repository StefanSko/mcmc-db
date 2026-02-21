# Release Checklist

This repository publishes two packages:

- `mcmc-ref` (core library + CLI)
- `mcmc-ref-data` (reference corpus)

Keep versions aligned for `mcmc-ref[data]`.

## 1) Pre-release checks

```bash
uv run ruff check .
uv run ty check .
uv run pytest
```

## 2) Build artifacts

```bash
uv build
cd packages/mcmc-ref-data && uv build && cd ../..
```

## 3) Data provenance refresh (before publish)

```bash
mcmc-ref provenance-scaffold --output-root /tmp/mcmc-db/provenance
mcmc-ref provenance-generate --scaffold-root /tmp/mcmc-db/provenance --output-root /tmp/mcmc-db/generated
mcmc-ref provenance-publish \
  --source-root /tmp/mcmc-db/generated \
  --scaffold-root /tmp/mcmc-db/provenance \
  --package-root packages/mcmc-ref-data/src/mcmc_ref_data/data
```

## 4) Publish order

Publish data package first, then core package:

```bash
cd packages/mcmc-ref-data
uv publish
cd ../..
uv publish
```

## 5) Consumer install (jaxstanv3)

```bash
uv add "mcmc-ref[data]==0.1.3"
```
