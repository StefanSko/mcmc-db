# mcmc-ref Consolidation Plan (2026-02-21)

## Goal
Make `mcmc-ref` a single, modern validation source for probabilistic inference libraries with:
- one canonical provenance pipeline,
- one canonical packaged data surface,
- one canonical integration path for consumers (for example `jaxstanv3`).

## Problems To Eliminate
- Multiple overlapping generation paths (`build_references`, `generate_draws`, `cmdstan_generate`, `local_generate`, `informed_references`, ad-hoc scripts).
- Mixed source-of-truth framing (posteriordb bootstrap flows vs mcmc-db-native provenance).
- Inconsistent model naming and parameter naming across bundled refs and pair refs.
- Consumer repos bundling local copies of draws/meta/pairs.

## Target Architecture

### 1) Canonical provenance flow (single path)
1. `mcmc-ref provenance-scaffold`
2. `mcmc-ref provenance-generate`
3. `mcmc-ref provenance-publish`
4. publish package artifacts (`mcmc-ref-data`, then `mcmc-ref`)

All core/informed/pair references must flow through this path.

### 2) Canonical data surface
- Keep packaged reference data in `packages/mcmc-ref-data/src/mcmc_ref_data/data/`.
- Keep `src/mcmc_ref/data/` only as runtime fallback for minimal local dev fixtures (if needed), not as an independently curated corpus.
- DataStore load order should be explicit and documented (local override -> package data).

### 3) Canonical consumer integration
- Consumers use `mcmc-ref`/`mcmc-ref-data` only.
- No checked-in local reference draws/meta/pairs in consumer repos.
- Optional `MCMC_REF_LOCAL_ROOT` only for local experimentation.

## Phased Execution

### Phase A (immediate, next release line)
- Finalize and stabilize provenance recipe inventory for all curated models.
- Add explicit provenance manifest fields for generator version and source commit.
- Keep legacy commands but mark as deprecated in CLI help/docs.
- Update docs to make provenance flow primary and legacy flows secondary.

### Phase B (one release later)
- Route legacy commands internally to the provenance pipeline where possible.
- Introduce a model-name alias layer for compatibility (`_` vs `-` legacy IDs).
- Add validation tests ensuring all packaged models have:
  - draws parquet,
  - meta,
  - stan model,
  - stan data,
  - provenance manifest entry.

### Phase C (cleanup release)
- Remove deprecated paths and scripts that duplicate provenance flow.
- Remove posteriordb-first language from integration docs; keep posteriordb only as optional import/bootstrap tool.
- Freeze a single model-id and parameter naming policy and document it as compatibility contract.

## Compatibility Policy
- Preserve existing public API (`reference.*`, `DataStore`, `pairs` API) across consolidation.
- Keep compatibility aliases for legacy model IDs for at least one minor release.
- Ship migration notes for consumer repos before removal of deprecated paths.

## Release/Validation Gates For Consolidation
- `uv run ruff check .`
- `uv run ty check .`
- `uv run pytest`
- Provenance smoke run in CI:
  - scaffold -> generate (small subset) -> publish -> consumer smoke test.

## jaxstanv3 Clean Integration Plan (Summary)
1. Depend on released `mcm-ref` + `mcmc-ref-data` version.
2. Point tests to package-backed references (optional `MCMC_REF_LOCAL_ROOT` in dev only).
3. Remove `tests/mcmcdb/references/**` local bundles from jaxstanv3.
4. Keep only model input fixtures that are library-specific.
5. Run full validation suite and ensure no fallback paths to `/tmp/jaxstan*` remain.
