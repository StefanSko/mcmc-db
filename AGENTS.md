# AGENTS.md

## Purpose
Provide a minimal, consistent workflow for contributors and coding agents.

## Tooling (Astral stack)
- Use `uv` for environments, dependency management, and running tools.
- Use `ruff` for linting/formatting.
- Use `ty` for type checking (type correctness is a hard requirement).

## Development Workflow (TDD)
- Follow **red -> green** TDD: write a failing test first, then implement the change until tests pass.
- Keep changes small and incremental, with tests updated/added for each behavior.

## Testing Expectations
- Unit tests for core logic.
- Integration tests for CLI and data I/O.
- Integration tests must compare against existing posteriordb models **when reference draws exist**.
- If posteriordb draws are missing, fall back to checking the local cache:
  - `/tmp/jaxstanv3/tests/posteriordb/reference_draws` (data path, not executable)
- If both are missing, integration tests may **optionally** generate reference draws
  via `/tmp/jaxstanv3/tests/posteriordb/reference_draws/generate_reference.py` when
  `MCMC_REF_GENERATE=1` is set (CmdStan is heavy; avoid in CI by default).

## Commands (examples)
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run ty check .`
- `uv run pytest`

## Clean Release Workflow (mcmc-db)
- Always release from a dedicated branch: `release/vX.Y.Z`.
- Keep versions aligned in all release-critical files before publishing:
  - Root `pyproject.toml`: `project.version = "X.Y.Z"`
  - Root `pyproject.toml`: `project.optional-dependencies.data = ["mcmc-ref-data==X.Y.Z"]`
  - `packages/mcmc-ref-data/pyproject.toml`: `project.version = "X.Y.Z"`
- Create release notes before publish: `docs/releases/X.Y.Z.md`.
- Refresh lock file after version/dependency changes: `uv lock`.
- Enforce release quality gates on the release branch:
  - `uv run ruff check .`
  - `uv run ty check .`
  - `uv run pytest`
- Build both packages:
  - `uv build`
  - `cd packages/mcmc-ref-data && uv build && cd ../..`
- Publish in strict order (data first, then core):
  - `cd packages/mcmc-ref-data && uv publish && cd ../..`
  - `uv publish`
- Tag and release from the exact published commit:
  - `git tag vX.Y.Z`
  - `git push origin vX.Y.Z`
  - Create GitHub Release `vX.Y.Z` using `docs/releases/X.Y.Z.md`.
- Merge `release/vX.Y.Z` to `main` only after publish/tag/release succeed.
- Downstream handoff (for clean integration):
  - Update consumers (for example `jaxstanv3`) to the released revision/version.
  - Re-lock dependencies (`uv lock`) and rerun their integration tests.
