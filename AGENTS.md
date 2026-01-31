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
