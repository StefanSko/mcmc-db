# Audit Follow-Up: AGENTS Release Workflow Update

## Context

On 2026-02-15, commit `3f54763` was pushed directly to `main`:

- `docs: add clean release workflow for mcmc-db`
- File changed: `AGENTS.md`

The change itself is valid and already on `main`, but this follow-up exists to
preserve explicit review/audit trail in PR history.

## What Was Added in the Direct Commit

- Dedicated release branch policy (`release/vX.Y.Z`)
- Version alignment checklist across core and data packages
- Required validation gates before publish
- Publish order (`mcmc-ref-data` first, then `mcmc-ref`)
- Tag/release steps and downstream handoff requirements

## Audit Decision

- Keep commit `3f54763` as-is on `main` (no revert).
- Record provenance and rationale in this audit note.
- Use this PR as the review artifact for the already-applied process change.
