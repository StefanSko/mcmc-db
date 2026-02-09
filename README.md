# mcmc-ref

Reference posterior validation toolkit for Bayesian inference frameworks.

For installation, integration, data build/publish workflow, and CI patterns,
see:

- `docs/integration-guide.md`

Current packaged corpus includes both default posteriordb references and
informed-prior variants (for models where informed references are available),
and can include `.stan` model files next to draws.
The integration guide documents posteriordb-free local generation with CmdStan.

Design notes:

- `docs/plans/2026-01-31-mcmc-ref-design.md`

## Origin Story

Brainstorm:
- https://gisthost.github.io/?a47c363027aba4615cd3247ed5f88793

Implemntation:
- https://gistpreview.github.io/?018116c7a98a7d75e363e77689ec7b26
