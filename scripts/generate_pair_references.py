#!/usr/bin/env python
"""Generate pair reference draws and publish to the data package.

Requires CmdStan. Install with: uv add --dev cmdstanpy
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcmc_ref.generate import generate_reference_corpus
from mcmc_ref.provenance import materialize_scaffold

PAIR_MODELS = [
    "bangladesh_contraceptive-noncentered",
    "eight_schools-noncentered",
    "hierarchical_lr-noncentered",
    "neals_funnel-noncentered",
    "varying_slopes-noncentered",
]


def main() -> None:
    scaffold_root = Path("/tmp/mcmc-ref-scaffold")
    output_root = Path("/tmp/mcmc-ref-generated")

    print("Materializing scaffold...")
    materialize_scaffold(scaffold_root)

    print(f"Generating {len(PAIR_MODELS)} pair reference models...")
    result = generate_reference_corpus(
        scaffold_root=scaffold_root,
        output_root=output_root,
        models=PAIR_MODELS,
    )
    print(f"Generated: {result.generated}, Failed: {result.failed}")
    for name, err in result.errors.items():
        print(f"  FAILED {name}: {err}")

    if result.failed > 0:
        raise SystemExit(1)

    # Copy to data package
    package_root = Path("packages/mcmc-ref-data/src/mcmc_ref_data/data")
    for f in sorted((output_root / "draws").glob("*.draws.parquet")):
        shutil.copy2(f, package_root / "draws" / f.name)
        print(f"  Copied draws/{f.name}")
    for f in sorted((output_root / "meta").glob("*.meta.json")):
        shutil.copy2(f, package_root / "meta" / f.name)
        print(f"  Copied meta/{f.name}")

    print("Done.")


if __name__ == "__main__":
    main()
