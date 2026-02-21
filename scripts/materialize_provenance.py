from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize mcmc-ref provenance scaffold.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where stan_models/, stan_data/, pairs/, and manifest are written.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from mcmc_ref.provenance import materialize_scaffold

    manifest_path = materialize_scaffold(args.output_root)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
