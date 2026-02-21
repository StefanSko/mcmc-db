from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish generated provenance references into package data."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scaffold-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from mcmc_ref.generate import publish_reference_data

    result = publish_reference_data(
        source_root=args.source_root,
        scaffold_root=args.scaffold_root,
        package_root=args.package_root,
    )
    print(
        "published "
        f"draws={result.draws_copied} meta={result.meta_copied} pairs={result.pairs_copied} "
        f"to={result.package_root}"
    )


if __name__ == "__main__":
    main()
