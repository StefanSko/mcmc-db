from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reference draws from provenance recipes."
    )
    parser.add_argument("--scaffold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from mcmc_ref.generate import generate_reference_corpus

    model_list = args.models.split(",") if args.models else None
    result = generate_reference_corpus(
        scaffold_root=args.scaffold_root,
        output_root=args.output_root,
        models=model_list,
        force=args.force,
    )
    print(f"generated={result.generated} failed={result.failed} output={result.output_root}")
    if result.errors:
        for name, message in sorted(result.errors.items()):
            print(f"- {name}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
