from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from mcmc_ref.informed_references import main as import_main

    import_main()


if __name__ == "__main__":
    main()
