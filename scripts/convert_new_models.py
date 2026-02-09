"""Convert generated reference draws to parquet + metadata for new benchmark models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcmc_ref.convert import convert_file

NEW_MODELS = [
    "admit-logistic_regression",
    "oceanic_tools-poisson_regression",
    "trolley-ordered_logistic",
    "oceanic_tools-negative_binomial",
    "manuscripts-zero_inflated_poisson",
    "admit_dept-beta_regression",
    "tadpoles-multilevel_binomial",
    "chimpanzees-multilevel_varying_slopes",
    "cafe-multivariate_normal",
    "earnings-robust_regression",
    "employees-weibull_survival",
    "coal_mining-changepoint",
]

INPUT_DIR = ROOT / "generated_references" / "new_models" / "draws"
OUT_DRAWS = ROOT / "packages" / "mcmc-ref-data" / "src" / "mcmc_ref_data" / "data" / "draws"
OUT_META = ROOT / "packages" / "mcmc-ref-data" / "src" / "mcmc_ref_data" / "data" / "meta"


def main() -> None:
    passed = []
    failed = []

    for model in NEW_MODELS:
        zip_path = INPUT_DIR / f"{model}.json.zip"
        if not zip_path.exists():
            print(f"MISSING {model}")
            failed.append((model, "zip not found"))
            continue

        try:
            result = convert_file(
                zip_path,
                name=model,
                out_draws_dir=OUT_DRAWS,
                out_meta_dir=OUT_META,
                force=False,
            )
            checks = result.meta["checks"]
            all_ok = all(checks.values())
            status = "PASS" if all_ok else "WARN"
            print(f"{status} {model}: {checks}")
            if all_ok:
                passed.append(model)
            else:
                failed.append((model, checks))
        except Exception as e:
            print(f"FAIL {model}: {e}")
            failed.append((model, str(e)))

    print(f"\n{len(passed)}/{len(NEW_MODELS)} models converted successfully")
    if failed:
        print("Failed models:")
        for model, reason in failed:
            print(f"  - {model}: {reason}")


if __name__ == "__main__":
    main()
