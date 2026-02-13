"""Generate reference draws for the 12 new benchmark models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cmdstanpy

from mcmc_ref.cmdstan_generate import (
    build_posteriordb_payload,
    parse_cmdstan_csv,
    write_posteriordb_json_zip,
)
from mcmc_ref.local_generate import discover_local_model_specs, load_stan_data

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

MODELS_DIR = ROOT / "packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_models"
DATA_DIR = ROOT / "packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_data"
OUTPUT_DIR = ROOT / "generated_references" / "new_models"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draws_dir = OUTPUT_DIR / "draws"
    draws_dir.mkdir(exist_ok=True)

    specs = discover_local_model_specs(
        models_dir=MODELS_DIR,
        data_dir=DATA_DIR,
        models=NEW_MODELS,
    )

    for spec in specs:
        zip_path = draws_dir / f"{spec.model}.json.zip"
        if zip_path.exists():
            print(f"SKIP {spec.model} (exists)")
            continue

        print(f"Generating {spec.model}...")
        data_dict = load_stan_data(spec.data_path)
        model = cmdstanpy.CmdStanModel(stan_file=str(spec.stan_path))
        fit = model.sample(
            data=data_dict,
            chains=10,
            iter_warmup=10_000,
            iter_sampling=10_000,
            thin=10,
            seed=4711,
            show_progress=False,
        )

        chain_draws = [
            parse_cmdstan_csv(Path(csv_file))
            for csv_file in fit.runset.csv_files
        ]
        payload = build_posteriordb_payload(chain_draws)
        write_posteriordb_json_zip(payload, zip_path, model_name=spec.model)

        n_draws = len(next(iter(payload[0].values()))) if payload else 0
        print(
            f"  OK: {len(payload)} chains x {n_draws} draws, "
            f"params: {sorted(payload[0].keys())}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
