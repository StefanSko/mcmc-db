from __future__ import annotations

import importlib.metadata
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click


def _require_cmdstanpy():
    try:
        import cmdstanpy
    except Exception as exc:
        raise SystemExit(
            "Missing generation dependency. Run: uv add --dev \"mcmc-ref[generate]\""
        ) from exc
    return cmdstanpy


@click.command("generate-local-references")
@click.option(
    "--models-dir",
    type=click.Path(path_type=Path),
    default=Path("packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_models"),
    show_default=True,
    help="Directory containing local .stan models.",
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_data"),
    show_default=True,
    help="Directory containing local .data.json files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("generated_references/local"),
    show_default=True,
    help="Output directory for draws/provenance files.",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    help="Model name to generate (repeat option for multiple).",
)
@click.option("--limit", default=None, type=int, help="Only generate first N models.")
@click.option("--chains", default=10, show_default=True, type=int)
@click.option("--iter-warmup", default=10_000, show_default=True, type=int)
@click.option("--iter-sampling", default=10_000, show_default=True, type=int)
@click.option("--thin", default=10, show_default=True, type=int)
@click.option("--seed", default=4711, show_default=True, type=int)
@click.option("--overwrite", is_flag=True, help="Overwrite existing json.zip/provenance files.")
def main(
    models_dir: Path,
    data_dir: Path,
    output_dir: Path,
    models: tuple[str, ...],
    limit: int | None,
    chains: int,
    iter_warmup: int,
    iter_sampling: int,
    thin: int,
    seed: int,
    overwrite: bool,
) -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from mcmc_ref.cmdstan_generate import (
        build_posteriordb_payload,
        parse_cmdstan_csv,
        write_posteriordb_json_zip,
        write_provenance,
    )
    from mcmc_ref.local_generate import discover_local_model_specs, load_stan_data

    cmdstanpy = _require_cmdstanpy()
    specs = discover_local_model_specs(
        models_dir=models_dir,
        data_dir=data_dir,
        models=models or None,
        limit=limit,
    )

    draws_dir = output_dir / "draws"
    prov_dir = output_dir / "provenance"
    stan_dir = output_dir / "stan_models"
    data_out_dir = output_dir / "stan_data"
    draws_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    stan_dir.mkdir(parents=True, exist_ok=True)
    data_out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "chains": chains,
        "iter_warmup": iter_warmup,
        "iter_sampling": iter_sampling,
        "thin": thin,
        "seed": seed,
    }
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "cmdstan+local_registry",
        "models_dir": str(models_dir),
        "data_dir": str(data_dir),
        "config": config,
        "cmdstanpy_version": importlib.metadata.version("cmdstanpy"),
        "cmdstan_version": str(cmdstanpy.cmdstan_version()),
        "results": [],
    }

    generated = 0
    for spec in specs:
        zip_path = draws_dir / f"{spec.model}.json.zip"
        prov_path = prov_dir / f"{spec.model}.provenance.json"
        if not overwrite and zip_path.exists() and prov_path.exists():
            manifest["results"].append({"model": spec.model, "status": "skipped_exists"})
            continue

        data_dict = load_stan_data(spec.data_path)
        model = cmdstanpy.CmdStanModel(stan_file=str(spec.stan_path))
        fit = model.sample(
            data=data_dict,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            thin=thin,
            seed=seed,
            show_progress=True,
        )
        chain_draws = [parse_cmdstan_csv(Path(csv_file)) for csv_file in fit.runset.csv_files]
        payload = build_posteriordb_payload(chain_draws)
        write_posteriordb_json_zip(payload, zip_path, model_name=spec.model)

        (stan_dir / spec.stan_path.name).write_text(spec.stan_path.read_text())
        (data_out_dir / spec.data_path.name).write_text(spec.data_path.read_text())

        provenance = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": spec.model,
            "source": "cmdstan+local_registry",
            "config": config,
            "cmdstan_version": manifest["cmdstan_version"],
            "cmdstanpy_version": manifest["cmdstanpy_version"],
            "stan_model": spec.stan_path.name,
            "stan_data": spec.data_path.name,
            "chain_csv_files": [Path(p).name for p in fit.runset.csv_files],
            "n_chains": len(payload),
            "n_draws_per_chain": len(next(iter(payload[0].values()))) if payload else 0,
        }
        write_provenance(prov_path, provenance)
        manifest["results"].append({"model": spec.model, "status": "generated"})
        generated += 1

    write_provenance(output_dir / "generation_manifest.json", manifest)
    click.echo(f"generated {generated}/{len(specs)} models -> {output_dir}")


if __name__ == "__main__":
    main()
