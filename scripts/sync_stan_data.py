from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.command("sync-stan-data")
@click.option(
    "--draws-dir",
    type=click.Path(path_type=Path),
    default=Path("packages/mcmc-ref-data/src/mcmc_ref_data/data/draws"),
    show_default=True,
    help="Directory containing *.draws.parquet files.",
)
@click.option(
    "--target-dir",
    type=click.Path(path_type=Path),
    default=Path("packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_data"),
    show_default=True,
    help="Destination directory for *.data.json files.",
)
@click.option(
    "--posteriordb-path",
    type=click.Path(path_type=Path),
    default=Path.home() / ".posteriordb" / "posterior_database",
    show_default=True,
    help="Path to local posteriordb database root.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing *.data.json files.")
def main(draws_dir: Path, target_dir: Path, posteriordb_path: Path, overwrite: bool) -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from mcmc_ref.model_inventory import reference_models_from_draws, split_informed_models

    try:
        from posteriordb import PosteriorDatabase
    except Exception as exc:
        raise SystemExit(
            "posteriordb is required to sync standard model data files. "
            'Install via: uv add --dev "mcmc-ref[bootstrap]"'
        ) from exc

    draws_dir = Path(draws_dir)
    target_dir = Path(target_dir)
    posteriordb_path = Path(posteriordb_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    models = reference_models_from_draws(draws_dir)
    standard_models, informed_models = split_informed_models(models)
    pdb = PosteriorDatabase(str(posteriordb_path))

    generated = 0
    skipped = 0
    for model in standard_models:
        out = target_dir / f"{model}.data.json"
        if out.exists() and not overwrite:
            skipped += 1
            continue
        posterior = pdb.posterior(model)
        data = posterior.data.values()
        out.write_text(json.dumps(data, indent=2, sort_keys=True))
        generated += 1

    click.echo(
        f"models: total={len(models)} standard={len(standard_models)} "
        f"informed={len(informed_models)}"
    )
    click.echo(f"synced stan data -> {target_dir}")
    click.echo(f"generated={generated} skipped={skipped}")
    if informed_models:
        click.echo(
            "note: informed models were not auto-exported here; "
            "they may require pre-standardized data."
        )


if __name__ == "__main__":
    main()
