from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click


@click.command("sync-stan-models")
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
    default=Path("packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_models"),
    show_default=True,
    help="Destination directory for *.stan files.",
)
@click.option(
    "--posteriordb-path",
    type=click.Path(path_type=Path),
    default=Path.home() / ".posteriordb" / "posterior_database",
    show_default=True,
    help="Path to local posteriordb database root.",
)
@click.option(
    "--informed-stan-dir",
    type=click.Path(path_type=Path),
    default=Path("generated_references/informed/stan_models"),
    show_default=True,
    help="Directory containing *_informed.stan files.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing *.stan files.")
def main(
    draws_dir: Path,
    target_dir: Path,
    posteriordb_path: Path,
    informed_stan_dir: Path,
    overwrite: bool,
) -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from mcmc_ref.model_inventory import reference_models_from_draws, split_informed_models

    draws_dir = Path(draws_dir)
    target_dir = Path(target_dir)
    posteriordb_path = Path(posteriordb_path)
    informed_stan_dir = Path(informed_stan_dir)

    models = reference_models_from_draws(draws_dir)
    standard_models, informed_models = split_informed_models(models)
    target_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    copied = 0
    skipped = 0
    missing_informed: list[str] = []

    if standard_models:
        try:
            from posteriordb import PosteriorDatabase
        except Exception as exc:
            raise SystemExit(
                "posteriordb is required to sync standard model .stan files. "
                'Install via: uv add --dev "mcmc-ref[bootstrap]"'
            ) from exc
        pdb = PosteriorDatabase(str(posteriordb_path))
        for model in standard_models:
            out_path = target_dir / f"{model}.stan"
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            posterior = pdb.posterior(model)
            code = posterior.model.stan_code()
            out_path.write_text(str(code))
            generated += 1

    for model in informed_models:
        src = informed_stan_dir / f"{model}.stan"
        out_path = target_dir / f"{model}.stan"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        if not src.exists():
            missing_informed.append(model)
            continue
        shutil.copy2(src, out_path)
        copied += 1

    click.echo(
        f"models: total={len(models)} standard={len(standard_models)} "
        f"informed={len(informed_models)}"
    )
    click.echo(f"synced stan models -> {target_dir}")
    click.echo(f"generated_from_posteriordb={generated} copied_informed={copied} skipped={skipped}")
    if missing_informed:
        click.echo("missing informed stan files:")
        for model in missing_informed:
            click.echo(f"- {model}")


if __name__ == "__main__":
    main()
