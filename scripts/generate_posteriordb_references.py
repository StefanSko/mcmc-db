from __future__ import annotations

import importlib.metadata
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click


def _require_generation_deps():
    try:
        import cmdstanpy
        from posteriordb import PosteriorDatabase
    except Exception as exc:
        raise SystemExit(
            "Missing generation dependencies. Run: "
            'uv add --dev "mcmc-ref[generate,bootstrap]"'
        ) from exc
    return cmdstanpy, PosteriorDatabase


@click.command("generate-posteriordb-references")
@click.option(
    "--posteriordb-path",
    type=click.Path(path_type=Path),
    default=Path.home() / ".posteriordb" / "posterior_database",
    show_default=True,
    help="Path to local posteriordb checkout root.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("generated_references/posteriordb"),
    show_default=True,
    help="Output directory for draws/provenance files.",
)
@click.option(
    "--posterior",
    "posteriors",
    multiple=True,
    help="Posterior name to generate (repeat option for multiple).",
)
@click.option("--limit", default=None, type=int, help="Only generate first N posteriors.")
@click.option("--chains", default=10, show_default=True, type=int)
@click.option("--iter-warmup", default=10_000, show_default=True, type=int)
@click.option("--iter-sampling", default=10_000, show_default=True, type=int)
@click.option("--thin", default=10, show_default=True, type=int)
@click.option("--seed", default=4711, show_default=True, type=int)
@click.option("--overwrite", is_flag=True, help="Overwrite existing json.zip/provenance files.")
def main(
    posteriordb_path: Path,
    output_dir: Path,
    posteriors: tuple[str, ...],
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

    """Generate posteriordb-style json.zip references directly with CmdStan."""
    from mcmc_ref.cmdstan_generate import (
        build_posteriordb_payload,
        parse_cmdstan_csv,
        write_posteriordb_json_zip,
        write_provenance,
    )

    cmdstanpy, PosteriorDatabase = _require_generation_deps()
    pdb = PosteriorDatabase(str(posteriordb_path))
    requested = list(posteriors) if posteriors else list(pdb.posterior_names())
    if limit is not None:
        requested = requested[:limit]

    draws_dir = output_dir / "draws"
    prov_dir = output_dir / "provenance"
    stan_dir = output_dir / "stan_models"
    draws_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    stan_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "posteriordb_path": str(posteriordb_path),
        "config": {
            "chains": chains,
            "iter_warmup": iter_warmup,
            "iter_sampling": iter_sampling,
            "thin": thin,
            "seed": seed,
        },
        "cmdstanpy_version": importlib.metadata.version("cmdstanpy"),
        "posteriordb_version": importlib.metadata.version("posteriordb"),
        "cmdstan_version": str(cmdstanpy.cmdstan_version()),
        "results": [],
    }

    converted = 0
    for posterior_name in requested:
        model_name = posterior_name
        zip_path = draws_dir / f"{model_name}.json.zip"
        prov_path = prov_dir / f"{model_name}.provenance.json"
        if not overwrite and zip_path.exists() and prov_path.exists():
            manifest["results"].append({"posterior": posterior_name, "status": "skipped_exists"})
            continue

        posterior = pdb.posterior(posterior_name)
        stan_code = posterior.model.stan_code()
        data_dict = posterior.data.values()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stan_path = tmp / f"{model_name}.stan"
            stan_path.write_text(stan_code)
            (stan_dir / f"{model_name}.stan").write_text(stan_code)
            model = cmdstanpy.CmdStanModel(stan_file=str(stan_path))
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
        write_posteriordb_json_zip(payload, zip_path, model_name=model_name)

        provenance = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "posterior": posterior_name,
            "source": "posteriordb+cmdstan",
            "config": manifest["config"],
            "cmdstan_version": manifest["cmdstan_version"],
            "cmdstanpy_version": manifest["cmdstanpy_version"],
            "posteriordb_version": manifest["posteriordb_version"],
            "chain_csv_files": [Path(p).name for p in fit.runset.csv_files],
            "n_chains": len(payload),
            "n_draws_per_chain": len(next(iter(payload[0].values()))) if payload else 0,
        }
        write_provenance(prov_path, provenance)
        manifest["results"].append({"posterior": posterior_name, "status": "generated"})
        converted += 1

    write_provenance(output_dir / "generation_manifest.json", manifest)
    click.echo(f"generated {converted}/{len(requested)} posteriors -> {output_dir}")


if __name__ == "__main__":
    main()
