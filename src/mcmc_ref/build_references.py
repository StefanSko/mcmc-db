"""Build a local reference corpus from posteriordb json.zip draw files."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import click

from . import convert

DEFAULT_POSTERIORDB_DRAWS = (
    Path.home() / ".posteriordb" / "posterior_database" / "reference_posteriors" / "draws" / "draws"
)


@dataclass(frozen=True)
class BuildFailure:
    model: str
    input_path: Path
    error: str


@dataclass(frozen=True)
class BuildResult:
    total: int
    converted: int
    failures: list[BuildFailure]
    output_root: Path


def build_references(
    source_dir: Path,
    output_root: Path,
    *,
    models: Sequence[str] | None = None,
    force: bool = False,
) -> BuildResult:
    source_dir = Path(source_dir)
    output_root = Path(output_root)
    draws_dir = output_root / "draws"
    meta_dir = output_root / "meta"
    draws_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(source_dir.glob("*.json.zip"))
    if models is not None:
        requested = set(models)
        files = [path for path in files if path.stem.replace(".json", "") in requested]

    failures: list[BuildFailure] = []
    converted = 0
    for input_path in files:
        model = input_path.stem.replace(".json", "")
        try:
            convert.convert_file(
                input_path=input_path,
                name=model,
                out_draws_dir=draws_dir,
                out_meta_dir=meta_dir,
                force=force,
            )
            converted += 1
        except Exception as exc:
            failures.append(BuildFailure(model=model, input_path=input_path, error=str(exc)))

    return BuildResult(
        total=len(files),
        converted=converted,
        failures=failures,
        output_root=output_root,
    )


DEFAULT_POSTERIORDB_ROOT = Path.home() / ".posteriordb" / "posterior_database"


@dataclass(frozen=True)
class AssetResult:
    total: int
    extracted: int
    failures: list[BuildFailure]
    output_root: Path


def extract_stan_assets(
    posteriordb_root: Path,
    output_root: Path,
    posteriors: Sequence[str],
) -> AssetResult:
    posteriordb_root = Path(posteriordb_root)
    output_root = Path(output_root)
    stan_data_dir = output_root / "stan_data"
    stan_code_dir = output_root / "stan_code"
    stan_data_dir.mkdir(parents=True, exist_ok=True)
    stan_code_dir.mkdir(parents=True, exist_ok=True)

    failures: list[BuildFailure] = []
    extracted = 0

    for posterior_name in posteriors:
        try:
            _extract_single_posterior(
                posteriordb_root, posterior_name, stan_data_dir, stan_code_dir
            )
            extracted += 1
        except Exception as exc:
            failures.append(
                BuildFailure(
                    model=posterior_name,
                    input_path=posteriordb_root / "posteriors" / f"{posterior_name}.json",
                    error=str(exc),
                )
            )

    return AssetResult(
        total=len(posteriors),
        extracted=extracted,
        failures=failures,
        output_root=output_root,
    )


def _extract_single_posterior(
    posteriordb_root: Path,
    posterior_name: str,
    stan_data_dir: Path,
    stan_code_dir: Path,
) -> None:
    # Read posterior definition
    posterior_path = posteriordb_root / "posteriors" / f"{posterior_name}.json"
    posterior_def = json.loads(posterior_path.read_text())
    data_name = posterior_def["data_name"]
    model_name = posterior_def["model_name"]

    # Extract data from json.zip
    data_zip_path = posteriordb_root / "data" / "data" / f"{data_name}.json.zip"
    with zipfile.ZipFile(data_zip_path) as zf:
        entry_name = zf.namelist()[0]
        data_content = zf.read(entry_name)
    out_data_path = stan_data_dir / f"{posterior_name}.data.json"
    out_data_path.write_bytes(data_content)

    # Copy Stan model code
    stan_path = posteriordb_root / "models" / "stan" / f"{model_name}.stan"
    out_code_path = stan_code_dir / f"{posterior_name}.stan"
    out_code_path.write_text(stan_path.read_text())


@click.command("build-references")
@click.option(
    "--source-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_POSTERIORDB_DRAWS,
    show_default=True,
    help="Directory containing posteriordb *.json.zip draw files.",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=Path.home() / ".mcmc-ref",
    show_default=True,
    help="Reference corpus root with draws/ and meta/ subdirectories.",
)
@click.option(
    "--models",
    default=None,
    help="Optional comma-separated model list (e.g. eight_schools-eight_schools_noncentered).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Allow conversion when quality checks fail (not recommended for canonical benchmarks).",
)
def main(source_dir: Path, output_root: Path, models: str | None, force: bool) -> None:
    """Convert all posteriordb draw archives to a local mcmc-ref corpus."""
    if not source_dir.exists():
        raise SystemExit(f"source directory not found: {source_dir}")
    model_list = models.split(",") if models else None
    result = build_references(
        source_dir=source_dir,
        output_root=output_root,
        models=model_list,
        force=force,
    )

    click.echo(f"source: {source_dir}")
    click.echo(f"output: {result.output_root}")
    click.echo(f"converted: {result.converted}/{result.total}")
    if result.failures:
        click.echo("failures:")
        for failure in result.failures:
            click.echo(f"- {failure.model}: {failure.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
