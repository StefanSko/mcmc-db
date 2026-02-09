"""Build a local reference corpus from posteriordb json.zip draw files."""

from __future__ import annotations

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
