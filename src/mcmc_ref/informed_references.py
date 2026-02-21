"""Import informed-prior reference draws into an mcmc-ref corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click

from . import convert


@dataclass(frozen=True)
class ImportFailure:
    model: str
    input_path: Path
    error: str


@dataclass(frozen=True)
class ImportResult:
    total: int
    converted: int
    failures: list[ImportFailure]
    output_root: Path


def import_informed_references(
    source_dir: Path,
    output_root: Path,
    *,
    force: bool = False,
) -> ImportResult:
    source_dir = Path(source_dir)
    output_root = Path(output_root)
    draws_dir = output_root / "draws"
    meta_dir = output_root / "meta"
    draws_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(source_dir.glob("*_informed.json.zip"))
    failures: list[ImportFailure] = []
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
            _annotate_meta(model=model, source_dir=source_dir, meta_dir=meta_dir)
            converted += 1
        except Exception as exc:
            failures.append(ImportFailure(model=model, input_path=input_path, error=str(exc)))

    return ImportResult(
        total=len(files),
        converted=converted,
        failures=failures,
        output_root=output_root,
    )


def _annotate_meta(model: str, source_dir: Path, meta_dir: Path) -> None:
    meta_path = meta_dir / f"{model}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["reference_variant"] = "informed_prior"

    info_path = source_dir / f"{model}.info.json"
    if info_path.exists():
        meta["informed_reference_info"] = json.loads(info_path.read_text())

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


@click.command("import-informed-references", deprecated=True)
@click.option(
    "--source-dir",
    type=click.Path(path_type=Path),
    default=Path.home() / ".mcmc-ref" / "legacy" / "informed_archives",
    show_default=True,
    help="Directory containing *_informed.json.zip and optional *.info.json files.",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=Path.home() / ".mcmc-ref",
    show_default=True,
    help="Reference corpus root with draws/ and meta/ subdirectories.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Allow conversion when quality checks fail.",
)
def main(source_dir: Path, output_root: Path, force: bool) -> None:
    """DEPRECATED: Import informed draws; generate informed references via provenance pipeline."""
    if not source_dir.exists():
        raise SystemExit(f"source directory not found: {source_dir}")
    result = import_informed_references(source_dir=source_dir, output_root=output_root, force=force)
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
