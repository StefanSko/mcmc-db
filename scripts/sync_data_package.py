from __future__ import annotations

import shutil
from pathlib import Path

import click


@click.command("sync-data-package")
@click.option(
    "--source-root",
    type=click.Path(path_type=Path),
    default=Path.home() / ".mcmc-ref",
    show_default=True,
    help="Reference corpus root containing draws/ and meta/.",
)
@click.option(
    "--target-root",
    type=click.Path(path_type=Path),
    default=Path("packages/mcmc-ref-data/src/mcmc_ref_data/data"),
    show_default=True,
    help="Destination package data root.",
)
def main(source_root: Path, target_root: Path) -> None:
    source_root = Path(source_root)
    target_root = Path(target_root)
    src_draws = source_root / "draws"
    src_meta = source_root / "meta"
    dst_draws = target_root / "draws"
    dst_meta = target_root / "meta"

    if not src_draws.exists() or not src_meta.exists():
        raise SystemExit(f"source root missing draws/meta: {source_root}")

    if dst_draws.exists():
        shutil.rmtree(dst_draws)
    if dst_meta.exists():
        shutil.rmtree(dst_meta)
    dst_draws.mkdir(parents=True, exist_ok=True)
    dst_meta.mkdir(parents=True, exist_ok=True)

    for path in sorted(src_draws.glob("*.draws.parquet")):
        shutil.copy2(path, dst_draws / path.name)
    for path in sorted(src_meta.glob("*.meta.json")):
        shutil.copy2(path, dst_meta / path.name)

    click.echo(f"synced draws -> {dst_draws}")
    click.echo(f"synced meta  -> {dst_meta}")
    click.echo(
        f"counts: draws={len(list(dst_draws.glob('*.draws.parquet')))} "
        f"meta={len(list(dst_meta.glob('*.meta.json')))}"
    )


if __name__ == "__main__":
    main()
