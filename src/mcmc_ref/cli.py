"""CLI entry point for mcmc-ref."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from . import convert as convert_mod
from . import generate as generate_mod
from . import pairs as pairs_mod
from . import provenance, reference
from .store import DataStore


@click.group()
def main() -> None:
    """mcmc-ref CLI."""


@main.command("list")
@click.option(
    "--format",
    "format_",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
)
def list_cmd(format_: str) -> None:
    models = reference.list_models()
    if format_ == "json":
        click.echo(json.dumps(models, indent=2))
        return
    for model in models:
        click.echo(model)


@main.command("data")
@click.argument("model")
def data_cmd(model: str) -> None:
    data = reference.stan_data(model)
    click.echo(json.dumps(data, indent=2))


@main.command("model-code")
@click.argument("model")
def model_code_cmd(model: str) -> None:
    code = reference.model_code(model)
    click.echo(code)


@main.command("stats")
@click.argument("model")
@click.option("--params", default=None, help="Comma-separated parameter list")
@click.option(
    "--format",
    "format_",
    type=click.Choice(["table", "csv", "json"], case_sensitive=False),
    default="table",
)
@click.option(
    "--backend", type=click.Choice(["arrow", "numpy"], case_sensitive=False), default="arrow"
)
@click.option(
    "--quantile-mode",
    type=click.Choice(["exact"], case_sensitive=False),
    default="exact",
)
@click.option("--include-diagnostics", is_flag=True, help="Include rhat/ess metrics")
def stats_cmd(
    model: str,
    params: str | None,
    format_: str,
    backend: str,
    quantile_mode: str,
    include_diagnostics: bool,
) -> None:
    param_list = params.split(",") if params else None
    stats = reference.stats(model, params=param_list, backend=backend, quantile_mode=quantile_mode)
    if include_diagnostics:
        diag = reference.diagnostics_for_model(model, params=param_list)
        for param, metrics in diag.items():
            stats.setdefault(param, {}).update(metrics)
    if format_ == "json":
        click.echo(json.dumps(stats, indent=2, sort_keys=True))
        return
    if format_ == "csv":
        headers = ["param"] + _metric_headers(stats)
        click.echo(",".join(headers))
        for param, metrics in stats.items():
            row = [param] + [str(metrics.get(h, "")) for h in headers[1:]]
            click.echo(",".join(row))
        return
    _print_table(stats)


@main.command("draws")
@click.argument("model")
@click.option("--params", default=None, help="Comma-separated parameter list")
@click.option("--chains", default=None, help="Comma-separated chain indices")
@click.option(
    "--format",
    "format_",
    type=click.Choice(["csv", "parquet"], case_sensitive=False),
    default="csv",
)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def draws_cmd(
    model: str, params: str | None, chains: str | None, format_: str, output: Path | None
) -> None:
    param_list = params.split(",") if params else None
    chain_list = [int(c) for c in chains.split(",")] if chains else None
    draws = reference.draws(model, params=param_list, chains=chain_list, return_="arrow")

    if format_ == "csv":
        out = sys.stdout.buffer if output is None else output.open("wb")
        pacsv.write_csv(draws, out)
        return

    if format_ == "parquet":
        table = draws.read_all() if hasattr(draws, "read_all") else draws
        dest = sys.stdout.buffer if output is None else output
        pq.write_table(table, dest)
        return


@main.command("diagnostics")
@click.argument("model")
@click.option(
    "--format",
    "format_",
    type=click.Choice(["table", "csv", "json"], case_sensitive=False),
    default="table",
)
def diagnostics_cmd(model: str, format_: str) -> None:
    diag = reference.diagnostics_for_model(model)
    if format_ == "json":
        click.echo(json.dumps(diag, indent=2, sort_keys=True))
        return
    if format_ == "csv":
        headers = ["param", "rhat", "ess_bulk", "ess_tail"]
        click.echo(",".join(headers))
        for param, metrics in diag.items():
            row = [
                param,
                str(metrics.get("rhat")),
                str(metrics.get("ess_bulk")),
                str(metrics.get("ess_tail")),
            ]
            click.echo(",".join(row))
        return
    _print_table(diag)


@main.command("info")
@click.argument("model")
def info_cmd(model: str) -> None:
    store = DataStore()
    meta = store.read_meta(model)
    click.echo(json.dumps(meta, indent=2, sort_keys=True))


@main.command("compare")
@click.argument("model")
@click.option("--actual", "actual_path", type=click.Path(path_type=Path), required=True)
@click.option("--tolerance", default=0.15, type=float)
@click.option(
    "--format",
    "format_",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
)
def compare_cmd(model: str, actual_path: Path, tolerance: float, format_: str) -> None:
    actual = _read_actual_csv(actual_path)
    result = reference.compare(model, actual=actual, tolerance=tolerance)
    if format_ == "json":
        click.echo(json.dumps(_compare_to_json(result), indent=2, sort_keys=True))
    else:
        _print_compare(result)
    raise SystemExit(0 if result.passed else 2)


@main.command("convert")
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option("--name", required=True)
@click.option("--force", is_flag=True)
def convert_cmd(input_path: Path, name: str, force: bool) -> None:
    local_root = Path.home() / ".mcmc-ref"
    draws_dir = local_root / "draws"
    meta_dir = local_root / "meta"
    draws_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    convert_mod.convert_file(
        input_path, name=name, out_draws_dir=draws_dir, out_meta_dir=meta_dir, force=force
    )
    click.echo(f"converted {name} -> {draws_dir}")


@main.command("pairs")
@click.option(
    "--format",
    "format_",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
)
def pairs_cmd(format_: str) -> None:
    """List all reparametrization pairs."""
    names = pairs_mod.list_pairs()
    if format_ == "json":
        click.echo(json.dumps(names, indent=2))
        return
    for name in names:
        click.echo(name)


@main.command("pair")
@click.argument("name")
def pair_cmd(name: str) -> None:
    """Show info about a reparametrization pair."""
    try:
        p = pairs_mod.pair(name)
    except FileNotFoundError as exc:
        click.echo(f"pair not found: {name}", err=True)
        raise SystemExit(1) from exc
    info = {
        "name": p.name,
        "description": p.description,
        "bad_variant": p.bad_variant,
        "good_variant": p.good_variant,
        "reference_model": p.reference_model,
        "expected_pathologies": p.expected_pathologies,
        "difficulty": p.difficulty,
    }
    click.echo(json.dumps(info, indent=2))


@main.command("provenance-scaffold")
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
def provenance_scaffold_cmd(output_root: Path) -> None:
    manifest_path = provenance.materialize_scaffold(output_root)
    click.echo(f"wrote {manifest_path}")


@main.command("provenance-generate")
@click.option("--scaffold-root", type=click.Path(path_type=Path), required=True)
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
@click.option("--models", default=None, help="Optional comma-separated recipe names.")
@click.option("--force", is_flag=True, help="Forward --force to convert quality checks.")
@click.option("--fake-runner", is_flag=True, help="Use deterministic fake runner (testing only).")
def provenance_generate_cmd(
    scaffold_root: Path,
    output_root: Path,
    models: str | None,
    force: bool,
    fake_runner: bool,
) -> None:
    model_list = models.split(",") if models else None
    runner = generate_mod.fake_jsonzip_runner if fake_runner else None
    result = generate_mod.generate_reference_corpus(
        scaffold_root=scaffold_root,
        output_root=output_root,
        models=model_list,
        force=force,
        runner=runner,
    )
    click.echo(f"generated={result.generated} failed={result.failed} output={result.output_root}")
    if result.errors:
        for name, message in sorted(result.errors.items()):
            click.echo(f"- {name}: {message}")
        raise SystemExit(1)


@main.command("provenance-publish")
@click.option("--source-root", type=click.Path(path_type=Path), required=True)
@click.option("--scaffold-root", type=click.Path(path_type=Path), required=True)
@click.option("--package-root", type=click.Path(path_type=Path), required=True)
def provenance_publish_cmd(source_root: Path, scaffold_root: Path, package_root: Path) -> None:
    result = generate_mod.publish_reference_data(
        source_root=source_root,
        scaffold_root=scaffold_root,
        package_root=package_root,
    )
    click.echo(
        "published "
        f"draws={result.draws_copied} meta={result.meta_copied} pairs={result.pairs_copied} "
        f"to={result.package_root}"
    )


def _read_actual_csv(path: Path) -> dict[str, list[float]]:
    table = pacsv.read_csv(path)
    params = [c for c in table.column_names if c not in {"chain", "draw"}]
    return {p: [float(v) for v in table.column(p).to_pylist()] for p in params}


def _metric_headers(stats: dict[str, dict[str, float]]) -> list[str]:
    keys: set[str] = set()
    for metrics in stats.values():
        keys.update(metrics.keys())
    return sorted(keys)


def _print_table(stats: dict[str, dict[str, float]]) -> None:
    headers = ["param"] + _metric_headers(stats)
    widths = [max(len(h), 6) for h in headers]
    line = " ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False))
    click.echo(line)
    for param, metrics in stats.items():
        row = [param] + [f"{metrics.get(h, float('nan')):.6g}" for h in headers[1:]]
        click.echo(" ".join(val.ljust(w) for val, w in zip(row, widths, strict=False)))


def _compare_to_json(result) -> dict:
    details = {}
    for param, metrics in result.details.items():
        details[param] = {k: vars(v) for k, v in metrics.items()}
    return {"passed": result.passed, "failures": result.failures, "details": details}


def _print_compare(result) -> None:
    click.echo("passed" if result.passed else "failed")
    for failure in result.failures:
        click.echo(f"- {failure}")
