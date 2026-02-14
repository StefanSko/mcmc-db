"""Generate reference draws using CmdStan via cmdstanpy."""

from __future__ import annotations

import tempfile
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.csv as pacsv

from . import convert, reference
from .store import DataStore


def generate_reference_draws(
    model_name: str,
    stan_code: str,
    stan_data: dict,
    output_root: Path,
    *,
    chains: int = 10,
    iter_warmup: int = 10_000,
    iter_sampling: int = 10_000,
    thin: int = 10,
    seed: int = 4711,
) -> Path:
    output_root = Path(output_root)
    draws_dir = output_root / "draws"
    meta_dir = output_root / "meta"
    draws_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    fit = _compile_and_sample(
        stan_code=stan_code,
        stan_data=stan_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        thin=thin,
        seed=seed,
    )

    # Extract draws via cmdstanpy's draws_pd() for reliable parsing of CmdStan CSV format
    combined = _fit_to_table(fit)

    # Write combined CSV, then use convert to produce parquet + meta
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / f"{model_name}.csv"
        pacsv.write_csv(combined, csv_path)

        convert.convert_file(
            input_path=csv_path,
            name=model_name,
            out_draws_dir=draws_dir,
            out_meta_dir=meta_dir,
            force=True,
        )

    return draws_dir / f"{model_name}.draws.parquet"


_STAN_INTERNAL_COLS = {
    "chain__",
    "iter__",
    "draw__",
    "lp__",
    "accept_stat__",
    "stepsize__",
    "treedepth__",
    "n_leapfrog__",
    "divergent__",
    "energy__",
}


def _fit_to_table(fit) -> pa.Table:
    """Extract draws from a CmdStanMCMC fit object into an Arrow table.

    Tries draws_pd() first (real cmdstanpy). Falls back to reading CSV files
    directly, which also supports mocked fits in tests.
    """
    if callable(getattr(fit, "draws_pd", None)):
        try:
            df = fit.draws_pd()
            table = pa.Table.from_pandas(df)
            return _clean_cmdstan_table(table)
        except (AttributeError, ImportError):
            pass

    # Fallback: read CSV files directly
    all_tables = []
    for chain_idx, csv_path in enumerate(fit.runset.csv_files):
        chain_table = pacsv.read_csv(csv_path)
        n_rows = chain_table.num_rows
        chain_col = pa.array([chain_idx] * n_rows, type=pa.int32())
        draw_col = pa.array(list(range(n_rows)), type=pa.int32())
        chain_table = chain_table.append_column("chain", chain_col)
        chain_table = chain_table.append_column("draw", draw_col)
        all_tables.append(chain_table)
    return pa.concat_tables(all_tables)


def _clean_cmdstan_table(table: pa.Table) -> pa.Table:
    """Convert cmdstanpy output table to the format expected by convert."""
    # Extract chain IDs from chain__ column (1-indexed in cmdstanpy -> 0-indexed)
    chain_ids = table.column("chain__").to_pylist()
    unique_chains = sorted(set(chain_ids))
    chain_map = {c: i for i, c in enumerate(unique_chains)}

    # Build chain and draw columns
    chain_list: list[int] = []
    draw_list: list[int] = []
    draw_counters: dict[int, int] = {i: 0 for i in range(len(unique_chains))}
    for c in chain_ids:
        mapped = chain_map[c]
        chain_list.append(mapped)
        draw_list.append(draw_counters[mapped])
        draw_counters[mapped] += 1

    # Keep only model parameters (drop Stan internal columns)
    param_cols = [c for c in table.column_names if c not in _STAN_INTERNAL_COLS]

    columns: dict[str, pa.Array] = {
        "chain": pa.array(chain_list, type=pa.int32()),
        "draw": pa.array(draw_list, type=pa.int32()),
    }
    for col_name in param_cols:
        columns[col_name] = table.column(col_name)

    return pa.table(columns)


def _compile_and_sample(
    stan_code: str,
    stan_data: dict,
    chains: int,
    iter_warmup: int,
    iter_sampling: int,
    thin: int,
    seed: int,
):
    import cmdstanpy  # type: ignore[import-untyped]

    # Use a persistent temp dir so the compiled binary survives through sampling
    tmpdir = tempfile.mkdtemp()
    stan_file = Path(tmpdir) / "model.stan"
    stan_file.write_text(stan_code)
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))

    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        thin=thin,
        seed=seed,
    )
    return fit


@click.command("generate")
@click.argument("model")
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=Path.home() / ".mcmc-ref",
    show_default=True,
)
@click.option("--seed", default=4711, type=int)
def main(model: str, output_root: Path, seed: int) -> None:
    """Generate reference draws for a model using CmdStan."""
    store = DataStore()
    code = reference.model_code(model, store=store)
    data = reference.stan_data(model, store=store)

    click.echo(f"Generating reference draws for {model}...")
    result = generate_reference_draws(
        model_name=model,
        stan_code=code,
        stan_data=data,
        output_root=output_root,
        seed=seed,
    )
    click.echo(f"Done: {result}")


if __name__ == "__main__":
    main()
