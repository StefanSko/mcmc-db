"""Convert reference draws from JSON-zip or CSV to Parquet + metadata."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from . import diagnostics


@dataclass(frozen=True)
class ConvertResult:
    draws_path: Path
    meta_path: Path
    meta: dict


def convert_file(
    input_path: Path,
    name: str,
    out_draws_dir: Path,
    out_meta_dir: Path,
    force: bool = False,
) -> ConvertResult:
    input_path = Path(input_path)
    out_draws_dir = Path(out_draws_dir)
    out_meta_dir = Path(out_meta_dir)

    table = _read_input(input_path)
    table = _ensure_chain_draw(table)

    params = [c for c in table.column_names if c not in {"chain", "draw"}]
    n_chains, n_draws = _count_chains_draws(table)

    diag = _compute_diagnostics(table, params, min_chains=1 if force else 4)
    checks = _checks(n_chains, n_draws, diag)

    if not force:
        _enforce_checks(checks)

    meta = {
        "model": name,
        "parameters": params,
        "n_chains": n_chains,
        "n_draws_per_chain": n_draws,
        "diagnostics": diag,
        "generated_date": date.today().isoformat(),
        "checks": checks,
        "source": "converted",
    }

    draws_path = out_draws_dir / f"{name}.draws.parquet"
    meta_path = out_meta_dir / f"{name}.meta.json"

    pq.write_table(table, draws_path)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    return ConvertResult(draws_path=draws_path, meta_path=meta_path, meta=meta)


def _read_input(path: Path) -> pa.Table:
    if path.suffix == ".csv":
        return pacsv.read_csv(path)
    if path.suffixes[-2:] == [".json", ".zip"]:
        return _read_json_zip(path)
    raise ValueError(f"Unsupported input format: {path}")


def _read_json_zip(path: Path) -> pa.Table:
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        payload = json.loads(zf.read(name))

    if not isinstance(payload, list) or not payload:
        raise ValueError("json-zip payload must be a non-empty list of chains")

    # Chain-list JSON-zip format: list[chain], each chain is dict[param] -> list[draws]
    chains = payload
    params = sorted(chains[0].keys())
    n_draws = len(next(iter(chains[0].values())))

    columns: dict[str, list] = {"chain": [], "draw": []}
    for param in params:
        columns[param] = []

    for chain_idx, chain in enumerate(chains):
        for draw_idx in range(n_draws):
            columns["chain"].append(chain_idx)
            columns["draw"].append(draw_idx)
            for param in params:
                columns[param].append(chain[param][draw_idx])

    return pa.table(columns)


def _ensure_chain_draw(table: pa.Table) -> pa.Table:
    cols = set(table.column_names)
    if "chain" in cols and "draw" in cols:
        return table

    if "chain" in cols and "draw" not in cols:
        draw = pa.array(list(range(table.num_rows)), type=pa.int32())
        return table.append_column("draw", draw)

    if "draw" in cols and "chain" not in cols:
        chain = pa.array([0] * table.num_rows, type=pa.int32())
        return table.append_column("chain", chain)

    chain = pa.array([0] * table.num_rows, type=pa.int32())
    draw = pa.array(list(range(table.num_rows)), type=pa.int32())
    return table.append_column("chain", chain).append_column("draw", draw)


def _count_chains_draws(table: pa.Table) -> tuple[int, int]:
    chain_values = table.column("chain").to_pylist()
    chains = set(chain_values)
    draws_per_chain: dict[int, int] = {int(c): 0 for c in chains}
    for chain in chain_values:
        draws_per_chain[int(chain)] += 1
    n_chains = len(chains)
    n_draws = min(draws_per_chain.values()) if draws_per_chain else 0
    return n_chains, n_draws


def _compute_diagnostics(
    table: pa.Table,
    params: Iterable[str],
    *,
    min_chains: int = 4,
) -> dict[str, dict[str, float]]:
    diag: dict[str, dict[str, float]] = {}
    for param in params:
        chains = _chains_from_table(table, param)
        rhat = diagnostics.split_rhat(chains, min_chains=min_chains)
        ess_b = diagnostics.ess_bulk(chains, min_chains=min_chains)
        ess_t = diagnostics.ess_tail(chains, min_chains=min_chains)
        diag[param] = {"rhat": rhat, "ess_bulk": ess_b, "ess_tail": ess_t}
    return diag


def _chains_from_table(table: pa.Table, param: str) -> list[list[float]]:
    chain_col = table.column("chain").to_pylist()
    draw_col = table.column("draw").to_pylist()
    param_col = table.column(param).to_pylist()
    buckets: dict[int, list[tuple[int, float]]] = {}
    for chain, draw, val in zip(chain_col, draw_col, param_col, strict=False):
        buckets.setdefault(int(chain), []).append((int(draw), float(val)))
    chains: list[list[float]] = []
    for chain in sorted(buckets):
        ordered = sorted(buckets[chain], key=lambda x: x[0])
        chains.append([v for _, v in ordered])
    return chains


def _checks(n_chains: int, n_draws: int, diag: dict[str, dict[str, float]]) -> dict[str, bool]:
    ess_ok = all(values.get("ess_bulk", 0.0) > 400 for values in diag.values())
    rhat_ok = all(values.get("rhat", 1.0) < 1.01 for values in diag.values())
    return {
        "ndraws_is_10k": n_chains * n_draws == 10_000,
        "nchains_is_gte_4": n_chains >= 4,
        "ess_above_400": ess_ok,
        "rhat_below_1_01": rhat_ok,
    }


def _enforce_checks(checks: dict[str, bool]) -> None:
    failures = [name for name, ok in checks.items() if not ok]
    if failures:
        raise ValueError(f"quality checks failed: {', '.join(failures)}")
