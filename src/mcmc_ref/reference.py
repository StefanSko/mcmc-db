"""Python API for reference draws."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from . import diagnostics
from .backends import get_backend
from .compare import compare_stats, compute_stats_from_draws
from .convert import _chains_from_table
from .draws import Draws, coerce_return
from .store import DataStore


def list_models(store: DataStore | None = None) -> list[str]:
    store = store or DataStore()
    return store.list_models()


def stan_data(model: str, store: DataStore | None = None) -> dict:
    store = store or DataStore()
    return store.read_stan_data(model)


def model_code(model: str, store: DataStore | None = None) -> str:
    store = store or DataStore()
    return store.read_stan_code(model)


def stats(
    model: str,
    params: Sequence[str] | None = None,
    backend: str = "arrow",
    quantile_mode: str = "exact",
    store: DataStore | None = None,
) -> dict[str, dict[str, float]]:
    """Compute summary statistics for a model.

    Example:
        stats = reference.stats("eight_schools", params=["mu", "tau"])
    """
    store = store or DataStore()
    reader = store.open_draws(model, params=params)
    table = reader.read_all()
    if params is None:
        params = [c for c in table.column_names if c not in {"chain", "draw"}]
    backend_impl = get_backend(backend)
    return backend_impl.stats(table, params, quantile_mode=quantile_mode)


def draws(
    model: str,
    params: Sequence[str] | None = None,
    chains: Sequence[int] | None = None,
    return_: str = "arrow",
    store: DataStore | None = None,
):
    """Return draws for a model.

    return_:
      - "arrow": pyarrow Table or RecordBatchReader
      - "draws": Draws wrapper with conversion helpers
      - "numpy": NumPy array (if installed)
      - "list": list of row dicts
    """
    store = store or DataStore()
    reader = store.open_draws(model, params=params, chains=chains)
    if params is None:
        table = reader.read_all()
        params = [c for c in table.column_names if c not in {"chain", "draw"}]
        reader = table
    draws_obj = Draws(data=reader, params=list(params), chains=list(chains) if chains else None)
    return coerce_return(draws_obj, return_)


def diagnostics_for_model(
    model: str,
    params: Sequence[str] | None = None,
    store: DataStore | None = None,
) -> dict[str, dict[str, float]]:
    store = store or DataStore()
    try:
        meta = store.read_meta(model)
    except FileNotFoundError:
        meta = {}
    diag = meta.get("diagnostics")
    if isinstance(diag, dict) and diag:
        if params is None:
            return diag
        return {p: diag[p] for p in params if p in diag}

    reader = store.open_draws(model, params=params)
    table = reader.read_all()
    if params is None:
        params = [c for c in table.column_names if c not in {"chain", "draw"}]
    result: dict[str, dict[str, float]] = {}
    for param in params:
        chains = _chains_from_table(table, param)
        result[param] = {
            "rhat": diagnostics.split_rhat(chains),
            "ess_bulk": diagnostics.ess_bulk(chains),
            "ess_tail": diagnostics.ess_tail(chains),
        }
    return result


def compare(
    model: str,
    actual: Mapping[str, Sequence[float]],
    tolerance: float = 0.15,
    metrics: Sequence[str] = ("mean", "std"),
    backend: str = "arrow",
    store: DataStore | None = None,
):
    """Compare actual draws against reference stats.

    Example:
        result = reference.compare("eight_schools", actual=fit.as_dict())
    """
    ref_stats = stats(model, params=list(actual.keys()), backend=backend, store=store)
    actual_stats = compute_stats_from_draws(actual)
    return compare_stats(ref_stats, actual_stats, tolerance=tolerance, metrics=metrics)
