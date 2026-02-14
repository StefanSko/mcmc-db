"""Data store for bundled + local reference draws."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.dataset as ds


@dataclass(frozen=True)
class StorePaths:
    root: Path
    draws: Path
    meta: Path
    stan_data: Path
    stan_code: Path


class DataStore:
    def __init__(self, local_root: Path | None = None, packaged_root: Path | None = None) -> None:
        self._local = self._init_root(local_root or _default_local_root())
        self._packaged = self._init_root(packaged_root or _default_packaged_root())

    def list_models(self) -> list[str]:
        names: set[str] = set()
        for root in (self._packaged, self._local):
            if root is None:
                continue
            for path in root.draws.glob("*.draws.parquet"):
                names.add(path.stem.replace(".draws", ""))
        return sorted(names)

    def resolve_draws_path(self, model: str) -> Path:
        return self._resolve(model, "draws", ".draws.parquet")

    def resolve_meta_path(self, model: str) -> Path:
        return self._resolve(model, "meta", ".meta.json")

    def read_meta(self, model: str) -> dict:
        path = self.resolve_meta_path(model)
        return json.loads(path.read_text())

    def resolve_stan_data_path(self, model: str) -> Path:
        return self._resolve(model, "stan_data", ".data.json")

    def resolve_stan_code_path(self, model: str) -> Path:
        return self._resolve(model, "stan_code", ".stan")

    def read_stan_data(self, model: str) -> dict:
        path = self.resolve_stan_data_path(model)
        return json.loads(path.read_text())

    def read_stan_code(self, model: str) -> str:
        path = self.resolve_stan_code_path(model)
        return path.read_text()

    def open_draws(
        self,
        model: str,
        params: Sequence[str] | None = None,
        chains: Sequence[int] | None = None,
        batch_size: int = 1024,
    ):
        path = self.resolve_draws_path(model)
        dataset = ds.dataset(path, format="parquet")
        available = dataset.schema.names
        selected_params = self._select_params(available, params)
        columns = ["chain", "draw", *selected_params]
        filt = None
        if chains is not None:
            filt = ds.field("chain").isin(list(chains))
        scanner = dataset.scanner(columns=columns, filter=filt, batch_size=batch_size)
        return scanner.to_reader()

    def _select_params(self, columns: Sequence[str], params: Sequence[str] | None) -> list[str]:
        if params is None:
            return [c for c in columns if c not in {"chain", "draw"}]
        return list(params)

    def _resolve(self, model: str, subdir: str, suffix: str) -> Path:
        for root in (self._local, self._packaged):
            result = self._resolve_in_root(root, model, subdir, suffix)
            if result is not None:
                return result
        raise FileNotFoundError(f"{subdir} not found for model: {model}")

    def _resolve_in_root(
        self, root: StorePaths | None, model: str, subdir: str, suffix: str
    ) -> Path | None:
        if root is None:
            return None
        path = getattr(root, subdir) / f"{model}{suffix}"
        return path if path.exists() else None

    def _init_root(self, root: Path | None) -> StorePaths | None:
        if root is None:
            return None
        draws = root / "draws"
        meta = root / "meta"
        stan_data = root / "stan_data"
        stan_code = root / "stan_code"
        if not any(d.exists() for d in (draws, meta, stan_data, stan_code)):
            return None
        return StorePaths(
            root=root, draws=draws, meta=meta, stan_data=stan_data, stan_code=stan_code
        )


def _default_local_root() -> Path:
    env = os.environ.get("MCMC_REF_LOCAL_ROOT")
    if env:
        return Path(env)
    return Path.home() / ".mcmc-ref"


def _default_packaged_root() -> Path | None:
    try:
        from importlib import resources

        package = resources.files("mcmc_ref").joinpath("data")
        return Path(str(package))
    except Exception:
        return None
