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
        local = self._resolve_in_root(self._local, model, "draws", ".draws.parquet")
        if local is not None:
            return local
        packaged = self._resolve_in_root(self._packaged, model, "draws", ".draws.parquet")
        if packaged is not None:
            return packaged
        raise FileNotFoundError(f"draws not found for model: {model}")

    def resolve_meta_path(self, model: str) -> Path:
        local = self._resolve_in_root(self._local, model, "meta", ".meta.json")
        if local is not None:
            return local
        packaged = self._resolve_in_root(self._packaged, model, "meta", ".meta.json")
        if packaged is not None:
            return packaged
        raise FileNotFoundError(f"metadata not found for model: {model}")

    def read_meta(self, model: str) -> dict:
        path = self.resolve_meta_path(model)
        return json.loads(path.read_text())

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
        if not draws.exists() and not meta.exists():
            return None
        return StorePaths(root=root, draws=draws, meta=meta)


def _default_local_root() -> Path:
    env = os.environ.get("MCMC_REF_LOCAL_ROOT")
    if env:
        return Path(env)
    return Path.home() / ".mcmc-ref"


def _default_packaged_root() -> Path | None:
    try:
        from importlib import resources
    except Exception:
        return None

    for package_name in ("mcmc_ref", "mcmc_ref_data"):
        try:
            package_data = Path(str(resources.files(package_name).joinpath("data")))
        except Exception:
            continue
        draws = package_data / "draws"
        meta = package_data / "meta"
        if draws.exists() or meta.exists():
            return package_data
    return None
