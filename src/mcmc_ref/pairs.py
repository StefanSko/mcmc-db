"""Reparametrization test pairs API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import reference
from .draws import Draws
from .store import DataStore


@dataclass(frozen=True)
class Pair:
    name: str
    description: str
    bad_variant: str
    good_variant: str
    reference_model: str
    expected_pathologies: list[str]
    difficulty: str
    bad_spec: dict[str, Any]
    good_spec: dict[str, Any]
    bad_stan: str
    good_stan: str
    data: dict[str, Any]
    _store: DataStore = field(repr=False)

    @property
    def reference_draws(self) -> Draws:
        return reference.draws(self.reference_model, return_="draws", store=self._store)

    @property
    def reference_stats(self) -> dict[str, dict[str, float]]:
        return reference.stats(self.reference_model, store=self._store)


def list_pairs(store: DataStore | None = None) -> list[str]:
    """List all available reparametrization pair names."""
    store = store or DataStore()
    names: set[str] = set()
    for pairs_dir in _pairs_dirs(store):
        if pairs_dir.is_dir():
            for child in pairs_dir.iterdir():
                if child.is_dir() and (child / "pair.json").exists():
                    names.add(child.name)
    return sorted(names)


def pair(name: str, store: DataStore | None = None) -> Pair:
    """Load a reparametrization pair by name."""
    store = store or DataStore()
    pair_dir = _resolve_pair_dir(name, store)
    meta = json.loads((pair_dir / "pair.json").read_text())

    bad_variant = meta["bad_variant"]
    good_variant = meta["good_variant"]

    bad_dir = pair_dir / bad_variant
    good_dir = pair_dir / good_variant

    bad_spec = json.loads((bad_dir / "model_spec.json").read_text())
    good_spec = json.loads((good_dir / "model_spec.json").read_text())
    bad_stan = (bad_dir / "model.stan").read_text()
    good_stan = (good_dir / "model.stan").read_text()

    # Data: prefer good variant's data.json, fall back to bad
    data_path = good_dir / "data.json"
    if not data_path.exists():
        data_path = bad_dir / "data.json"
    data = json.loads(data_path.read_text()) if data_path.exists() else {}

    return Pair(
        name=meta["name"],
        description=meta.get("description", ""),
        bad_variant=bad_variant,
        good_variant=good_variant,
        reference_model=meta["reference_model"],
        expected_pathologies=meta.get("expected_pathologies", []),
        difficulty=meta.get("difficulty", ""),
        bad_spec=bad_spec,
        good_spec=good_spec,
        bad_stan=bad_stan,
        good_stan=good_stan,
        data=data,
        _store=store,
    )


def _pairs_dirs(store: DataStore) -> list[Path]:
    """Return all pairs directories to search (local + packaged)."""
    dirs = []
    if store._local is not None:
        d = store._local.root / "pairs"
        if d.is_dir():
            dirs.append(d)
    if store._packaged is not None:
        d = store._packaged.root / "pairs"
        if d.is_dir():
            dirs.append(d)
    return dirs


def _resolve_pair_dir(name: str, store: DataStore) -> Path:
    """Find the pair directory by name."""
    for pairs_dir in _pairs_dirs(store):
        candidate = pairs_dir / name
        if candidate.is_dir() and (candidate / "pair.json").exists():
            return candidate
    raise FileNotFoundError(f"pair not found: {name}")
