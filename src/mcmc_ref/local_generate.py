"""Helpers for local (posteriordb-free) reference generation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LocalModelSpec:
    model: str
    stan_path: Path
    data_path: Path


def discover_local_model_specs(
    *,
    models_dir: Path,
    data_dir: Path,
    models: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[LocalModelSpec]:
    models_dir = Path(models_dir)
    data_dir = Path(data_dir)

    if models is None:
        model_names = sorted(path.stem for path in models_dir.glob("*.stan"))
    else:
        model_names = list(models)

    if limit is not None:
        model_names = model_names[:limit]

    specs: list[LocalModelSpec] = []
    missing: list[str] = []
    for model in model_names:
        stan_path = models_dir / f"{model}.stan"
        data_path = data_dir / f"{model}.data.json"
        if not stan_path.exists() or not data_path.exists():
            missing.append(model)
            continue
        specs.append(LocalModelSpec(model=model, stan_path=stan_path, data_path=data_path))

    if missing:
        raise ValueError(
            "Missing local model inputs for: " + ", ".join(missing)
        )
    return specs


def load_stan_data(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Stan data JSON must be an object: {path}")
    return payload
