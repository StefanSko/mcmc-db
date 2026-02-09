"""Helpers for working with packaged reference model inventories."""

from __future__ import annotations

from pathlib import Path


def reference_models_from_draws(draws_dir: Path) -> list[str]:
    draws_dir = Path(draws_dir)
    return sorted(path.stem.replace(".draws", "") for path in draws_dir.glob("*.draws.parquet"))


def split_informed_models(models: list[str]) -> tuple[list[str], list[str]]:
    informed: list[str] = []
    standard: list[str] = []
    for model in models:
        if model.endswith("_informed"):
            informed.append(model)
        else:
            standard.append(model)
    return standard, informed
