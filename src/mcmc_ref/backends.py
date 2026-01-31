"""Lightweight backend interface for stats and conversions.

This keeps Arrow as the default compute path and allows optional NumPy
acceleration without making NumPy a hard dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol


class Backend(Protocol):
    name: str

    def stats(
        self,
        table: Any,
        params: Iterable[str],
        quantiles: Iterable[float] = (0.05, 0.5, 0.95),
        quantile_mode: str = "exact",
    ) -> dict[str, dict[str, float]]:
        """Compute per-parameter stats from an Arrow Table or batches."""


@dataclass(frozen=True)
class BackendSpec:
    name: str
    loader: Callable[[], Backend]


def _load_arrow() -> Backend:
    from .backends_arrow import ArrowBackend  # lazy import

    return ArrowBackend()


def _load_numpy() -> Backend:
    from .backends_numpy import NumpyBackend  # lazy import

    return NumpyBackend()


BACKENDS: dict[str, BackendSpec] = {
    "arrow": BackendSpec(name="arrow", loader=_load_arrow),
    "numpy": BackendSpec(name="numpy", loader=_load_numpy),
}


def get_backend(name: str) -> Backend:
    spec = BACKENDS.get(name)
    if spec is None:
        raise ValueError(f"Unknown backend: {name}")
    return spec.loader()
