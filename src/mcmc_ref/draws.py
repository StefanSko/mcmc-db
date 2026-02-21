"""Lightweight draws wrapper with optional NumPy conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Draws:
    data: Any
    params: list[str]
    chains: list[int] | None = None
    meta: dict[str, Any] | None = None

    def to_arrow(self) -> Any:
        """Return the underlying Arrow object (Table or RecordBatchReader)."""
        return self.data

    def to_numpy(self) -> Any:
        """Materialize to NumPy (raises if NumPy is not installed)."""
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError("numpy is required for to_numpy()") from exc

        table = self._as_table()
        cols = [table.column(p).to_numpy(zero_copy_only=False) for p in self.params]
        return np.stack(cols, axis=-1)

    def to_list(self) -> list[dict[str, Any]]:
        """Return a list of row dicts if available from Arrow."""
        table = self._as_table()
        if hasattr(table, "to_pylist"):
            return table.to_pylist()
        return list(table)

    def _as_table(self) -> Any:
        if hasattr(self.data, "read_all"):
            return self.data.read_all()
        return self.data


def coerce_return(draws: Draws, return_: str) -> Any:
    if return_ == "draws":
        return draws
    if return_ == "arrow":
        return draws.to_arrow()
    if return_ == "numpy":
        return draws.to_numpy()
    if return_ == "list":
        return draws.to_list()
    raise ValueError(f"Unknown return type: {return_}")
