"""NumPy-accelerated stats backend (optional)."""

from __future__ import annotations

from collections.abc import Iterable


class NumpyBackend:
    name = "numpy"

    def __init__(self) -> None:
        try:
            import numpy  # noqa: F401
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError("numpy is required for the numpy backend") from exc

    def stats(
        self,
        table,
        params: Iterable[str],
        quantiles: Iterable[float] = (0.05, 0.5, 0.95),
        quantile_mode: str = "exact",
    ):
        import numpy as np

        if hasattr(table, "read_all"):
            table = table.read_all()

        qs = list(quantiles)
        results: dict[str, dict[str, float]] = {}
        for param in params:
            if hasattr(table, "column"):
                col = table.column(param)
                if hasattr(col, "to_numpy"):
                    data = col.to_numpy(zero_copy_only=False)
                else:
                    data = np.asarray(col)
            else:
                data = np.asarray(table[param])
            entry = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data, ddof=1)),
            }
            q_vals = np.quantile(data, qs)
            for q, v in zip(qs, q_vals, strict=False):
                key = f"q{int(q * 100)}"
                entry[key] = float(v)
            results[param] = entry
        return results
