"""Arrow-based stats backend.

This backend depends only on pyarrow and can operate on Arrow Tables
or RecordBatchReaders. It keeps memory reasonable by working column-wise.
"""

from __future__ import annotations

from collections.abc import Iterable


class ArrowBackend:
    name = "arrow"

    def __init__(self) -> None:
        try:
            import pyarrow  # noqa: F401
            import pyarrow.compute as pc  # noqa: F401
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError("pyarrow is required for the arrow backend") from exc

    def stats(
        self,
        table,
        params: Iterable[str],
        quantiles: Iterable[float] = (0.05, 0.5, 0.95),
        quantile_mode: str = "exact",
    ) -> dict[str, dict[str, float]]:
        import pyarrow.compute as pc

        if hasattr(table, "read_all"):
            table = table.read_all()

        qs = list(quantiles)
        results: dict[str, dict[str, float]] = {}
        for param in params:
            col = table.column(param)
            mean = pc.mean(col).as_py()  # type: ignore[attr-defined]
            std = pc.stddev(col).as_py()  # type: ignore[attr-defined]
            q_vals = pc.quantile(  # type: ignore[attr-defined]
                col, q=qs, interpolation="linear", skip_nulls=True
            )
            if hasattr(q_vals, "to_pylist"):
                q_list = [float(v) for v in q_vals.to_pylist()]
            else:
                q_list = [float(q_vals.as_py())]
            entry = {"mean": float(mean), "std": float(std)}
            for q, v in zip(qs, q_list, strict=False):
                key = f"q{int(q * 100)}"
                entry[key] = float(v)
            results[param] = entry
        return results
