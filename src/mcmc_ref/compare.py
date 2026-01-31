"""Comparison utilities for reference vs actual draws."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ParamResult:
    ref: float
    actual: float
    rel_error: float
    passed: bool


@dataclass(frozen=True)
class CompareResult:
    passed: bool
    details: dict[str, dict[str, ParamResult]]
    failures: list[str]


def compare_stats(
    ref_stats: Mapping[str, Mapping[str, float]],
    actual_stats: Mapping[str, Mapping[str, float]],
    tolerance: float,
    metrics: Sequence[str],
) -> CompareResult:
    details: dict[str, dict[str, ParamResult]] = {}
    failures: list[str] = []

    for param, stats in ref_stats.items():
        if param not in actual_stats:
            failures.append(f"missing param: {param}")
            continue
        param_details: dict[str, ParamResult] = {}
        for metric in metrics:
            ref_val = float(stats.get(metric, float("nan")))
            actual_val = float(actual_stats[param].get(metric, float("nan")))
            denom = max(abs(ref_val), 1e-12)
            rel_error = abs(actual_val - ref_val) / denom
            passed = rel_error <= tolerance
            if not passed:
                failures.append(f"{param}.{metric} rel_error={rel_error:.3g} > {tolerance}")
            param_details[metric] = ParamResult(
                ref=ref_val,
                actual=actual_val,
                rel_error=rel_error,
                passed=passed,
            )
        details[param] = param_details

    passed = len(failures) == 0
    return CompareResult(passed=passed, details=details, failures=failures)


def compute_basic_stats(values: Sequence[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan")}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {"mean": float(mean), "std": float(var**0.5)}


def compute_stats_from_draws(draws: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    return {param: compute_basic_stats(values) for param, values in draws.items()}
