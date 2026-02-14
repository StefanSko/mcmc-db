"""Helpers for generating posteriordb-style reference zips from CmdStan output."""

from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any


def parse_cmdstan_csv(path: Path) -> dict[str, list[float]]:
    """Parse one CmdStan chain CSV into {param: draws}, skipping internal columns."""
    rows: list[str] = []
    with Path(path).open() as f:
        for line in f:
            if not line.startswith("#"):
                rows.append(line)
    reader = csv.DictReader(rows)

    columns: dict[str, list[float]] = {}
    for row in reader:
        for key, value in row.items():
            if key is None or key.endswith("__"):
                continue
            normalized = _normalize_cmdstan_param_name(key)
            columns.setdefault(normalized, []).append(float(value))
    return columns


_VECTOR_SUFFIX_RE = re.compile(r"^(?P<base>[A-Za-z_][A-Za-z0-9_]*)((?:\.\d+)+)$")


def _normalize_cmdstan_param_name(name: str) -> str:
    """Convert CmdStan CSV names (theta.1.2) to Stan-style names (theta[1,2])."""
    m = _VECTOR_SUFFIX_RE.match(name)
    if not m:
        return name
    indices = m.group(2).lstrip(".").split(".")
    return f"{m.group('base')}[{','.join(indices)}]"


def build_posteriordb_payload(
    chain_draws: list[dict[str, list[float]]],
) -> list[dict[str, list[float]]]:
    """Validate and return posteriordb payload: list[chain][param] -> draws."""
    if not chain_draws:
        raise ValueError("no chain draws provided")

    params = set(chain_draws[0].keys())
    if not params:
        raise ValueError("chain draws contain no parameters")

    for idx, chain in enumerate(chain_draws):
        if set(chain.keys()) != params:
            raise ValueError(f"chain {idx} parameter keys mismatch")
        lens = {len(values) for values in chain.values()}
        if len(lens) != 1:
            raise ValueError(f"chain {idx} has inconsistent draw counts")
    return chain_draws


def write_posteriordb_json_zip(
    payload: list[dict[str, list[float]]],
    out_path: Path,
    *,
    model_name: str,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{model_name}.json", json.dumps(payload))
    return out_path


def write_provenance(path: Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path
