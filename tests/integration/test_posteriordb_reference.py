from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mcmc_ref import convert

POSTERIORDB_ENV = [
    "POSTERIORDB_REF_DRAWS",
    "POSTERIORDB_DRAWS_DIR",
    "POSTERIORDB_PATH",
    "POSTERIORDB_ROOT",
]


def _find_draws_dir() -> Path | None:
    for key in POSTERIORDB_ENV:
        value = os.environ.get(key)
        if value:
            path = Path(value)
            if path.exists():
                if path.name == "draws":
                    return path
                candidate = path / "posterior_database" / "reference_posteriors" / "draws" / "draws"
                if candidate.exists():
                    return candidate
                return path
    fallback = Path("/tmp/jaxstanv3/tests/posteriordb/reference_draws")
    if fallback.exists():
        return fallback
    _maybe_generate_reference_draws(fallback)
    if fallback.exists():
        return fallback
    return None


def _maybe_generate_reference_draws(target_dir: Path) -> None:
    if os.environ.get("MCMC_REF_GENERATE") != "1":
        return
    script = target_dir / "generate_reference.py"
    if not script.exists():
        return
    model = os.environ.get("MCMC_REF_GENERATE_MODEL", "wells_data-wells_dist")
    subprocess.run([sys.executable, str(script), model], check=True)


def test_posteriordb_conversion_smoke(tmp_path: Path) -> None:
    draws_dir = _find_draws_dir()
    if draws_dir is None:
        pytest.skip("No posteriordb reference draws available")
    assert draws_dir is not None

    # Pick a single json.zip for a quick smoke test
    candidates = sorted(draws_dir.glob("*.json.zip"))
    if not candidates:
        pytest.skip("No reference draw files found")

    input_path = candidates[0]
    out_draws = tmp_path / "draws"
    out_meta = tmp_path / "meta"
    out_draws.mkdir()
    out_meta.mkdir()

    convert.convert_file(
        input_path,
        name=input_path.stem.replace(".json", ""),
        out_draws_dir=out_draws,
        out_meta_dir=out_meta,
        force=True,
    )

    assert any(out_draws.glob("*.draws.parquet"))
    assert any(out_meta.glob("*.meta.json"))
