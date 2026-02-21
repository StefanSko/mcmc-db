from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from mcmc_ref import convert

REFERENCE_ARCHIVE_ENV = [
    "MCMC_REF_ARCHIVE_DIR",
    "MCMC_REF_ARCHIVE_ROOT",
]


def _write_reference_json_zip(path: Path, chains: int = 4, draws: int = 4) -> None:
    payload: list[dict[str, list[float]]] = []
    for chain_idx in range(chains):
        mu = [1.0 + chain_idx * 0.01 + draw_idx * 0.1 for draw_idx in range(draws)]
        tau = [2.0 + chain_idx * 0.01 + draw_idx * 0.1 for draw_idx in range(draws)]
        payload.append({"mu": mu, "tau": tau})

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.stem.replace(".json", "") + ".json", json.dumps(payload))


def _find_archive_dir(default_dir: Path) -> Path | None:
    for key in REFERENCE_ARCHIVE_ENV:
        value = os.environ.get(key)
        if value:
            path = Path(value)
            if path.exists():
                return path
    if default_dir.exists():
        return default_dir
    _maybe_generate_reference_archives(default_dir)
    return default_dir if default_dir.exists() else None


def _maybe_generate_reference_archives(target_dir: Path) -> None:
    if os.environ.get("MCMC_REF_GENERATE") != "1":
        return
    script = target_dir / "generate_reference.py"
    if not script.exists():
        return
    model = os.environ.get("MCMC_REF_GENERATE_MODEL", "wells_data-wells_dist")
    subprocess.run([sys.executable, str(script), model], check=True)


def test_reference_archive_conversion_smoke(tmp_path: Path) -> None:
    archive_dir = tmp_path / "reference_archives"
    archive_dir.mkdir()
    _write_reference_json_zip(archive_dir / "example-example.json.zip")

    draws_dir = _find_archive_dir(archive_dir)
    assert draws_dir is not None

    # Pick a single json.zip for a quick smoke test
    candidates = sorted(draws_dir.glob("*.json.zip"))
    assert candidates

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
