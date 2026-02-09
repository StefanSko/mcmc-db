from __future__ import annotations

import json
import zipfile
from pathlib import Path

from mcmc_ref.informed_references import import_informed_references


def _write_posteriordb_json_zip(path: Path, chains: int = 4, draws: int = 4) -> None:
    payload: list[dict[str, list[float]]] = []
    for chain_idx in range(chains):
        mu = [1.0 + chain_idx * 0.01 + draw_idx * 0.1 for draw_idx in range(draws)]
        payload.append({"mu": mu})

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.stem.replace(".json", "") + ".json", json.dumps(payload))


def test_import_informed_references_attaches_variant_and_info(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_root = tmp_path / "out"
    source_dir.mkdir()
    _write_posteriordb_json_zip(source_dir / "toy_informed.json.zip")
    info = {
        "model": "toy_informed",
        "standardization": {"x": {"mean": 0.1, "std": 2.3}},
    }
    (source_dir / "toy_informed.info.json").write_text(json.dumps(info))

    result = import_informed_references(source_dir=source_dir, output_root=output_root, force=True)

    assert result.total == 1
    assert result.converted == 1
    assert not result.failures

    meta_path = output_root / "meta" / "toy_informed.meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["reference_variant"] == "informed_prior"
    assert meta["informed_reference_info"] == info


def test_import_informed_references_handles_missing_info_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_root = tmp_path / "out"
    source_dir.mkdir()
    _write_posteriordb_json_zip(source_dir / "toy_informed.json.zip")

    result = import_informed_references(source_dir=source_dir, output_root=output_root, force=True)

    assert result.total == 1
    assert result.converted == 1
    assert not result.failures

    meta_path = output_root / "meta" / "toy_informed.meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["reference_variant"] == "informed_prior"
    assert "informed_reference_info" not in meta
