from __future__ import annotations

import json
import zipfile
from pathlib import Path

from mcmc_ref.build_references import build_references


def _write_posteriordb_json_zip(path: Path, chains: int = 4, draws: int = 4) -> None:
    payload: list[dict[str, list[float]]] = []
    for chain_idx in range(chains):
        mu = [1.0 + chain_idx * 0.01 + draw_idx * 0.1 for draw_idx in range(draws)]
        tau = [2.0 + chain_idx * 0.01 + draw_idx * 0.1 for draw_idx in range(draws)]
        payload.append({"mu": mu, "tau": tau})

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.stem.replace(".json", "") + ".json", json.dumps(payload))


def test_build_references_converts_all_models(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_root = tmp_path / "out"
    source_dir.mkdir()
    _write_posteriordb_json_zip(source_dir / "model_a-model_a.json.zip")
    _write_posteriordb_json_zip(source_dir / "model_b-model_b.json.zip")

    result = build_references(source_dir=source_dir, output_root=output_root, force=True)

    assert result.total == 2
    assert result.converted == 2
    assert not result.failures
    assert (output_root / "draws" / "model_a-model_a.draws.parquet").exists()
    assert (output_root / "meta" / "model_b-model_b.meta.json").exists()


def test_build_references_filters_models(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_root = tmp_path / "out"
    source_dir.mkdir()
    _write_posteriordb_json_zip(source_dir / "model_a-model_a.json.zip")
    _write_posteriordb_json_zip(source_dir / "model_b-model_b.json.zip")

    result = build_references(
        source_dir=source_dir,
        output_root=output_root,
        models=["model_b-model_b"],
        force=True,
    )

    assert result.total == 1
    assert result.converted == 1
    assert not result.failures
    assert not (output_root / "draws" / "model_a-model_a.draws.parquet").exists()
    assert (output_root / "draws" / "model_b-model_b.draws.parquet").exists()


def test_build_references_reports_strict_check_failures(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_root = tmp_path / "out"
    source_dir.mkdir()
    _write_posteriordb_json_zip(source_dir / "model_a-model_a.json.zip", chains=1)

    result = build_references(source_dir=source_dir, output_root=output_root)

    assert result.total == 1
    assert result.converted == 0
    assert len(result.failures) == 1
    assert "at least 4 chains" in result.failures[0].error
