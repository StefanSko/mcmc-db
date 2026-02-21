from __future__ import annotations

import json
import zipfile
from pathlib import Path

from mcmc_ref.build_references import build_references


def _write_reference_json_zip(path: Path, chains: int = 4, draws: int = 4) -> None:
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
    _write_reference_json_zip(source_dir / "model_a-model_a.json.zip")
    _write_reference_json_zip(source_dir / "model_b-model_b.json.zip")

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
    _write_reference_json_zip(source_dir / "model_a-model_a.json.zip")
    _write_reference_json_zip(source_dir / "model_b-model_b.json.zip")

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
    _write_reference_json_zip(source_dir / "model_a-model_a.json.zip", chains=1)

    result = build_references(source_dir=source_dir, output_root=output_root)

    assert result.total == 1
    assert result.converted == 0
    assert len(result.failures) == 1
    assert "at least 4 chains" in result.failures[0].error


def _create_posteriordb_layout(root: Path, posterior_name: str, data_name: str, model_name: str):
    """Create a minimal posteriordb directory layout for testing."""
    # Posterior definition
    posteriors_dir = root / "posteriors"
    posteriors_dir.mkdir(parents=True, exist_ok=True)
    posterior_def = {
        "name": posterior_name,
        "model_name": model_name,
        "data_name": data_name,
        "reference_posterior_name": None,
    }
    (posteriors_dir / f"{posterior_name}.json").write_text(json.dumps(posterior_def))

    # Stan model code
    models_dir = root / "models" / "stan"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{model_name}.stan").write_text("data { int N; }\nparameters { real mu; }")

    # Data as json.zip
    data_dir = root / "data" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_content = json.dumps({"N": 10, "x": [1.0, 2.0, 3.0]})
    zip_path = data_dir / f"{data_name}.json.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{data_name}.json", data_content)


def test_extract_stan_assets(tmp_path: Path) -> None:
    from mcmc_ref.build_references import extract_stan_assets

    pdb_root = tmp_path / "posteriordb"
    output_root = tmp_path / "out"

    _create_posteriordb_layout(pdb_root, "test_data-test_model", "test_data", "test_model")

    result = extract_stan_assets(
        posteriordb_root=pdb_root,
        output_root=output_root,
        posteriors=["test_data-test_model"],
    )

    assert result.extracted == 1
    assert not result.failures

    # Check stan_data was extracted
    data_path = output_root / "stan_data" / "test_data-test_model.data.json"
    assert data_path.exists()
    data = json.loads(data_path.read_text())
    assert data["N"] == 10

    # Check stan_code was copied
    code_path = output_root / "stan_code" / "test_data-test_model.stan"
    assert code_path.exists()
    assert "data { int N; }" in code_path.read_text()


def test_extract_stan_assets_reports_missing(tmp_path: Path) -> None:
    from mcmc_ref.build_references import extract_stan_assets

    pdb_root = tmp_path / "posteriordb"
    output_root = tmp_path / "out"

    # Create posteriors dir but no actual files
    (pdb_root / "posteriors").mkdir(parents=True)

    result = extract_stan_assets(
        posteriordb_root=pdb_root,
        output_root=output_root,
        posteriors=["nonexistent-model"],
    )

    assert result.extracted == 0
    assert len(result.failures) == 1
