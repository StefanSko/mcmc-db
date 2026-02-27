from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mcmc_ref import pairs, reference
from mcmc_ref.store import DataStore


def test_package_data_contains_provenance_manifest_pairs_and_issue_models() -> None:
    package_root = Path("packages/mcmc-ref-data/src/mcmc_ref_data/data")

    assert (package_root / "provenance_manifest.json").exists()
    assert (package_root / "pairs" / "neals_funnel" / "pair.json").exists()
    assert (package_root / "draws" / "dugongs.draws.parquet").exists()
    assert (package_root / "draws" / "radon_pooled.draws.parquet").exists()
    assert (package_root / "meta" / "dugongs.meta.json").exists()
    assert (package_root / "meta" / "radon_pooled.meta.json").exists()
    assert (package_root / "stan_models" / "dugongs.stan").exists()
    assert (package_root / "stan_models" / "radon_pooled.stan").exists()
    assert (package_root / "stan_data" / "dugongs.data.json").exists()
    assert (package_root / "stan_data" / "radon_pooled.data.json").exists()

    store = DataStore(local_root=None, packaged_root=package_root)
    assert len(pairs.list_pairs(store=store)) == 5
    models = reference.list_models(store=store)
    assert "dugongs" in models
    assert "radon_pooled" in models

    draws_dir = package_root / "draws"
    meta_dir = package_root / "meta"
    stan_models_dir = package_root / "stan_models"
    stan_data_dir = package_root / "stan_data"

    draws = {p.name.replace(".draws.parquet", "") for p in draws_dir.glob("*.draws.parquet")}
    meta = {p.name.replace(".meta.json", "") for p in meta_dir.glob("*.meta.json")}
    stan_models = {p.name.replace(".stan", "") for p in stan_models_dir.glob("*.stan")}
    stan_data = {p.name.replace(".data.json", "") for p in stan_data_dir.glob("*.data.json")}

    assert draws - meta == set()
    assert draws - stan_models == set()
    assert draws - stan_data == set()


def test_stan_models_contains_only_stan_source_files() -> None:
    """stan_models/ should only contain .stan source files, not compiled binaries."""
    stan_models_dir = Path("packages/mcmc-ref-data/src/mcmc_ref_data/data/stan_models")
    non_stan = [f.name for f in stan_models_dir.iterdir() if f.is_file() and f.suffix != ".stan"]
    assert non_stan == [], f"Non-.stan files in stan_models/: {non_stan}"


def test_every_pair_reference_model_has_draws_and_meta() -> None:
    """Issue #14: every pair's reference_model must have packaged draws + meta."""
    package_root = Path("packages/mcmc-ref-data/src/mcmc_ref_data/data")
    pairs_dir = package_root / "pairs"
    draws_dir = package_root / "draws"
    meta_dir = package_root / "meta"

    missing: list[str] = []
    for pair_dir in sorted(pairs_dir.iterdir()):
        pair_json = pair_dir / "pair.json"
        if not pair_json.exists():
            continue
        pair_meta = json.loads(pair_json.read_text())
        ref_model = pair_meta["reference_model"]

        draws_path = draws_dir / f"{ref_model}.draws.parquet"
        meta_path = meta_dir / f"{ref_model}.meta.json"

        if not draws_path.exists():
            missing.append(f"{pair_dir.name}: missing draws/{ref_model}.draws.parquet")
        if not meta_path.exists():
            missing.append(f"{pair_dir.name}: missing meta/{ref_model}.meta.json")

    assert missing == [], "Pair reference artifacts missing:\n" + "\n".join(missing)


def test_provenance_manifest_hashes_match_packaged_files() -> None:
    package_root = Path("packages/mcmc-ref-data/src/mcmc_ref_data/data")
    manifest_path = package_root / "provenance_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    expected_files = manifest.get("files")
    assert isinstance(expected_files, dict)
    assert expected_files

    actual_files: dict[str, str] = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(package_root).as_posix()
        if rel == "provenance_manifest.json":
            continue
        actual_files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()

    assert expected_files == actual_files
