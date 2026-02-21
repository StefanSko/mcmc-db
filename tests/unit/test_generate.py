from __future__ import annotations

import json
import zipfile
from pathlib import Path

from mcmc_ref import generate, provenance


def _fake_runner(
    *,
    model_name: str,
    recipe,
    stan_file: Path,
    data_file: Path,
    archive_path: Path,
    config: generate.GenerationConfig,
) -> None:
    _ = recipe
    _ = stan_file
    _ = data_file
    payload: list[dict[str, list[float]]] = []
    for chain_idx in range(config.chains):
        draws = [float(chain_idx) + 0.01 * float(i) for i in range(config.iter_sampling)]
        payload.append({"mu": draws, "sigma": [1.0 + value for value in draws]})

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{model_name}.json", json.dumps(payload))


def test_generate_reference_corpus_from_scaffold(tmp_path: Path) -> None:
    scaffold_root = tmp_path / "scaffold"
    provenance.materialize_scaffold(scaffold_root)

    output_root = tmp_path / "generated"
    result = generate.generate_reference_corpus(
        scaffold_root=scaffold_root,
        output_root=output_root,
        models=["dugongs", "radon_pooled_informed"],
        runner=_fake_runner,
        force=True,
    )

    assert result.generated == 2
    assert result.failed == 0
    assert (output_root / "archives" / "dugongs.json.zip").exists()
    assert (output_root / "draws" / "dugongs.draws.parquet").exists()
    assert (output_root / "meta" / "radon_pooled_informed.meta.json").exists()


def test_publish_reference_data(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    (source_root / "draws").mkdir(parents=True)
    (source_root / "meta").mkdir(parents=True)
    (source_root / "draws" / "demo.draws.parquet").write_bytes(b"parquet")
    (source_root / "meta" / "demo.meta.json").write_text("{}")

    scaffold_root = tmp_path / "scaffold"
    provenance.materialize_scaffold(scaffold_root)

    package_root = tmp_path / "package_data"
    publish = generate.publish_reference_data(
        source_root=source_root,
        scaffold_root=scaffold_root,
        package_root=package_root,
    )

    assert publish.draws_copied == 1
    assert publish.meta_copied == 1
    assert publish.pairs_copied == 5
    assert (package_root / "draws" / "demo.draws.parquet").exists()
    assert (package_root / "meta" / "demo.meta.json").exists()
    assert (package_root / "pairs" / "neals_funnel" / "pair.json").exists()
    assert (package_root / "provenance_manifest.json").exists()


def test_publish_reference_data_requires_source_dirs(tmp_path: Path) -> None:
    scaffold_root = tmp_path / "scaffold"
    provenance.materialize_scaffold(scaffold_root)

    source_root = tmp_path / "missing_source"
    package_root = tmp_path / "package_data"

    try:
        generate.publish_reference_data(
            source_root=source_root,
            scaffold_root=scaffold_root,
            package_root=package_root,
        )
    except FileNotFoundError as exc:
        assert "source draws/meta directories must exist" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError when source draws/meta are missing")


def test_publish_reference_data_replaces_stale_targets(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    (source_root / "draws").mkdir(parents=True)
    (source_root / "meta").mkdir(parents=True)
    (source_root / "draws" / "fresh.draws.parquet").write_bytes(b"parquet")
    (source_root / "meta" / "fresh.meta.json").write_text("{}")

    scaffold_root = tmp_path / "scaffold"
    provenance.materialize_scaffold(scaffold_root)

    package_root = tmp_path / "package_data"
    (package_root / "draws").mkdir(parents=True)
    (package_root / "meta").mkdir(parents=True)
    (package_root / "pairs" / "stale_pair").mkdir(parents=True)
    (package_root / "draws" / "stale.draws.parquet").write_bytes(b"stale")
    (package_root / "meta" / "stale.meta.json").write_text("{}")
    (package_root / "pairs" / "stale_pair" / "pair.json").write_text("{}")
    (package_root / "provenance_manifest.json").write_text('{"old": true}')

    publish = generate.publish_reference_data(
        source_root=source_root,
        scaffold_root=scaffold_root,
        package_root=package_root,
    )

    assert publish.draws_copied == 1
    assert publish.meta_copied == 1
    assert publish.pairs_copied == 5
    assert (package_root / "draws" / "fresh.draws.parquet").exists()
    assert (package_root / "meta" / "fresh.meta.json").exists()
    assert not (package_root / "draws" / "stale.draws.parquet").exists()
    assert not (package_root / "meta" / "stale.meta.json").exists()
    assert not (package_root / "pairs" / "stale_pair").exists()
