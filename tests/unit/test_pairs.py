"""Tests for reparametrization pair discovery and loading."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mcmc_ref.store import DataStore


def _make_pair_fixture(tmp_path: Path, pair_name: str = "neals_funnel") -> Path:
    """Create a minimal pair directory structure for testing."""
    pairs_dir = tmp_path / "pairs"
    pair_dir = pairs_dir / pair_name
    pair_dir.mkdir(parents=True)

    # pair.json
    (pair_dir / "pair.json").write_text(
        json.dumps(
            {
                "name": pair_name,
                "description": "Neal's funnel: pure funnel geometry",
                "bad_variant": "centered",
                "good_variant": "noncentered",
                "reference_model": f"{pair_name}-noncentered",
                "expected_pathologies": ["divergences", "high_rhat", "low_ess"],
                "difficulty": "easy",
            }
        )
    )

    # centered variant
    centered = pair_dir / "centered"
    centered.mkdir()
    (centered / "model.stan").write_text("parameters { real v; }\nmodel { v ~ normal(0,3); }")
    (centered / "model_spec.json").write_text(
        json.dumps(
            {
                "parameters": [{"name": "v", "shape": [], "constraint": "real"}],
                "data": [{"name": "N", "type": "int"}],
            }
        )
    )
    (centered / "data.json").write_text(json.dumps({"N": 9}))

    # noncentered variant
    noncentered = pair_dir / "noncentered"
    noncentered.mkdir()
    (noncentered / "model.stan").write_text("parameters { real v; }\nmodel { v ~ normal(0,3); }")
    (noncentered / "model_spec.json").write_text(
        json.dumps(
            {
                "parameters": [{"name": "v", "shape": [], "constraint": "real"}],
                "data": [{"name": "N", "type": "int"}],
            }
        )
    )
    (noncentered / "data.json").write_text(json.dumps({"N": 9}))

    return pairs_dir


def _make_reference_draws(tmp_path: Path, model_name: str = "neals_funnel-noncentered") -> None:
    """Create a minimal reference draws parquet for the good variant."""
    draws_dir = tmp_path / "draws"
    meta_dir = tmp_path / "meta"
    draws_dir.mkdir(exist_ok=True)
    meta_dir.mkdir(exist_ok=True)

    table = pa.table(
        {
            "chain": pa.array([0, 0, 1, 1, 2, 2, 3, 3], type=pa.int32()),
            "draw": pa.array([0, 1, 0, 1, 0, 1, 0, 1], type=pa.int32()),
            "v": pa.array([1.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.5, 2.5], type=pa.float64()),
        }
    )
    pq.write_table(table, draws_dir / f"{model_name}.draws.parquet")
    (meta_dir / f"{model_name}.meta.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "parameters": ["v"],
                "n_chains": 4,
                "n_draws_per_chain": 2,
            }
        )
    )


# ------------------------------------------------------------------
# Test: list_pairs
# ------------------------------------------------------------------


def test_list_pairs_discovers_pairs(tmp_path: Path) -> None:
    from mcmc_ref.pairs import list_pairs

    _make_pair_fixture(tmp_path, "neals_funnel")
    _make_pair_fixture(tmp_path, "eight_schools")

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path)
    result = list_pairs(store=store)

    assert sorted(result) == ["eight_schools", "neals_funnel"]


def test_list_pairs_empty_when_no_pairs(tmp_path: Path) -> None:
    from mcmc_ref.pairs import list_pairs

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path)
    result = list_pairs(store=store)

    assert result == []


# ------------------------------------------------------------------
# Test: pair()
# ------------------------------------------------------------------


def test_pair_loads_metadata(tmp_path: Path) -> None:
    from mcmc_ref.pairs import pair

    _make_pair_fixture(tmp_path)
    _make_reference_draws(tmp_path)

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path)
    p = pair("neals_funnel", store=store)

    assert p.name == "neals_funnel"
    assert p.bad_variant == "centered"
    assert p.good_variant == "noncentered"
    assert p.expected_pathologies == ["divergences", "high_rhat", "low_ess"]
    assert p.difficulty == "easy"
    assert p.description == "Neal's funnel: pure funnel geometry"


def test_pair_loads_specs(tmp_path: Path) -> None:
    from mcmc_ref.pairs import pair

    _make_pair_fixture(tmp_path)
    _make_reference_draws(tmp_path)

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path)
    p = pair("neals_funnel", store=store)

    assert isinstance(p.bad_spec, dict)
    assert isinstance(p.good_spec, dict)
    assert p.bad_spec["parameters"][0]["name"] == "v"
    assert p.good_spec["parameters"][0]["name"] == "v"


def test_pair_loads_data(tmp_path: Path) -> None:
    from mcmc_ref.pairs import pair

    _make_pair_fixture(tmp_path)
    _make_reference_draws(tmp_path)

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path)
    p = pair("neals_funnel", store=store)

    # Data comes from the good variant's data.json
    assert isinstance(p.data, dict)
    assert p.data["N"] == 9


def test_pair_loads_stan_code(tmp_path: Path) -> None:
    from mcmc_ref.pairs import pair

    _make_pair_fixture(tmp_path)
    _make_reference_draws(tmp_path)

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path)
    p = pair("neals_funnel", store=store)

    assert isinstance(p.bad_stan, str)
    assert isinstance(p.good_stan, str)
    assert "normal" in p.bad_stan
    assert "normal" in p.good_stan


def test_pair_reference_draws(tmp_path: Path) -> None:
    from mcmc_ref.draws import Draws
    from mcmc_ref.pairs import pair

    _make_pair_fixture(tmp_path)
    _make_reference_draws(tmp_path)

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path)
    p = pair("neals_funnel", store=store)

    draws = p.reference_draws
    assert isinstance(draws, Draws)
    assert "v" in draws.params


def test_pair_reference_stats(tmp_path: Path) -> None:
    from mcmc_ref.pairs import pair

    _make_pair_fixture(tmp_path)
    _make_reference_draws(tmp_path)

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path)
    p = pair("neals_funnel", store=store)

    stats = p.reference_stats
    assert isinstance(stats, dict)
    assert "v" in stats
    assert "mean" in stats["v"]


def test_pair_not_found_raises(tmp_path: Path) -> None:
    import pytest

    from mcmc_ref.pairs import pair

    store = DataStore(local_root=tmp_path, packaged_root=tmp_path)

    with pytest.raises(FileNotFoundError):
        pair("nonexistent", store=store)


# ------------------------------------------------------------------
# Test: bundled pairs (integration-like, uses real package data)
# ------------------------------------------------------------------


def test_bundled_pairs_discoverable() -> None:
    """The package should bundle all 5 reparametrization pairs."""
    from mcmc_ref.pairs import list_pairs

    pairs = list_pairs()
    assert "bangladesh_contraceptive" in pairs
    assert "eight_schools" in pairs
    assert "hierarchical_lr" in pairs
    assert "neals_funnel" in pairs
    assert "varying_slopes" in pairs


def test_bundled_pair_has_reference_draws() -> None:
    """Each bundled good variant should have reference draws."""
    from mcmc_ref.pairs import pair

    p = pair("neals_funnel")
    draws = p.reference_draws
    assert draws is not None
    assert "v" in draws.params
