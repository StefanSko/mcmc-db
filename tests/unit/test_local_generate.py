from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcmc_ref.local_generate import discover_local_model_specs, load_stan_data


def test_discover_local_model_specs(tmp_path: Path) -> None:
    models_dir = tmp_path / "stan_models"
    data_dir = tmp_path / "stan_data"
    models_dir.mkdir()
    data_dir.mkdir()

    (models_dir / "a.stan").write_text("parameters { real x; } model { x ~ normal(0,1); }")
    (data_dir / "a.data.json").write_text("{}")
    (models_dir / "b.stan").write_text("parameters { real x; } model { x ~ normal(0,1); }")
    (data_dir / "b.data.json").write_text("{}")

    specs = discover_local_model_specs(models_dir=models_dir, data_dir=data_dir)

    assert [s.model for s in specs] == ["a", "b"]


def test_discover_local_model_specs_missing_inputs_raises(tmp_path: Path) -> None:
    models_dir = tmp_path / "stan_models"
    data_dir = tmp_path / "stan_data"
    models_dir.mkdir()
    data_dir.mkdir()

    (models_dir / "a.stan").write_text("parameters { real x; } model { x ~ normal(0,1); }")

    with pytest.raises(ValueError, match="Missing local model inputs"):
        discover_local_model_specs(models_dir=models_dir, data_dir=data_dir)


def test_load_stan_data_requires_object(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps([1, 2, 3]))

    with pytest.raises(ValueError, match="must be an object"):
        load_stan_data(path)
