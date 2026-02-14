"""Tests for generate_draws module with mocked cmdstanpy."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.csv as pacsv

from mcmc_ref.generate_draws import generate_reference_draws


def _create_mock_fit(tmp_path: Path, params: list[str], n_chains: int = 10, n_draws: int = 1000):
    """Create a mock CmdStanMCMC object with CSV output files."""
    fit = MagicMock()
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Create CSV files that mimic CmdStan output
    csv_paths = []
    for chain_idx in range(n_chains):
        csv_path = tmp_path / f"output_{chain_idx}.csv"
        # Build table data
        columns: dict[str, list[float]] = {}
        for p in params:
            columns[p] = [1.0 + chain_idx * 0.01 + i * 0.001 for i in range(n_draws)]
        table = pa.table(columns)
        pacsv.write_csv(table, csv_path)
        csv_paths.append(str(csv_path))

    fit.runset.csv_files = csv_paths
    fit.draws_pd.side_effect = AttributeError("mocked - no draws_pd")
    return fit


@patch("mcmc_ref.generate_draws._compile_and_sample")
def test_generate_reference_draws(mock_sample, tmp_path: Path) -> None:
    stan_code = "data { int N; } parameters { real mu; } model { mu ~ normal(0, 1); }"
    stan_data = {"N": 10}
    output_root = tmp_path / "output"

    mock_fit = _create_mock_fit(tmp_path / "cmdstan_out", params=["mu"], n_chains=10, n_draws=1000)
    mock_sample.return_value = mock_fit

    result = generate_reference_draws(
        model_name="test_model",
        stan_code=stan_code,
        stan_data=stan_data,
        output_root=output_root,
    )

    assert result.exists()
    assert (output_root / "draws" / "test_model.draws.parquet").exists()
    assert (output_root / "meta" / "test_model.meta.json").exists()

    meta = json.loads((output_root / "meta" / "test_model.meta.json").read_text())
    assert meta["model"] == "test_model"
    assert meta["n_chains"] == 10


@patch("mcmc_ref.generate_draws._compile_and_sample")
def test_generate_reference_draws_multi_param(mock_sample, tmp_path: Path) -> None:
    stan_code = "parameters { real mu; real sigma; }"
    stan_data = {}
    output_root = tmp_path / "output"

    mock_fit = _create_mock_fit(
        tmp_path / "cmdstan_out", params=["mu", "sigma"], n_chains=10, n_draws=1000
    )
    mock_sample.return_value = mock_fit

    result = generate_reference_draws(
        model_name="multi_param",
        stan_code=stan_code,
        stan_data=stan_data,
        output_root=output_root,
    )

    assert result.exists()
    meta = json.loads((output_root / "meta" / "multi_param.meta.json").read_text())
    assert "mu" in meta["parameters"]
    assert "sigma" in meta["parameters"]
