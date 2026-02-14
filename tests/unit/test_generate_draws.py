"""Tests for generate_draws module with mocked cmdstanpy."""

from __future__ import annotations

import json
import random
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.csv as pacsv

from mcmc_ref.generate_draws import _compile_and_sample, generate_reference_draws


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
        for param_idx, p in enumerate(params):
            # IID normal draws per chain/parameter for stable R-hat/ESS diagnostics.
            rng = random.Random(10_000 * chain_idx + param_idx)
            columns[p] = [rng.gauss(0.0, 1.0) for _ in range(n_draws)]
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


@patch("mcmc_ref.generate_draws._compile_and_sample")
@patch("mcmc_ref.generate_draws.convert.convert_file")
def test_generate_reference_draws_enforces_convert_checks(
    mock_convert_file, mock_sample, tmp_path: Path
) -> None:
    stan_code = "parameters { real mu; }"
    stan_data = {}
    output_root = tmp_path / "output"

    mock_fit = _create_mock_fit(tmp_path / "cmdstan_out", params=["mu"], n_chains=4, n_draws=2500)
    mock_sample.return_value = mock_fit

    generate_reference_draws(
        model_name="strict_checks",
        stan_code=stan_code,
        stan_data=stan_data,
        output_root=output_root,
    )

    kwargs = mock_convert_file.call_args.kwargs
    assert kwargs.get("force", False) is False


def test_compile_and_sample_cleans_temp_build_dir() -> None:
    tracker: dict[str, Path] = {}

    class FakeCmdStanModel:
        def __init__(self, *, stan_file: str) -> None:
            tracker["stan_file"] = Path(stan_file)

        def sample(self, **_: object) -> object:
            assert tracker["stan_file"].exists()
            return object()

    fake_cmdstanpy = types.SimpleNamespace(CmdStanModel=FakeCmdStanModel)

    with patch.dict("sys.modules", {"cmdstanpy": fake_cmdstanpy}):
        _compile_and_sample(
            stan_code="parameters { real mu; }",
            stan_data={},
            chains=4,
            iter_warmup=100,
            iter_sampling=100,
            thin=1,
            seed=123,
        )

    assert not tracker["stan_file"].parent.exists()
