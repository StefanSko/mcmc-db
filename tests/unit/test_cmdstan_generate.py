from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from mcmc_ref.cmdstan_generate import (
    build_posteriordb_payload,
    parse_cmdstan_csv,
    write_posteriordb_json_zip,
)


def test_parse_cmdstan_csv_skips_internal_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "chain.csv"
    csv_path.write_text(
        "# comment\n"
        "lp__,accept_stat__,mu,tau,theta.1,beta.2.3\n"
        "-1.0,0.9,1.0,2.0,5.0,0.1\n"
        "-0.9,0.8,1.1,2.1,5.1,0.2\n"
    )

    parsed = parse_cmdstan_csv(csv_path)

    assert parsed == {
        "mu": [1.0, 1.1],
        "tau": [2.0, 2.1],
        "theta[1]": [5.0, 5.1],
        "beta[2,3]": [0.1, 0.2],
    }


def test_build_posteriordb_payload_validates_shape() -> None:
    with pytest.raises(ValueError, match="no chain draws provided"):
        build_posteriordb_payload([])

    bad = [{"mu": [1.0, 2.0]}, {"tau": [1.0, 2.0]}]
    with pytest.raises(ValueError, match="parameter keys mismatch"):
        build_posteriordb_payload(bad)


def test_write_posteriordb_json_zip_roundtrip(tmp_path: Path) -> None:
    payload = [{"mu": [1.0, 1.1]}, {"mu": [0.9, 1.2]}]
    out = write_posteriordb_json_zip(payload, tmp_path / "model.json.zip", model_name="model")

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert names == ["model.json"]
        loaded = json.loads(zf.read(names[0]))
    assert loaded == payload
