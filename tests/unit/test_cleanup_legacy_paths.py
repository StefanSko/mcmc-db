from __future__ import annotations

import importlib.util
from pathlib import Path


def test_legacy_entry_points_removed() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text()
    assert "mcmc-ref-build-references" not in text
    assert "mcmc-ref-import-informed" not in text


def test_legacy_modules_removed() -> None:
    assert importlib.util.find_spec("mcmc_ref.build_references") is None
    assert importlib.util.find_spec("mcmc_ref.informed_references") is None
