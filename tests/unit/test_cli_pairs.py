"""Tests for CLI pair commands."""

from __future__ import annotations

import json

from click.testing import CliRunner

from mcmc_ref.cli import main


def test_cli_pairs_lists_bundled_pairs() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["pairs"])

    assert result.exit_code == 0
    assert "neals_funnel" in result.output
    assert "eight_schools" in result.output
    assert "hierarchical_lr" in result.output
    assert "varying_slopes" in result.output


def test_cli_pairs_json_format() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["pairs", "--format", "json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert "neals_funnel" in data


def test_cli_pair_info() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["pair", "neals_funnel"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == "neals_funnel"
    assert data["bad_variant"] == "centered"
    assert data["good_variant"] == "noncentered"
    assert "expected_pathologies" in data
    assert "difficulty" in data


def test_cli_pair_not_found() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["pair", "nonexistent"])

    assert result.exit_code != 0
