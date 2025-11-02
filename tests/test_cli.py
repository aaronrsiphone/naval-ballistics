from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from navalballistics import cli


runner = CliRunner()


def test_cli_run_command(tmp_path: Path) -> None:
    scenario = Path("tests/data/baseline.yaml")
    result = runner.invoke(cli.app, ["run", str(scenario), "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert "Loaded scenario configuration" in result.stdout


def test_cli_sweep_command() -> None:
    sweep = Path("tests/data/sweep.yaml")
    result = runner.invoke(cli.app, ["sweep", str(sweep)])
    assert result.exit_code == 0
    assert "Loaded 2 scenarios" in result.stdout
