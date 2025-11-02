"""Typer-based command-line interface for navalballistics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import typer
import yaml

from .scenario import Scenario, load_scenario, load_scenarios

app = typer.Typer(help="Naval exterior ballistics simulation toolkit")


def _echo_scenario_summary(scenario: Scenario) -> None:
    typer.echo(scenario.summary())


@app.command()
def run(
    scenario: Path = typer.Argument(..., exists=True, readable=True, help="Scenario YAML file"),
    model: Optional[str] = typer.Option(None, help="Override model type (mpm or sixdof)"),
    output: Optional[Path] = typer.Option(None, help="Optional output directory for run artifacts"),
) -> None:
    """Execute a single scenario."""

    scenario_data = load_scenario(scenario)
    if model:
        scenario_data = replace(scenario_data, model=model)
    typer.echo("Loaded scenario configuration:")
    _echo_scenario_summary(scenario_data)
    if output:
        output.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Artifacts would be written to: {output}")
    typer.echo("Simulation execution is not yet implemented.")


@app.command()
def sweep(
    sweep_file: Path = typer.Argument(..., exists=True, readable=True, help="YAML file listing scenarios"),
    model: Optional[str] = typer.Option(None, help="Override model for all scenarios"),
) -> None:
    """Run a parameter sweep defined by a YAML file."""

    with sweep_file.open("r", encoding="utf8") as fh:
        data = yaml.safe_load(fh)
    scenarios_section = data.get("scenarios", []) if isinstance(data, dict) else []
    if not scenarios_section:
        typer.echo("Sweep file does not define any scenarios.")
        raise typer.Exit(code=1)
    scenario_paths = [Path(path) for path in scenarios_section]
    loaded = load_scenarios(scenario_paths)
    typer.echo(f"Loaded {len(loaded)} scenarios:")
    for scenario_data in loaded:
        effective_model = model or scenario_data.model
        typer.echo(f"- {scenario_data.name} (model={effective_model})")
    typer.echo("Sweep execution is not yet implemented.")


@app.command()
def compare(
    runs: List[Path] = typer.Argument(..., exists=True, readable=True, help="Run directories to compare"),
    metrics: str = typer.Option("range,drop,drift,tof", help="Comma-separated list of metrics"),
) -> None:
    """Compare existing simulation runs."""

    typer.echo("Comparison tooling is not yet implemented.")
    typer.echo(f"Selected runs: {', '.join(str(run) for run in runs)}")
    typer.echo(f"Metrics: {metrics}")


@app.command()
def plot(
    runs: List[Path] = typer.Argument(..., exists=True, readable=True, help="Run directories to plot"),
    view: str = typer.Option("3d", help="Plot view: 3d, groundtrack, or range"),
) -> None:
    """Plot simulation results."""

    typer.echo("Plotting is not yet implemented.")
    typer.echo(f"Runs: {', '.join(str(run) for run in runs)}")
    typer.echo(f"View: {view}")


@app.command()
def export(
    run: Path = typer.Argument(..., exists=True, readable=True, help="Run directory to export"),
    format: str = typer.Option("csv", help="Export format (csv, parquet, json)"),
) -> None:
    """Export simulation data to a chosen format."""

    typer.echo("Export is not yet implemented.")
    typer.echo(f"Run: {run}")
    typer.echo(f"Format: {format}")


def main() -> None:  # pragma: no cover - CLI entry point
    app()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
