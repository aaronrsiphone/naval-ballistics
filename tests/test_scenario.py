from __future__ import annotations

from pathlib import Path

from navalballistics.scenario import load_scenario


def test_load_scenario_round_trip(tmp_path: Path) -> None:
    scenario_path = Path("tests/data/baseline.yaml")
    scenario = load_scenario(scenario_path)
    assert scenario.name == "baseline_equator_east"
    assert scenario.launch_site.latitude_deg == 0.0
    assert scenario.projectile.mass_kg == 118.0
    # ensure summary runs without error
    text = scenario.summary()
    assert "Scenario: baseline_equator_east" in text
