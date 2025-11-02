from __future__ import annotations

import math

from navalballistics.atmosphere import StandardAtmosphere1976


def test_sea_level_properties() -> None:
    atmosphere = StandardAtmosphere1976()
    props = atmosphere.properties(0.0)
    assert math.isclose(props["temperature"], 288.15, rel_tol=1e-5)
    assert math.isclose(props["pressure"], 101_325.0, rel_tol=1e-4)
    assert math.isclose(props["density"], 1.225, rel_tol=1e-3)


def test_stratosphere_temperature_plateau() -> None:
    atmosphere = StandardAtmosphere1976()
    props = atmosphere.properties(11_000.0)
    assert math.isclose(props["temperature"], 216.65, rel_tol=1e-4)
    assert props["pressure"] < 23_000.0
