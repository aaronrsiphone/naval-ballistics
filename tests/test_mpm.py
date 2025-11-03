from __future__ import annotations

import numpy as np
import pytest

from navalballistics import frames
from navalballistics.atmosphere import StandardAtmosphere1976
from navalballistics.mpm import MPMConfig, simulate_mpm


def test_mpm_vertical_throw_no_drag_no_rotation() -> None:
    latitude = np.deg2rad(0.0)
    longitude = np.deg2rad(0.0)
    altitude = 0.0
    initial_position = frames.geodetic_to_ecef(latitude, longitude, altitude)
    velocity_ned = np.array([0.0, 0.0, -300.0])
    initial_velocity = frames.ned_to_ecef_matrix(latitude, longitude) @ velocity_ned
    config = MPMConfig(
        mass_kg=100.0,
        reference_area_m2=0.1,
        drag_coefficient=0.0,
        include_earth_rotation=False,
        gravity_is_normal=False,
        time_step=0.01,
        max_time=90.0,
    )
    result = simulate_mpm(initial_position, initial_velocity, config, gravity="point-mass")
    tof_expected = 2.0 * 300.0 / 9.80665
    assert result.times[-1] == pytest.approx(tof_expected, rel=5e-3)
    assert result.altitudes_m[-1] == pytest.approx(0.0, abs=1e-3)
    assert np.max(result.altitudes_m) > 4000.0


def test_mpm_wind_callable_sets_relative_flow() -> None:
    latitude = np.deg2rad(45.0)
    longitude = np.deg2rad(12.0)
    altitude = 0.0
    initial_position = frames.geodetic_to_ecef(latitude, longitude, altitude)
    initial_velocity = np.zeros(3)
    atmosphere = StandardAtmosphere1976()
    properties = atmosphere.properties(0.0)

    wind_speed = 50.0

    def wind_fn(position: np.ndarray, time: float) -> np.ndarray:
        return np.array([wind_speed, 0.0, 0.0])

    config = MPMConfig(
        mass_kg=50.0,
        reference_area_m2=0.05,
        drag_coefficient=0.2,
        include_earth_rotation=False,
        gravity_is_normal=False,
        time_step=0.1,
        max_time=0.2,
    )
    result = simulate_mpm(
        initial_position,
        initial_velocity,
        config,
        gravity="point-mass",
        atmosphere=atmosphere,
        wind_ecef=np.zeros(3),
        wind_func_ecef=wind_fn,
    )

    expected_dynamic_pressure = 0.5 * properties["density"] * wind_speed**2
    expected_mach = wind_speed / properties["speed_of_sound"]

    assert result.dynamic_pressure_pa[0] == pytest.approx(expected_dynamic_pressure, rel=1e-6)
    assert result.mach_number[0] == pytest.approx(expected_mach, rel=1e-6)
