from __future__ import annotations

import math

import numpy as np

from navalballistics.gravity import normal_gravity, point_mass_gravity


def test_normal_gravity_at_equator() -> None:
    value = normal_gravity(0.0)
    assert math.isclose(value, 9.780325, rel_tol=1e-6)


def test_point_mass_gravity_direction() -> None:
    position = np.array([6_371_000.0, 0.0, 0.0])
    accel = point_mass_gravity(position)
    assert accel[1] == 0.0 and accel[2] == 0.0
    assert accel[0] < 0.0
