from __future__ import annotations

import numpy as np
import pytest

from navalballistics.rotation import (
    OMEGA_ECEF,
    centrifugal_acceleration,
    coriolis_acceleration,
    earth_rotation_vector,
    rotation_apparent_accel,
)


def test_earth_rotation_vector_points_z() -> None:
    omega = earth_rotation_vector()
    assert np.allclose(omega[:2], 0.0)
    assert omega[2] > 0.0


def test_coriolis_acceleration_sign() -> None:
    velocity = np.array([100.0, 0.0, 0.0])
    accel = coriolis_acceleration(velocity)
    # At the equator with eastward velocity, expect a southward Coriolis acceleration.
    assert accel[1] < 0.0
    assert accel[0] == pytest.approx(0.0)
    assert accel[2] == pytest.approx(0.0)


def test_centrifugal_acceleration_direction() -> None:
    position = np.array([6_371_000.0, 0.0, 0.0])
    accel = centrifugal_acceleration(position)
    assert accel[0] > 0.0
    assert accel[1] == pytest.approx(0.0)
    assert accel[2] == pytest.approx(0.0)


def test_centrifugal_equator_matches_omega_squared_r() -> None:
    radius = 6_378_137.0
    accel = centrifugal_acceleration(np.array([radius, 0.0, 0.0]))
    magnitude = np.linalg.norm(accel)
    expected = (OMEGA_ECEF[2] ** 2) * radius
    assert magnitude == pytest.approx(expected, rel=1e-6)


def test_centrifugal_vanishes_at_pole() -> None:
    radius = 6_378_137.0
    accel = centrifugal_acceleration(np.array([0.0, 0.0, radius]))
    assert np.linalg.norm(accel) == pytest.approx(0.0, abs=1e-12)


def test_coriolis_supports_batch_inputs() -> None:
    velocity = np.repeat(np.array([[100.0, 0.0, 0.0]]), repeats=3, axis=0)
    accel = coriolis_acceleration(velocity)
    assert accel.shape == (3, 3)
    assert np.all(accel[:, 0] == pytest.approx(0.0))


def test_rotation_apparent_guard_for_normal_gravity() -> None:
    position = np.array([6_371_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 100.0, 0.0])
    accel = rotation_apparent_accel(position, velocity, include_centrifugal=False)
    accel_only_coriolis = coriolis_acceleration(velocity)
    assert np.allclose(accel, accel_only_coriolis)
