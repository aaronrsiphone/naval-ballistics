from __future__ import annotations

import math

import numpy as np

from navalballistics import frames


def test_geodetic_round_trip() -> None:
    lat = math.radians(45.0)
    lon = math.radians(12.0)
    alt = 150.0
    ecef = frames.geodetic_to_ecef(lat, lon, alt)
    lat2, lon2, alt2 = frames.ecef_to_geodetic(ecef)
    assert math.isclose(lat, lat2, abs_tol=1e-8)
    assert math.isclose(lon, lon2, abs_tol=1e-8)
    assert math.isclose(alt, alt2, abs_tol=1e-3)


def test_ned_round_trip() -> None:
    lat = math.radians(30.0)
    lon = math.radians(-70.0)
    alt = 10.0
    offset = np.array([100.0, -50.0, 20.0])
    ecef = frames.ned_to_ecef(offset, lat, lon, alt)
    ned = frames.ecef_to_ned(ecef, lat, lon, alt)
    assert np.allclose(offset, ned, atol=1e-6)
