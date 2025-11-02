"""Coordinate frame utilities for the naval ballistics simulations."""

from __future__ import annotations

from math import atan2, cos, sin, sqrt
from typing import Tuple

import numpy as np

from .constants import WGS84_A, WGS84_E2

__all__ = [
    "ecef_to_ned_matrix",
    "ned_to_ecef_matrix",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "ecef_to_ned",
    "ned_to_ecef",
]


def ecef_to_ned_matrix(latitude_rad: float, longitude_rad: float) -> np.ndarray:
    """Return the rotation matrix from ECEF to local NED coordinates."""

    s_lat = sin(latitude_rad)
    c_lat = cos(latitude_rad)
    s_lon = sin(longitude_rad)
    c_lon = cos(longitude_rad)

    return np.array(
        [
            [-s_lat * c_lon, -s_lat * s_lon, c_lat],
            [-s_lon, c_lon, 0.0],
            [-c_lat * c_lon, -c_lat * s_lon, -s_lat],
        ]
    )


def ned_to_ecef_matrix(latitude_rad: float, longitude_rad: float) -> np.ndarray:
    """Return the rotation matrix from local NED to ECEF coordinates."""

    return ecef_to_ned_matrix(latitude_rad, longitude_rad).T


def geodetic_to_ecef(latitude_rad: float, longitude_rad: float, altitude_m: float) -> np.ndarray:
    """Convert geodetic coordinates to ECEF."""

    s_lat = sin(latitude_rad)
    c_lat = cos(latitude_rad)
    s_lon = sin(longitude_rad)
    c_lon = cos(longitude_rad)
    n = WGS84_A / sqrt(1.0 - WGS84_E2 * s_lat * s_lat)

    x = (n + altitude_m) * c_lat * c_lon
    y = (n + altitude_m) * c_lat * s_lon
    z = (n * (1.0 - WGS84_E2) + altitude_m) * s_lat
    return np.array([x, y, z])


def ecef_to_geodetic(position_ecef: np.ndarray, tolerance: float = 1e-12, max_iter: int = 10) -> Tuple[float, float, float]:
    """Convert ECEF coordinates to geodetic latitude, longitude, and altitude."""

    x, y, z = position_ecef
    longitude = atan2(y, x)
    p = sqrt(x * x + y * y)
    latitude = atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(max_iter):
        s_lat = sin(latitude)
        n = WGS84_A / sqrt(1.0 - WGS84_E2 * s_lat * s_lat)
        altitude = p / cos(latitude) - n
        new_latitude = atan2(z, p * (1.0 - WGS84_E2 * n / (n + altitude)))
        if abs(new_latitude - latitude) < tolerance:
            latitude = new_latitude
            break
        latitude = new_latitude
    s_lat = sin(latitude)
    n = WGS84_A / sqrt(1.0 - WGS84_E2 * s_lat * s_lat)
    altitude = p / cos(latitude) - n
    return latitude, longitude, altitude


def ecef_to_ned(position_ecef: np.ndarray, reference_lat_rad: float, reference_lon_rad: float, reference_alt_m: float) -> np.ndarray:
    """Convert an ECEF position to local NED coordinates about a reference geodetic location."""

    reference_ecef = geodetic_to_ecef(reference_lat_rad, reference_lon_rad, reference_alt_m)
    delta = position_ecef - reference_ecef
    rotation = ecef_to_ned_matrix(reference_lat_rad, reference_lon_rad)
    return rotation @ delta


def ned_to_ecef(ned: np.ndarray, reference_lat_rad: float, reference_lon_rad: float, reference_alt_m: float) -> np.ndarray:
    """Convert local NED coordinates to an ECEF position about a reference."""

    rotation = ned_to_ecef_matrix(reference_lat_rad, reference_lon_rad)
    reference_ecef = geodetic_to_ecef(reference_lat_rad, reference_lon_rad, reference_alt_m)
    return reference_ecef + rotation @ ned
