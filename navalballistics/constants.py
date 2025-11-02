"""Physical constants used by the naval ballistics models."""

from __future__ import annotations

from math import sqrt

WGS84_A = 6_378_137.0
"""Equatorial radius of the Earth in meters (WGS-84)."""

WGS84_F = 1.0 / 298.257_223_563
"""Flattening of the Earth (WGS-84)."""

WGS84_B = WGS84_A * (1.0 - WGS84_F)
"""Polar radius of the Earth in meters (derived)."""

WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
"""First eccentricity squared of the WGS-84 ellipsoid."""

EARTH_ANGULAR_RATE = 7.292_115_0e-5
"""Magnitude of Earth's rotation rate in rad/s."""

EARTH_MU = 3.986_004_418e14
"""Earth's standard gravitational parameter in m^3/s^2."""

STANDARD_GRAVITY = 9.80665
"""Standard gravitational acceleration in m/s^2."""

GAS_CONSTANT_AIR = 287.052_874
"""Specific gas constant for dry air in J/(kg*K)."""

HEAT_CAPACITY_RATIO_AIR = 1.4
"""Specific heat ratio for diatomic gases (dry air)."""


def radius_of_curvature_meridian(latitude_rad: float) -> float:
    """Return the meridional radius of curvature for the WGS-84 ellipsoid."""

    sin_lat = __import__("math").sin(latitude_rad)
    denom = (1.0 - WGS84_E2 * sin_lat * sin_lat) ** 1.5
    return WGS84_A * (1.0 - WGS84_E2) / denom


def radius_of_curvature_prime_vertical(latitude_rad: float) -> float:
    """Return the prime vertical radius of curvature for WGS-84."""

    sin_lat = __import__("math").sin(latitude_rad)
    return WGS84_A / sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
