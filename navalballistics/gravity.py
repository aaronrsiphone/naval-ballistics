"""Gravity models for the naval ballistics simulations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sin, sqrt
from typing import Mapping

import numpy as np

from .constants import EARTH_MU, WGS84_E2

__all__ = [
    "GravityModel",
    "NormalGravity",
    "PointMassGravity",
    "normal_gravity",
    "point_mass_gravity",
]


@dataclass(frozen=True)
class GravityModel:
    """Base class for gravity models."""

    name: str

    def acceleration(self, position_ecef: np.ndarray) -> np.ndarray:
        """Return the gravity acceleration vector in ECEF coordinates."""

        raise NotImplementedError


@dataclass(frozen=True)
class NormalGravity(GravityModel):
    """Normal gravity model using the Somigliana formula with free-air correction."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        object.__setattr__(self, "name", "normal")

    def acceleration(self, position_ecef: np.ndarray) -> np.ndarray:
        latitude = np.arctan2(position_ecef[2], sqrt(position_ecef[0] ** 2 + position_ecef[1] ** 2))
        altitude = np.linalg.norm(position_ecef) - 6_371_000.0
        magnitude = normal_gravity(latitude, altitude)
        unit = -position_ecef / np.linalg.norm(position_ecef)
        return magnitude * unit


@dataclass(frozen=True)
class PointMassGravity(GravityModel):
    """Gravity model that uses an inverse-square law about Earth's center."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        object.__setattr__(self, "name", "point-mass")

    def acceleration(self, position_ecef: np.ndarray) -> np.ndarray:
        return point_mass_gravity(position_ecef)


def normal_gravity(latitude_rad: float, altitude_m: float = 0.0) -> float:
    """Compute the WGS-84 normal gravity magnitude.

    Parameters
    ----------
    latitude_rad:
        Geodetic latitude in radians.
    altitude_m:
        Altitude above the WGS-84 ellipsoid in meters.
    """

    sin_lat = sin(latitude_rad)
    sin2 = sin_lat * sin_lat
    gamma_e = 9.780_325_3359
    k = 0.001_931_852_65241
    gamma_0 = gamma_e * (1.0 + k * sin2) / sqrt(1.0 - WGS84_E2 * sin2)
    # Free-air correction with a small latitude-dependent term (Heiskanen & Moritz).
    return gamma_0 - (3.086e-6 - 0.000_000_004_27 * sin2) * altitude_m + 7.2e-13 * altitude_m**2


def point_mass_gravity(position_ecef: np.ndarray) -> np.ndarray:
    """Return the point-mass gravity acceleration vector."""

    radius = np.linalg.norm(position_ecef)
    if radius == 0.0:
        raise ValueError("Position for gravity calculation cannot be the origin.")
    return -(EARTH_MU / radius**3) * position_ecef


def resolve_gravity(model: str | Mapping[str, object] | GravityModel) -> GravityModel:
    """Resolve a gravity model specification into a concrete object."""

    if isinstance(model, GravityModel):
        return model
    if isinstance(model, Mapping):
        model = model.get("name", "normal")
    model = str(model).lower()
    if model in {"normal", "wgs84"}:
        return NormalGravity()
    if model in {"point", "point-mass", "point_mass"}:
        return PointMassGravity()
    raise ValueError(f"Unsupported gravity model: {model}")
