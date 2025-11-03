"""Earth rotation utilities for Coriolis and centrifugal accelerations.

All functions return apparent accelerations expressed in Earth-Centred, Earth-Fixed
(ECEF) coordinates. The sign conventions match the rotating-frame equation of
motion::

    a_total = (forces / m) - 2 Ω×v - Ω×(Ω×r)

Only add the centrifugal term when your gravity model does *not* already include
centrifugal relief (e.g., point-mass gravity). WGS-84 "normal" gravity embeds the
centrifugal contribution, so adding it again would double-count and bias results.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .constants import EARTH_ANGULAR_RATE

__all__ = [
    "OMEGA_ECEF",
    "earth_rotation_vector",
    "coriolis_acceleration",
    "centrifugal_acceleration",
    "rotation_apparent_accel",
]


# Cache Ω as a module-level constant to avoid repeated allocations. Values are in
# rad/s about the positive ECEF z-axis.
OMEGA_ECEF = np.array([0.0, 0.0, float(EARTH_ANGULAR_RATE)], dtype=float)


def earth_rotation_vector() -> np.ndarray:
    """Return Earth's angular velocity vector in ECEF (shape ``(3,)``).

    The returned array is an alias of :data:`OMEGA_ECEF`; callers must not mutate
    it in-place.
    """

    return OMEGA_ECEF


def _as_vec3(array: np.ndarray) -> np.ndarray:
    """Convert *array* to ``float64`` and validate the trailing dimension."""

    values = np.asarray(array, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError(f"Expected last dimension of size 3, received {values.shape}.")
    return values


def coriolis_acceleration(
    velocity_ecef: np.ndarray,
    angular_velocity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Coriolis acceleration ``a_cor = -2 Ω × v`` in ECEF coordinates.

    Parameters
    ----------
    velocity_ecef:
        Array whose final dimension contains the velocity components in ECEF. Any
        leading batch dimensions are preserved in the output.
    angular_velocity:
        Optional angular velocity vector. Defaults to :data:`OMEGA_ECEF`.
    """

    v = _as_vec3(velocity_ecef)
    omega = _as_vec3(OMEGA_ECEF if angular_velocity is None else angular_velocity)
    return -2.0 * np.cross(omega, v)


def centrifugal_acceleration(
    position_ecef: np.ndarray,
    angular_velocity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Centrifugal acceleration ``a_cen = -Ω × (Ω × r)`` in ECEF coordinates.

    Parameters
    ----------
    position_ecef:
        Array whose final dimension contains the position components in ECEF. Any
        leading batch dimensions are preserved in the output.
    angular_velocity:
        Optional angular velocity vector. Defaults to :data:`OMEGA_ECEF`.
    """

    r = _as_vec3(position_ecef)
    omega = _as_vec3(OMEGA_ECEF if angular_velocity is None else angular_velocity)
    return -np.cross(omega, np.cross(omega, r))


def rotation_apparent_accel(
    position_ecef: np.ndarray,
    velocity_ecef: np.ndarray,
    *,
    include_centrifugal: bool = True,
    angular_velocity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return ``a_cor + a_cen`` with a guard for normal-gravity usage.

    Set ``include_centrifugal=False`` when using a "normal" gravity model that
    already incorporates centrifugal relief (Somigliana/WGS-84). This helper keeps
    the call-sites honest and minimises double-counting mistakes.
    """

    a_cor = coriolis_acceleration(velocity_ecef, angular_velocity)
    if include_centrifugal:
        a_cen = centrifugal_acceleration(position_ecef, angular_velocity)
        return a_cor + a_cen
    return a_cor
