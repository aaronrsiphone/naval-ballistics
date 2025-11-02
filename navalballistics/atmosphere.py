"""1976 Standard Atmosphere implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .constants import GAS_CONSTANT_AIR, HEAT_CAPACITY_RATIO_AIR, STANDARD_GRAVITY

__all__ = [
    "StandardAtmosphere1976",
]


_LAYER_BASES_M = np.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 86_000.0])
_LAPSE_RATES_K_M = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])
_BASE_TEMPERATURES_K = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65])
_BASE_PRESSURES_PA = np.array([101_325.0, 22_632.06, 5_474.889, 868.0187, 110.9063, 66.93887, 3.956420])
_GAMMA = HEAT_CAPACITY_RATIO_AIR


@dataclass(frozen=True)
class StandardAtmosphere1976:
    """Evaluate properties of the 1976 Standard Atmosphere."""

    def properties(self, altitude_m: float) -> Dict[str, float]:
        """Return temperature, pressure, density, and speed of sound."""

        if altitude_m < 0.0:
            altitude_m = 0.0
        if altitude_m > _LAYER_BASES_M[-1]:
            altitude_m = _LAYER_BASES_M[-1]
        idx = int(np.searchsorted(_LAYER_BASES_M[1:], altitude_m, side="right"))
        h0 = _LAYER_BASES_M[idx]
        T0 = _BASE_TEMPERATURES_K[idx]
        P0 = _BASE_PRESSURES_PA[idx]
        lapse = _LAPSE_RATES_K_M[idx]
        dh = altitude_m - h0
        if abs(lapse) < 1e-12:
            temperature = T0
            exponent = -STANDARD_GRAVITY * dh / (GAS_CONSTANT_AIR * T0)
            pressure = P0 * np.exp(exponent)
        else:
            temperature = T0 + lapse * dh
            pressure = P0 * (T0 / temperature) ** (STANDARD_GRAVITY / (GAS_CONSTANT_AIR * lapse))
        density = pressure / (GAS_CONSTANT_AIR * temperature)
        speed_of_sound = np.sqrt(_GAMMA * GAS_CONSTANT_AIR * temperature)
        return {
            "temperature": float(temperature),
            "pressure": float(pressure),
            "density": float(density),
            "speed_of_sound": float(speed_of_sound),
        }

    def __call__(self, altitude_m: float) -> Dict[str, float]:  # pragma: no cover - delegation
        return self.properties(altitude_m)
