"""Simplified Modified Point-Mass (MPM) trajectory integrator.

Heights reported by this module are referenced to the WGS-84 ellipsoid via the
``ecef_to_geodetic`` conversion. If you require sea-level impact detection, supply
an appropriate geoid correction before interpreting the final altitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .atmosphere import StandardAtmosphere1976
from .frames import ecef_to_geodetic, ecef_to_ned
from .gravity import GravityModel, resolve_gravity
from .rotation import centrifugal_acceleration, coriolis_acceleration

__all__ = [
    "MPMConfig",
    "MPMResult",
    "simulate_mpm",
    "WindFunc",
]


WindFunc = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class MPMConfig:
    """Configuration parameters for the simplified MPM integrator."""

    mass_kg: float
    reference_area_m2: float
    drag_coefficient: float = 0.0
    cd_of_mach: Optional[Callable[[float], float]] = None
    include_earth_rotation: bool = True
    gravity_is_normal: bool = True
    time_step: float = 0.01
    max_time: float = 300.0


@dataclass(frozen=True)
class MPMResult:
    """Timeseries output from the MPM simulation."""

    times: np.ndarray
    positions_ecef: np.ndarray
    velocities_ecef: np.ndarray
    latitudes_rad: np.ndarray
    longitudes_rad: np.ndarray
    altitudes_m: np.ndarray
    mach_number: np.ndarray
    dynamic_pressure_pa: np.ndarray
    north_displacement_m: np.ndarray
    east_displacement_m: np.ndarray
    ground_range_m: np.ndarray


def _drag_acceleration(
    velocity_rel: np.ndarray,
    density: float,
    speed_of_sound: float,
    config: MPMConfig,
) -> np.ndarray:
    speed = np.linalg.norm(velocity_rel)
    if speed == 0.0:
        return np.zeros(3)
    if config.cd_of_mach is not None and speed_of_sound > 0.0:
        mach = speed / speed_of_sound
        drag_coefficient = float(config.cd_of_mach(mach))
    else:
        drag_coefficient = config.drag_coefficient
    if drag_coefficient == 0.0:
        return np.zeros(3)
    coeff = -0.5 * density * drag_coefficient * config.reference_area_m2 / config.mass_kg
    return coeff * speed * velocity_rel


def simulate_mpm(
    initial_position_ecef: np.ndarray,
    initial_velocity_ecef: np.ndarray,
    config: MPMConfig,
    gravity: str | GravityModel = "normal",
    atmosphere: Optional[StandardAtmosphere1976] = None,
    wind_ecef: Optional[np.ndarray] = None,
    wind_func_ecef: Optional[WindFunc] = None,
) -> MPMResult:
    """Simulate a trajectory using a stripped-down MPM formulation."""

    gravity_model = resolve_gravity(gravity)
    atmosphere_model = atmosphere or StandardAtmosphere1976()
    wind_vector = np.zeros(3, dtype=float) if wind_ecef is None else np.asarray(wind_ecef, dtype=float)
    wind_fn: WindFunc = wind_func_ecef or (lambda position, time: wind_vector)

    use_coriolis = config.include_earth_rotation
    use_centrifugal = config.include_earth_rotation and not config.gravity_is_normal

    dt = config.time_step
    max_steps = int(np.ceil(config.max_time / dt))

    def derivatives(state: np.ndarray, time: float) -> np.ndarray:
        position = state[:3]
        velocity = state[3:]
        latitude, longitude, altitude = ecef_to_geodetic(position)
        properties = atmosphere_model.properties(max(0.0, altitude))
        density = properties["density"]
        speed_of_sound = properties.get("speed_of_sound", 0.0)
        wind = wind_fn(position, time)
        relative_velocity = velocity - wind
        drag = _drag_acceleration(relative_velocity, density, speed_of_sound, config)
        gravity_vector = gravity_model.acceleration(position)
        coriolis = coriolis_acceleration(velocity) if use_coriolis else np.zeros(3)
        centrifugal = (
            centrifugal_acceleration(position)
            if use_centrifugal
            else np.zeros(3)
        )
        acceleration = drag + gravity_vector + coriolis + centrifugal
        return np.concatenate((velocity, acceleration))

    def sample(state: np.ndarray, time: float, reference: tuple[float, float, float]) -> dict[str, float]:
        position = state[:3]
        velocity = state[3:]
        latitude, longitude, altitude = ecef_to_geodetic(position)
        altitude_clamped = max(0.0, altitude)
        properties = atmosphere_model.properties(altitude_clamped)
        density = properties["density"]
        speed_of_sound = properties.get("speed_of_sound", 0.0)
        wind = wind_fn(position, time)
        relative_velocity = velocity - wind
        speed = np.linalg.norm(relative_velocity)
        mach = speed / speed_of_sound if speed_of_sound > 0.0 else 0.0
        dynamic_pressure = 0.5 * density * speed * speed
        ned = ecef_to_ned(position, *reference)
        north = float(ned[0])
        east = float(ned[1])
        ground_range = float(np.hypot(north, east))
        return {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "altitude": float(altitude),
            "mach": float(mach),
            "dynamic_pressure": float(dynamic_pressure),
            "north": north,
            "east": east,
            "ground_range": ground_range,
        }

    state = np.concatenate(
        (
            np.asarray(initial_position_ecef, dtype=float),
            np.asarray(initial_velocity_ecef, dtype=float),
        )
    )
    if not np.all(np.isfinite(state)):
        raise ValueError("Initial state must contain finite values.")

    latitude0, longitude0, altitude0 = ecef_to_geodetic(state[:3])
    reference = (latitude0, longitude0, altitude0)

    current_sample = sample(state, 0.0, reference)

    times = [0.0]
    positions = [state[:3].copy()]
    velocities = [state[3:].copy()]
    latitudes = [current_sample["latitude"]]
    longitudes = [current_sample["longitude"]]
    altitudes = [current_sample["altitude"]]
    mach_numbers = [current_sample["mach"]]
    dynamic_pressures = [current_sample["dynamic_pressure"]]
    north_displacements = [current_sample["north"]]
    east_displacements = [current_sample["east"]]
    ground_ranges = [current_sample["ground_range"]]

    time = 0.0

    for step in range(1, max_steps + 1):
        prev_state = state.copy()
        prev_time = time
        prev_altitude = float(current_sample["altitude"])
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Non-finite state encountered during integration.")

        k1 = derivatives(state, time)
        k2 = derivatives(state + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = derivatives(state + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = derivatives(state + dt * k3, time + dt)
        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(new_state)):
            raise RuntimeError("Non-finite state encountered during integration.")
        new_time = time + dt
        new_sample = sample(new_state, new_time, reference)

        altitude = float(new_sample["altitude"])
        if altitude < 0.0 and prev_altitude > 0.0:
            fraction = prev_altitude / (prev_altitude - altitude + 1e-12)
            state = prev_state + fraction * (new_state - prev_state)
            time = prev_time + fraction * dt
            current_sample = sample(state, time, reference)
            times.append(time)
            positions.append(state[:3].copy())
            velocities.append(state[3:].copy())
            latitudes.append(current_sample["latitude"])
            longitudes.append(current_sample["longitude"])
            altitudes.append(current_sample["altitude"])
            mach_numbers.append(current_sample["mach"])
            dynamic_pressures.append(current_sample["dynamic_pressure"])
            north_displacements.append(current_sample["north"])
            east_displacements.append(current_sample["east"])
            ground_ranges.append(current_sample["ground_range"])
            break

        state = new_state
        time = new_time
        current_sample = new_sample

        times.append(time)
        positions.append(state[:3].copy())
        velocities.append(state[3:].copy())
        latitudes.append(current_sample["latitude"])
        longitudes.append(current_sample["longitude"])
        altitudes.append(current_sample["altitude"])
        mach_numbers.append(current_sample["mach"])
        dynamic_pressures.append(current_sample["dynamic_pressure"])
        north_displacements.append(current_sample["north"])
        east_displacements.append(current_sample["east"])
        ground_ranges.append(current_sample["ground_range"])

        if time >= config.max_time:
            break

    return MPMResult(
        times=np.asarray(times),
        positions_ecef=np.vstack(positions),
        velocities_ecef=np.vstack(velocities),
        latitudes_rad=np.asarray(latitudes),
        longitudes_rad=np.asarray(longitudes),
        altitudes_m=np.asarray(altitudes),
        mach_number=np.asarray(mach_numbers),
        dynamic_pressure_pa=np.asarray(dynamic_pressures),
        north_displacement_m=np.asarray(north_displacements),
        east_displacement_m=np.asarray(east_displacements),
        ground_range_m=np.asarray(ground_ranges),
    )
