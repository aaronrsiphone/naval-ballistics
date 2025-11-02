"""Scenario configuration models and loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import yaml

__all__ = [
    "LaunchSiteConfig",
    "LaunchConfig",
    "ProjectileConfig",
    "IntegratorConfig",
    "EnvironmentConfig",
    "Scenario",
    "load_scenario",
    "load_scenarios",
]


@dataclass(frozen=True)
class LaunchSiteConfig:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0


@dataclass(frozen=True)
class LaunchConfig:
    muzzle_velocity_mps: float
    elevation_deg: float
    azimuth_deg: float


@dataclass(frozen=True)
class ProjectileConfig:
    mass_kg: float
    diameter_m: float
    length_m: float | None = None
    spin_rps: float | None = None


@dataclass(frozen=True)
class IntegratorConfig:
    method: str = "RK45"
    rtol: float = 1e-9
    atol: float = 1e-12


@dataclass(frozen=True)
class EnvironmentConfig:
    atmosphere: str = "1976-standard"
    wind_model: str = "none"
    gravity: str | Mapping[str, Any] = "normal"


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str
    launch_site: LaunchSiteConfig
    launch: LaunchConfig
    projectile: ProjectileConfig
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of the scenario."""

        lines = [f"Scenario: {self.name}", f"  Model: {self.model}"]
        if self.description:
            lines.append(f"  Description: {self.description}")
        lines.extend(
            [
                "  Launch Site:",
                f"    Latitude : {self.launch_site.latitude_deg:.4f}°",
                f"    Longitude: {self.launch_site.longitude_deg:.4f}°",
                f"    Altitude : {self.launch_site.altitude_m:.1f} m",
                "  Launch Conditions:",
                f"    Muzzle Velocity: {self.launch.muzzle_velocity_mps:.2f} m/s",
                f"    Elevation      : {self.launch.elevation_deg:.2f}°",
                f"    Azimuth        : {self.launch.azimuth_deg:.2f}°",
                "  Projectile:",
                f"    Mass          : {self.projectile.mass_kg:.2f} kg",
                f"    Diameter      : {self.projectile.diameter_m:.4f} m",
            ]
        )
        if self.projectile.length_m is not None:
            lines.append(f"    Length        : {self.projectile.length_m:.4f} m")
        if self.projectile.spin_rps is not None:
            lines.append(f"    Spin Rate     : {self.projectile.spin_rps:.2f} rps")
        lines.append("  Environment:")
        lines.append(f"    Atmosphere    : {self.environment.atmosphere}")
        lines.append(f"    Wind Model    : {self.environment.wind_model}")
        lines.append(f"    Gravity       : {self.environment.gravity}")
        lines.append("  Integrator:")
        lines.append(f"    Method        : {self.integrator.method}")
        lines.append(f"    rtol          : {self.integrator.rtol}")
        lines.append(f"    atol          : {self.integrator.atol}")
        return "\n".join(lines)

    @staticmethod
    def from_mapping(mapping: Mapping[str, Any]) -> "Scenario":
        name = str(mapping.get("name", "unnamed"))
        model = str(mapping.get("model", "sixdof"))
        description = mapping.get("description")
        launch_site = _parse_launch_site(mapping.get("launch_site", {}))
        launch = _parse_launch(mapping.get("launch", {}))
        projectile = _parse_projectile(mapping.get("projectile", {}))
        integrator = _parse_integrator(mapping.get("integrator", {}))
        environment = _parse_environment(mapping.get("environment", {}))
        metadata = mapping.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        return Scenario(
            name=name,
            model=model,
            description=description,
            launch_site=launch_site,
            launch=launch,
            projectile=projectile,
            integrator=integrator,
            environment=environment,
            metadata=metadata,
        )


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario configuration from a YAML file."""

    with Path(path).open("r", encoding="utf8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise TypeError("Scenario file must contain a mapping at the top level")
    return Scenario.from_mapping(data)


def load_scenarios(paths: Iterable[str | Path]) -> List[Scenario]:
    """Load multiple scenarios from a collection of YAML files."""

    return [load_scenario(path) for path in paths]


def _parse_launch_site(mapping: Mapping[str, Any]) -> LaunchSiteConfig:
    try:
        latitude = float(mapping["latitude_deg"])
        longitude = float(mapping["longitude_deg"])
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Missing required launch_site field: {exc.args[0]}") from exc
    altitude = float(mapping.get("altitude_m", 0.0))
    return LaunchSiteConfig(latitude, longitude, altitude)


def _parse_launch(mapping: Mapping[str, Any]) -> LaunchConfig:
    try:
        muzzle_velocity = float(mapping["muzzle_velocity_mps"])
        elevation = float(mapping["elevation_deg"])
        azimuth = float(mapping["azimuth_deg"])
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Missing required launch field: {exc.args[0]}") from exc
    return LaunchConfig(muzzle_velocity, elevation, azimuth)


def _parse_projectile(mapping: Mapping[str, Any]) -> ProjectileConfig:
    try:
        mass = float(mapping["mass_kg"])
        diameter = float(mapping["diameter_m"])
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Missing required projectile field: {exc.args[0]}") from exc
    length = mapping.get("length_m")
    spin = mapping.get("spin_rps")
    return ProjectileConfig(mass, diameter, None if length is None else float(length), None if spin is None else float(spin))


def _parse_integrator(mapping: Mapping[str, Any]) -> IntegratorConfig:
    method = str(mapping.get("method", "RK45"))
    rtol = float(mapping.get("rtol", 1e-9))
    atol = float(mapping.get("atol", 1e-12))
    return IntegratorConfig(method, rtol, atol)


def _parse_environment(mapping: Mapping[str, Any]) -> EnvironmentConfig:
    atmosphere = str(mapping.get("atmosphere", "1976-standard"))
    wind_model = str(mapping.get("wind_model", "none"))
    gravity = mapping.get("gravity", "normal")
    return EnvironmentConfig(atmosphere, wind_model, gravity)
