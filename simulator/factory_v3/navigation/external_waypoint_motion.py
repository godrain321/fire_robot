"""Load a ROS map-frame waypoint queue for motion-only simulation override."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(frozen=True)
class ExternalWaypointMotionConfig:
    enabled: bool = False
    waypoint_file: str | None = None
    source_frame: str = "map"
    target_frame: str = "factory_v3_world_xy_m"
    simulation_to_map_translation_x_m: float = 13.00189199
    simulation_to_map_translation_y_m: float = -29.41813371
    simulation_to_map_rotation_deg: float = 60.29982582450894

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("external waypoint enabled must be bool")
        if self.enabled and not self.waypoint_file:
            raise ValueError("enabled external waypoint motion requires waypoint_file")
        if self.source_frame != "map":
            raise ValueError("external waypoint source_frame must be 'map'")
        if self.target_frame != "factory_v3_world_xy_m":
            raise ValueError(
                "external waypoint target_frame must be factory_v3_world_xy_m"
            )
        values = (
            self.simulation_to_map_translation_x_m,
            self.simulation_to_map_translation_y_m,
            self.simulation_to_map_rotation_deg,
        )
        if any(isinstance(value, bool) or not math.isfinite(float(value))
               for value in values):
            raise ValueError("external waypoint transform values must be finite")

    @classmethod
    def from_mapping(cls, values) -> "ExternalWaypointMotionConfig":
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"unknown external_waypoint_motion settings: {sorted(unknown)}"
            )
        return cls(**values)

    def resolve_waypoint_file(self, config_directory: Path) -> Path:
        path = Path(str(self.waypoint_file)).expanduser()
        if not path.is_absolute():
            path = config_directory / path
        path = path.resolve(strict=False)
        if not path.is_file():
            raise ValueError(f"external waypoint file does not exist: {path}")
        return path


def _finite(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _map_to_simulation(x_map: float, y_map: float,
                       config: ExternalWaypointMotionConfig):
    """Invert the measured factory_v3-world -> ROS-map rigid transform."""
    dx = x_map - float(config.simulation_to_map_translation_x_m)
    dy = y_map - float(config.simulation_to_map_translation_y_m)
    angle = math.radians(float(config.simulation_to_map_rotation_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def load_external_waypoints(
    path: Path, config: ExternalWaypointMotionConfig,
) -> tuple[tuple[float, float], ...]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read external waypoint YAML {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("external waypoint YAML root must be a mapping")
    frame = document.get("frame_id")
    if frame is None and isinstance(document.get("header"), Mapping):
        frame = document["header"].get("frame_id")
    if frame != config.source_frame:
        raise ValueError(
            f"external waypoint frame must be {config.source_frame!r}, got {frame!r}"
        )
    poses = document.get("poses")
    if not isinstance(poses, list) or not poses:
        raise ValueError("external waypoint YAML requires non-empty poses")
    output = []
    for index, entry in enumerate(poses):
        try:
            pose = entry["pose"]
            position = pose["position"]
            orientation = pose["orientation"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"external waypoint {index} is malformed") from exc
        x_map = _finite(position.get("x"), f"waypoint {index} x")
        y_map = _finite(position.get("y"), f"waypoint {index} y")
        quaternion = tuple(
            _finite(orientation.get(name), f"waypoint {index} orientation.{name}")
            for name in ("x", "y", "z", "w")
        )
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise ValueError(f"external waypoint {index} quaternion is not normalized")
        output.append(_map_to_simulation(x_map, y_map, config))
    return tuple(output)


class ExternalWaypointFollower:
    """Motion-only waypoint replay; planning safety remains a separate layer."""

    def __init__(self, points_world, *, speed_mps: float,
                 angular_speed_rad_s: float, tolerance_m: float):
        self.world_path = tuple(
            (float(point[0]), float(point[1])) for point in points_world
        )
        if not self.world_path:
            raise ValueError("external motion path cannot be empty")
        values = (speed_mps, angular_speed_rad_s, tolerance_m)
        if any(not math.isfinite(float(value)) or float(value) <= 0.0
               for value in values):
            raise ValueError("external motion speed, turn rate and tolerance must be positive")
        self.speed_mps = float(speed_mps)
        self.angular_speed_rad_s = float(angular_speed_rad_s)
        self.tolerance_m = float(tolerance_m)
        self.waypoint_index = 0

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def update(self, state, dt: float) -> tuple[float, str]:
        if self.waypoint_index >= len(self.world_path):
            return 0.0, "complete"
        target_x, target_y = self.world_path[self.waypoint_index]
        dx, dy = target_x - state.x, target_y - state.y
        distance = math.hypot(dx, dy)
        if distance <= self.tolerance_m:
            state.x, state.y = target_x, target_y
            self.waypoint_index += 1
            return distance, "waypoint reached"
        target_yaw = math.atan2(dy, dx)
        error = self._wrap(target_yaw - state.theta)
        max_turn = self.angular_speed_rad_s * float(dt)
        if abs(error) > math.radians(1.0):
            state.theta = self._wrap(
                state.theta + max(-max_turn, min(max_turn, error))
            )
            return 0.0, "turning"
        state.theta = target_yaw
        step = min(self.speed_mps * float(dt), distance)
        state.x += math.cos(state.theta) * step
        state.y += math.sin(state.theta) * step
        if step >= distance - 1e-12:
            state.x, state.y = target_x, target_y
            self.waypoint_index += 1
        return step, "moving"
