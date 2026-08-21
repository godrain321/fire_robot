"""Load measured ROS-map waypoints in the rotated factory_v5 world frame."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SlamReferenceWaypointConfig:
    enabled: bool = False
    waypoint_file: str = "config/slam_reference_waypoints.yaml"
    map_metadata_file: str = "config/map_metadata.yaml"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("slam_reference_waypoints.enabled must be bool")
        if not isinstance(self.waypoint_file, str) or not self.waypoint_file.strip():
            raise ValueError("slam reference waypoint_file must be non-empty")
        if not isinstance(self.map_metadata_file, str) or not self.map_metadata_file.strip():
            raise ValueError("slam reference map_metadata_file must be non-empty")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"unknown slam_reference_waypoints settings: {sorted(unknown)}"
            )
        return cls(**values)


@dataclass(frozen=True)
class SlamReferenceWaypoint:
    waypoint_id: str
    ros_map_world: tuple[float, float]
    factory_world: tuple[float, float]
    factory_grid: tuple[int, int]
    ros_yaw: float
    factory_yaw: float


@dataclass(frozen=True)
class RosFactoryTransform:
    ros_origin: tuple[float, float]
    rotation_rad: float
    pivot_fds_v2: tuple[float, float]
    translation_raw_min: tuple[float, float]

    @classmethod
    def from_mapping(cls, values):
        origin = tuple(float(item) for item in values["ros_origin"][:2])
        transform = values["factory_v3_transform"]
        return cls(
            origin,
            math.radians(float(transform["rotation_angle_deg"])),
            tuple(float(item) for item in transform["pivot_fds_v2"]),
            tuple(float(item) for item in transform["translation_raw_min"]),
        )

    def ros_to_factory(self, x: float, y: float) -> tuple[float, float]:
        source_x = float(x) - self.ros_origin[0]
        source_y = float(y) - self.ros_origin[1]
        local_x = source_x - self.pivot_fds_v2[0]
        local_y = source_y - self.pivot_fds_v2[1]
        cosine = math.cos(self.rotation_rad)
        sine = math.sin(self.rotation_rad)
        return (
            cosine * local_x - sine * local_y - self.translation_raw_min[0],
            sine * local_x + cosine * local_y - self.translation_raw_min[1],
        )

    def factory_to_ros(self, x: float, y: float) -> tuple[float, float]:
        rotated_x = float(x) + self.translation_raw_min[0]
        rotated_y = float(y) + self.translation_raw_min[1]
        cosine = math.cos(self.rotation_rad)
        sine = math.sin(self.rotation_rad)
        source_x = cosine * rotated_x + sine * rotated_y + self.pivot_fds_v2[0]
        source_y = -sine * rotated_x + cosine * rotated_y + self.pivot_fds_v2[1]
        return source_x + self.ros_origin[0], source_y + self.ros_origin[1]

    def ros_yaw_to_factory(self, yaw: float) -> float:
        value = float(yaw) + self.rotation_rad
        return math.atan2(math.sin(value), math.cos(value))


def load_slam_reference_waypoints(
    waypoint_file: Path, map_metadata_file: Path, planner_metadata,
) -> tuple[tuple[SlamReferenceWaypoint, ...], RosFactoryTransform]:
    """Load, transform and grid-index measured map-frame reference points."""
    waypoint_file = Path(waypoint_file)
    map_metadata_file = Path(map_metadata_file)
    document = yaml.safe_load(waypoint_file.read_text(encoding="utf-8"))
    if document.get("frame_id") != "map":
        raise ValueError("SLAM reference waypoints must use frame_id 'map'")
    poses = document.get("poses")
    if not isinstance(poses, dict) or len(poses) < 2:
        raise ValueError("SLAM reference poses must be a mapping with at least two points")
    metadata_values = yaml.safe_load(map_metadata_file.read_text(encoding="utf-8"))
    transform = RosFactoryTransform.from_mapping(metadata_values)
    output = []
    occupied_grids = set()
    for waypoint_id, pose in poses.items():
        if not isinstance(pose, dict):
            raise ValueError(f"invalid reference waypoint {waypoint_id}")
        values = tuple(float(pose[name]) for name in ("x", "y", "yaw"))
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f"non-finite reference waypoint {waypoint_id}")
        factory = transform.ros_to_factory(values[0], values[1])
        grid = planner_metadata.world_to_grid(*factory)
        if grid in occupied_grids:
            raise ValueError(f"reference waypoints collapse onto grid cell {grid}")
        occupied_grids.add(grid)
        output.append(SlamReferenceWaypoint(
            str(waypoint_id), (values[0], values[1]), factory, grid, values[2],
            transform.ros_yaw_to_factory(values[2]),
        ))
    return tuple(output), transform
