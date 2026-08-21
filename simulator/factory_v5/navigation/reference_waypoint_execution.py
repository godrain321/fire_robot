"""Convert a reference-graph route into sequential YAML waypoint targets."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ReferenceWaypointExecutionConfig:
    enabled: bool = True
    direction_change_threshold_deg: float = 12.0
    fallback_to_simplified_path: bool = True

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not isinstance(
            self.fallback_to_simplified_path, bool
        ):
            raise TypeError("reference waypoint execution flags must be bool")
        value = float(self.direction_change_threshold_deg)
        if not math.isfinite(value) or not 0.0 <= value <= 180.0:
            raise ValueError("direction_change_threshold_deg must be in [0,180]")

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"unknown reference_waypoint_execution settings: {sorted(unknown)}"
            )
        return cls(**values)


@dataclass(frozen=True)
class ReferenceExecutionPath:
    used_reference_targets: bool
    reference_waypoint_ids: tuple[str, ...]
    grid_path: tuple[tuple[int, int], ...]
    world_path: tuple[tuple[float, float], ...]
    fallback_reason: str | None = None


def extract_reference_turn_ids(waypoints, threshold_deg: float):
    """Keep the first/last YAML point and geometrically meaningful turns."""
    points = tuple(waypoints)
    if len(points) <= 2:
        return tuple(item.waypoint_id for item in points)
    output = [points[0].waypoint_id]
    threshold = math.radians(float(threshold_deg))
    for previous, current, following in zip(points, points[1:], points[2:]):
        first = (
            current.factory_world[0] - previous.factory_world[0],
            current.factory_world[1] - previous.factory_world[1],
        )
        second = (
            following.factory_world[0] - current.factory_world[0],
            following.factory_world[1] - current.factory_world[1],
        )
        if math.hypot(*first) <= 1e-12 or math.hypot(*second) <= 1e-12:
            continue
        first_yaw = math.atan2(first[1], first[0])
        second_yaw = math.atan2(second[1], second[0])
        change = abs(math.atan2(
            math.sin(second_yaw - first_yaw),
            math.cos(second_yaw - first_yaw),
        ))
        if change + 1e-12 >= threshold:
            output.append(current.waypoint_id)
    output.append(points[-1].waypoint_id)
    return tuple(dict.fromkeys(output))


def build_reference_execution_path(
    *, original_path_grid, simplified_result, reference_waypoint_ids,
    waypoint_by_id, config, path_simplifier, costmap,
    static_obstacle_map, dynamic_obstacle_map, estimated_fire_map,
):
    """Insert required YAML turns and validate the complete execution route."""
    fallback = ReferenceExecutionPath(
        False, tuple(), tuple(simplified_result.simplified_path_grid),
        tuple(simplified_result.waypoints_world), None,
    )
    ids = tuple(reference_waypoint_ids)
    if not config.enabled or not ids:
        return fallback
    try:
        anchors = tuple(waypoint_by_id[item] for item in ids)
    except KeyError as exc:
        return ReferenceExecutionPath(
            False, tuple(), fallback.grid_path, fallback.world_path,
            f"unknown_reference_waypoint:{exc.args[0]}",
        )
    turn_ids = extract_reference_turn_ids(
        anchors, config.direction_change_threshold_deg
    )
    mandatory = {waypoint_by_id[item].factory_grid: waypoint_by_id[item]
                 for item in turn_ids}
    original = tuple(original_path_grid)
    positions = {cell: index for index, cell in enumerate(original)}
    if any(cell not in positions for cell in mandatory):
        return ReferenceExecutionPath(
            False, tuple(), fallback.grid_path, fallback.world_path,
            "reference_waypoint_not_on_original_path",
        )
    first_index = min(positions[cell] for cell in mandatory)
    last_index = max(positions[cell] for cell in mandatory)
    selected = {}
    simplified_world = dict(zip(
        simplified_result.simplified_path_grid,
        simplified_result.waypoints_world,
    ))
    for cell in simplified_result.simplified_path_grid:
        index = positions.get(cell)
        if index is not None and (index < first_index or index > last_index):
            selected[cell] = simplified_world[cell]
    for cell, waypoint in mandatory.items():
        selected[cell] = waypoint.factory_world
    selected[original[0]] = simplified_result.waypoints_world[0]
    selected[original[-1]] = simplified_result.waypoints_world[-1]
    ordered = tuple(sorted(selected, key=lambda cell: positions[cell]))
    worlds = tuple(selected[cell] for cell in ordered)
    validation = path_simplifier.validate_path(
        ordered, costmap=costmap, static_obstacle_map=static_obstacle_map,
        dynamic_obstacle_map=dynamic_obstacle_map,
        estimated_fire_map=estimated_fire_map,
    )
    if not validation.safe:
        reason = validation.rejection_reasons[0].value
        if config.fallback_to_simplified_path:
            return ReferenceExecutionPath(
                False, tuple(), fallback.grid_path, fallback.world_path,
                f"reference_execution_unsafe:{reason}",
            )
        return ReferenceExecutionPath(False, tuple(), tuple(), tuple(), reason)
    return ReferenceExecutionPath(True, turn_ids, ordered, worlds, None)
