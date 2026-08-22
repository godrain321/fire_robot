"""Convert a reference-graph route into sequential YAML waypoint targets."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from planner.a_star import weighted_a_star


@dataclass(frozen=True)
class ReferenceWaypointExecutionConfig:
    enabled: bool = True
    direction_change_threshold_deg: float = 12.0
    waypoint_turn_threshold_deg: float = 50.0
    fallback_to_simplified_path: bool = True

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not isinstance(
            self.fallback_to_simplified_path, bool
        ):
            raise TypeError("reference waypoint execution flags must be bool")
        for name in (
            "direction_change_threshold_deg", "waypoint_turn_threshold_deg"
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 180.0:
                raise ValueError(f"{name} must be in [0,180]")

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
    """Extract graph turns, then join each consecutive target with weighted A*."""
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
        anchors, config.waypoint_turn_threshold_deg
    )
    original = tuple(original_path_grid)
    targets = [original[0]]
    target_worlds = [tuple(simplified_result.waypoints_world[0])]
    for waypoint_id in turn_ids:
        waypoint = waypoint_by_id[waypoint_id]
        if waypoint.factory_grid != targets[-1]:
            targets.append(waypoint.factory_grid)
            target_worlds.append(tuple(waypoint.factory_world))
    if original[-1] != targets[-1]:
        targets.append(original[-1])
        target_worlds.append(tuple(simplified_result.waypoints_world[-1]))

    effective = np.asarray(costmap, dtype=float).copy()
    effective[np.asarray(static_obstacle_map, dtype=bool)] = np.inf
    effective[np.asarray(dynamic_obstacle_map, dtype=bool)] = np.inf
    effective[np.asarray(estimated_fire_map.blocked_mask, dtype=bool)] = np.inf
    ordered_list = []
    worlds_list = []
    for index, (start, end) in enumerate(zip(targets, targets[1:])):
        segment = weighted_a_star(effective, start, end)
        if not segment.path:
            reason = f"corner_astar_failed:{segment.reason}"
            if config.fallback_to_simplified_path:
                return ReferenceExecutionPath(
                    False, tuple(), fallback.grid_path, fallback.world_path, reason,
                )
            return ReferenceExecutionPath(False, tuple(), tuple(), tuple(), reason)
        simplified_segment = path_simplifier.simplify(
            segment.path, costmap=costmap,
            static_obstacle_map=static_obstacle_map,
            dynamic_obstacle_map=dynamic_obstacle_map,
            estimated_fire_map=estimated_fire_map,
            start_world=target_worlds[index],
            goal_world=target_worlds[index + 1],
        )
        if not simplified_segment.success:
            reason = f"corner_astar_unsafe:{simplified_segment.failure_reason}"
            if config.fallback_to_simplified_path:
                return ReferenceExecutionPath(
                    False, tuple(), fallback.grid_path, fallback.world_path, reason,
                )
            return ReferenceExecutionPath(False, tuple(), tuple(), tuple(), reason)
        for cell, world in zip(
            simplified_segment.simplified_path_grid,
            simplified_segment.waypoints_world,
        ):
            if not ordered_list or ordered_list[-1] != cell:
                ordered_list.append(cell)
                worlds_list.append(world)
    ordered = tuple(ordered_list)
    worlds = tuple(worlds_list)
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
