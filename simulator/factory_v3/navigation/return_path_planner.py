"""Validate reverse travel history against current robot belief only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any

import numpy as np


class ReturnFailureReason(Enum):
    EMPTY_HISTORY = "empty_history"
    INSUFFICIENT_HISTORY = "insufficient_history"
    CURRENT_POSITION_INVALID = "current_position_invalid"
    PATH_OUT_OF_BOUNDS = "path_out_of_bounds"
    BLOCKED_BY_STATIC_OBSTACLE = "blocked_by_static_obstacle"
    BLOCKED_BY_DYNAMIC_OBSTACLE = "blocked_by_dynamic_obstacle"
    BLOCKED_BY_FIRE_RISK = "blocked_by_fire_risk"
    UNOBSERVED_OR_STALE = "unobserved_or_stale"
    INVALID_COST = "invalid_cost"
    NO_SAFE_RETURN_PATH = "no_safe_return_path"
    PATH_DEVIATION = "path_deviation"


@dataclass(frozen=True)
class ReturnPathConfig:
    enabled: bool = True
    validate_before_start: bool = True
    validate_during_return: bool = True
    lookahead_distance_m: float = 1.0
    path_deviation_tolerance_m: float = 0.5
    prefer_normal_planner: bool = True
    allow_history_fallback: bool = True
    max_observation_age_s: float | None = 10.0

    def __post_init__(self) -> None:
        if self.lookahead_distance_m <= 0 or self.path_deviation_tolerance_m <= 0:
            raise ValueError("return path distances must be positive")
        if self.max_observation_age_s is not None and self.max_observation_age_s <= 0:
            raise ValueError("max_observation_age_s must be positive or null")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "ReturnPathConfig":
        values = dict(values or {})
        allowed = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown return_path settings: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class ReturnPathPlan:
    success: bool
    path_world: tuple[tuple[float, float], ...]
    path_grid: tuple[tuple[int, int], ...]
    source_point_count: int
    output_point_count: int
    blocked_segment_index: int | None
    failure_reason: ReturnFailureReason | None
    created_at: float
    blocked_grid: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path_world"] = [list(item) for item in self.path_world]
        result["path_grid"] = [list(item) for item in self.path_grid]
        result["blocked_grid"] = None if self.blocked_grid is None else list(self.blocked_grid)
        result["failure_reason"] = None if self.failure_reason is None else self.failure_reason.value
        return result


def _line_cells(start: tuple[int, int], end: tuple[int, int]):
    """Bresenham cells in explicit ``(col,row)`` order, including endpoints."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx - dy
    while True:
        yield x0, y0
        if (x0, y0) == (x1, y1):
            break
        twice = 2 * error
        if twice > -dy:
            error -= dy
            x0 += sx
        if twice < dx:
            error += dx
            y0 += sy


class ReturnPathPlanner:
    def __init__(self, map_metadata, config: ReturnPathConfig | None = None) -> None:
        self.map_metadata = map_metadata
        self.config = config or ReturnPathConfig()

    def _cell_failure(
        self, cell, cost_map, static_map, dynamic_map, estimated_map,
        *, current_time: float,
    ) -> ReturnFailureReason | None:
        col, row = cell
        if not self.map_metadata.is_grid_position_in_bounds(col, row):
            return ReturnFailureReason.PATH_OUT_OF_BOUNDS
        if static_map[row, col]:
            return ReturnFailureReason.BLOCKED_BY_STATIC_OBSTACLE
        if dynamic_map[row, col]:
            return ReturnFailureReason.BLOCKED_BY_DYNAMIC_OBSTACLE
        if estimated_map.blocked_mask[row, col]:
            return ReturnFailureReason.BLOCKED_BY_FIRE_RISK
        if not estimated_map.observed_mask[row, col]:
            return ReturnFailureReason.UNOBSERVED_OR_STALE
        observed_at = estimated_map.last_observed_time[row, col]
        if (
            self.config.max_observation_age_s is not None
            and (not np.isfinite(observed_at)
                 or current_time - observed_at > self.config.max_observation_age_s)
        ):
            return ReturnFailureReason.UNOBSERVED_OR_STALE
        if not np.isfinite(cost_map[row, col]):
            return ReturnFailureReason.INVALID_COST
        return None

    def _validate_grid_path(
        self, path_grid, cost_map, static_map, dynamic_map, estimated_map,
        *, current_time: float,
    ):
        for index, (start, end) in enumerate(zip(path_grid, path_grid[1:])):
            for cell in _line_cells(start, end):
                failure = self._cell_failure(
                    cell, cost_map, static_map, dynamic_map, estimated_map,
                    current_time=current_time,
                )
                if failure is not None:
                    return index, cell, failure
        if len(path_grid) == 1:
            failure = self._cell_failure(
                path_grid[0], cost_map, static_map, dynamic_map, estimated_map,
                current_time=current_time,
            )
            if failure is not None:
                return 0, path_grid[0], failure
        return None

    @staticmethod
    def _collinear(a, b, c) -> bool:
        first = (b[0] - a[0], b[1] - a[1])
        second = (c[0] - b[0], c[1] - b[1])
        return first[0] * second[1] == first[1] * second[0] and (
            first[0] * second[0] + first[1] * second[1] >= 0
        )

    def _simplify(self, path_grid):
        if len(path_grid) <= 2:
            return list(path_grid)
        output = [path_grid[0]]
        for index in range(1, len(path_grid) - 1):
            if not self._collinear(output[-1], path_grid[index], path_grid[index + 1]):
                output.append(path_grid[index])
        output.append(path_grid[-1])
        return output

    def create_plan(
        self, travel_history, current_position_world, *, cost_map,
        static_obstacle_map, dynamic_obstacle_map, estimated_fire_map,
        created_at: float,
    ) -> ReturnPathPlan:
        points = travel_history.get_points()
        if not points:
            return self._failure(ReturnFailureReason.EMPTY_HISTORY, 0, created_at)
        try:
            current_world = (float(current_position_world[0]), float(current_position_world[1]))
            current_grid = self.map_metadata.world_to_grid(*current_world)
        except (ValueError, TypeError, IndexError):
            return self._failure(ReturnFailureReason.CURRENT_POSITION_INVALID, len(points), created_at)
        reverse_grid = [item.position_grid for item in reversed(points)]
        reverse_world = [item.position_world for item in reversed(points)]
        if reverse_grid and reverse_grid[0] == current_grid:
            reverse_grid[0] = current_grid
            reverse_world[0] = current_world
        else:
            reverse_grid.insert(0, current_grid)
            reverse_world.insert(0, current_world)
        dedup_grid, dedup_world = [], []
        for grid, world in zip(reverse_grid, reverse_world):
            if not dedup_grid or grid != dedup_grid[-1]:
                dedup_grid.append(grid)
                dedup_world.append(world)
        if len(points) == 1 and current_grid != points[0].position_grid:
            return self._failure(ReturnFailureReason.INSUFFICIENT_HISTORY, 1, created_at)
        if travel_history.config.simplify_reverse_path:
            simplified_grid = self._simplify(dedup_grid)
            world_by_grid = {grid: world for grid, world in zip(dedup_grid, dedup_world)}
            simplified_world = [world_by_grid[grid] for grid in simplified_grid]
        else:
            simplified_grid, simplified_world = dedup_grid, dedup_world
        failure = self._validate_grid_path(
            simplified_grid, np.asarray(cost_map), np.asarray(static_obstacle_map),
            np.asarray(dynamic_obstacle_map), estimated_fire_map,
            current_time=float(created_at),
        )
        if failure:
            index, cell, reason = failure
            return ReturnPathPlan(
                False, tuple(), tuple(), len(points), len(simplified_grid),
                index, reason, float(created_at), cell,
            )
        return ReturnPathPlan(
            True, tuple(simplified_world), tuple(simplified_grid), len(points),
            len(simplified_grid), None, None, float(created_at), None,
        )

    def validate_plan(
        self, plan: ReturnPathPlan, *, cost_map, static_obstacle_map,
        dynamic_obstacle_map, estimated_fire_map, current_time: float,
        start_index: int = 0, current_position_world=None,
    ) -> ReturnPathPlan:
        if not plan.success:
            return plan
        remaining_grid = plan.path_grid[start_index:]
        if current_position_world is not None and len(remaining_grid) >= 1:
            remaining_world = plan.path_world[start_index:]
            if self._distance_to_polyline(current_position_world, remaining_world) > self.config.path_deviation_tolerance_m:
                return ReturnPathPlan(
                    False, tuple(), tuple(), plan.source_point_count,
                    len(remaining_grid), start_index,
                    ReturnFailureReason.PATH_DEVIATION, float(current_time),
                    remaining_grid[0],
                )
        lookahead_cells = max(
            1, int(math.ceil(
                self.config.lookahead_distance_m / self.map_metadata.resolution_m
            ))
        )
        expanded = []
        for start, end in zip(remaining_grid, remaining_grid[1:]):
            for cell in _line_cells(start, end):
                if not expanded or cell != expanded[-1]:
                    expanded.append(cell)
        if len(remaining_grid) == 1:
            expanded = [remaining_grid[0]]
        validation_grid = expanded[:lookahead_cells + 1]
        failure = self._validate_grid_path(
            validation_grid, np.asarray(cost_map), np.asarray(static_obstacle_map),
            np.asarray(dynamic_obstacle_map), estimated_fire_map,
            current_time=float(current_time),
        )
        if not failure:
            return plan
        index, cell, reason = failure
        return ReturnPathPlan(
            False, tuple(), tuple(), plan.source_point_count,
            len(remaining_grid), start_index + index, reason,
            float(current_time), cell,
        )

    @staticmethod
    def _distance_to_polyline(point, path_world) -> float:
        px, py = float(point[0]), float(point[1])
        if len(path_world) == 1:
            return math.hypot(px - path_world[0][0], py - path_world[0][1])
        best = math.inf
        for start, end in zip(path_world, path_world[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length_squared = dx * dx + dy * dy
            ratio = 0.0 if length_squared == 0 else max(
                0.0, min(1.0, ((px - start[0]) * dx + (py - start[1]) * dy) / length_squared)
            )
            nearest = (start[0] + ratio * dx, start[1] + ratio * dy)
            best = min(best, math.hypot(px - nearest[0], py - nearest[1]))
        return best

    @staticmethod
    def _failure(reason, count, created_at):
        return ReturnPathPlan(
            False, tuple(), tuple(), count, 0, None, reason,
            float(created_at), None,
        )
