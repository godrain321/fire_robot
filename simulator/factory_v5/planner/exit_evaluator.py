"""Evaluate one exit using only Estimated Fire Map and partial costmap data.

World coordinates are ``(x,y)`` metres, grid nodes are ``(col,row)``, and all
NumPy layers are indexed ``[row,col] == [y,x]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any

import numpy as np

from planner.a_star import weighted_a_star
from world.entities import ExitStatus


class ExitRejectionReason(Enum):
    EXIT_BLOCKED = "exit_blocked"
    EXIT_DANGEROUS = "exit_dangerous"
    EXIT_DANGER_EXPECTED = "exit_danger_expected"
    INVALID_EXIT_POSITION = "invalid_exit_position"
    NO_APPROACH_CELL = "no_approach_cell"
    NO_PATH = "no_path"
    STATIC_OBSTACLE = "static_obstacle"
    DYNAMIC_OBSTACLE = "dynamic_obstacle"
    TEMPERATURE_LIMIT_EXCEEDED = "temperature_limit_exceeded"
    CO_LIMIT_EXCEEDED = "co_limit_exceeded"
    PATH_RISK_COST_EXCEEDED = "path_risk_cost_exceeded"
    INVALID_COST = "invalid_cost"
    OUT_OF_MAP = "out_of_map"


@dataclass(frozen=True)
class ExitEvaluationConfig:
    exit_neighborhood_radius_m: float = 1.0
    approach_search_radius_m: float = 1.0
    reject_blocked_exit: bool = True
    reject_dangerous_exit: bool = True
    reject_path_over_threshold: bool = True
    reject_invalid_cost: bool = True
    usable_confirmation_distance_m: float = 3.0
    dangerous_accumulated_risk_cost: float | None = None
    dangerous_average_risk_cost: float | None = None
    dangerous_max_cell_risk_cost: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "reject_blocked_exit", "reject_dangerous_exit",
            "reject_path_over_threshold", "reject_invalid_cost",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in (
            "exit_neighborhood_radius_m", "approach_search_radius_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
        if self.exit_neighborhood_radius_m < 0:
            raise ValueError("exit_neighborhood_radius_m must be non-negative")
        if self.approach_search_radius_m <= 0:
            raise ValueError("approach_search_radius_m must be positive")
        if (
            isinstance(self.usable_confirmation_distance_m, bool)
            or not math.isfinite(float(self.usable_confirmation_distance_m))
            or self.usable_confirmation_distance_m <= 0.0
        ):
            raise ValueError(
                "usable_confirmation_distance_m must be finite and positive"
            )
        for name in (
            "dangerous_accumulated_risk_cost",
            "dangerous_average_risk_cost",
            "dangerous_max_cell_risk_cost",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be a finite positive number or null")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "ExitEvaluationConfig":
        values = dict(values or {})
        # Thresholds belong to PartialCostmapConfig and must not be duplicated.
        forbidden = {"temperature_block_threshold_c", "co_block_threshold_ppm"}
        if forbidden & set(values):
            raise ValueError("fire thresholds must come from PartialCostmapConfig")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown exit_evaluation settings: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class ExitEvaluation:
    exit_id: str
    exit_status: str
    exit_position_world: tuple[float, float]
    approach_position_world: tuple[float, float] | None
    approach_position_grid: tuple[int, int] | None
    reachable: bool
    accepted: bool
    path_world: tuple[tuple[float, float], ...]
    path_grid: tuple[tuple[int, int], ...]
    path_length_m: float | None
    accumulated_risk_cost: float | None
    max_path_temperature_c: float | None
    max_path_co_ppm: float | None
    exit_temperature_c: float | None
    exit_co_ppm: float | None
    unknown_ratio: float | None
    rejection_reasons: tuple[ExitRejectionReason, ...]
    evaluated_at: float
    reference_waypoint_ids: tuple[str, ...] = tuple()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["exit_position_world"] = list(self.exit_position_world)
        result["approach_position_world"] = (
            None if self.approach_position_world is None
            else list(self.approach_position_world)
        )
        result["approach_position_grid"] = (
            None if self.approach_position_grid is None
            else list(self.approach_position_grid)
        )
        result["path_world"] = [list(item) for item in self.path_world]
        result["path_grid"] = [list(item) for item in self.path_grid]
        result["rejection_reasons"] = [item.value for item in self.rejection_reasons]
        return result


def within_usable_confirmation_distance(
    robot_position_world, exit_approach_world, config: ExitEvaluationConfig,
) -> bool:
    """Return whether the robot is close enough for direct USABLE confirmation."""
    robot = tuple(float(value) for value in robot_position_world)
    approach = tuple(float(value) for value in exit_approach_world)
    if len(robot) != 2 or len(approach) != 2 or not all(
        math.isfinite(value) for value in (*robot, *approach)
    ):
        raise ValueError("confirmation positions must be finite (x, y) points")
    return math.dist(robot, approach) <= (
        config.usable_confirmation_distance_m + 1e-12
    )


class ExitEvaluator:
    def __init__(
        self, map_metadata, config: ExitEvaluationConfig, *,
        temperature_blocked_c: float, co_blocked_ppm: float,
        base_cost: float, path_planner=None,
    ) -> None:
        self.metadata = map_metadata
        self.config = config
        self.temperature_blocked_c = float(temperature_blocked_c)
        self.co_blocked_ppm = float(co_blocked_ppm)
        self.base_cost = float(base_cost)
        self.path_planner = path_planner or weighted_a_star
        if self.temperature_blocked_c < 0 or self.co_blocked_ppm < 0:
            raise ValueError("fire thresholds must be non-negative")
        if self.base_cost <= 0:
            raise ValueError("base_cost must be positive")

    def evaluate(
        self, exit_item, start_position_world, *, cost_map,
        static_obstacle_map, dynamic_obstacle_map, estimated_fire_map,
        evaluated_at: float,
    ) -> ExitEvaluation:
        reasons: list[ExitRejectionReason] = []
        if exit_item.status is ExitStatus.BLOCKED and self.config.reject_blocked_exit:
            reasons.append(ExitRejectionReason.EXIT_BLOCKED)
        if exit_item.status is ExitStatus.DANGEROUS and self.config.reject_dangerous_exit:
            reasons.append(ExitRejectionReason.EXIT_DANGEROUS)
        if (
            exit_item.status is ExitStatus.DANGER_EXPECTED
            and self.config.reject_dangerous_exit
        ):
            reasons.append(ExitRejectionReason.EXIT_DANGER_EXPECTED)
        try:
            self.metadata.world_to_grid(*exit_item.position_world)
        except ValueError:
            reasons.append(ExitRejectionReason.OUT_OF_MAP)
        if reasons:
            return self._rejected(exit_item, reasons, evaluated_at)

        arrays = self._validate_arrays(
            cost_map, static_obstacle_map, dynamic_obstacle_map,
            estimated_fire_map,
        )
        costs, static, dynamic = arrays
        planning_costs = costs.copy()
        planning_costs[static | dynamic | estimated_fire_map.blocked_mask] = np.inf
        try:
            start_grid = self.metadata.world_to_grid(*start_position_world)
        except ValueError:
            return self._rejected(
                exit_item, [ExitRejectionReason.OUT_OF_MAP], evaluated_at
            )
        start_failure = self._cell_failure(start_grid, planning_costs, static, dynamic, estimated_fire_map)
        if start_failure is not None:
            return self._rejected(exit_item, [start_failure], evaluated_at)

        if exit_item.approach_position_world is not None:
            try:
                registered_grid = self.metadata.world_to_grid(
                    *exit_item.approach_position_world
                )
            except ValueError:
                return self._rejected(
                    exit_item, [ExitRejectionReason.OUT_OF_MAP], evaluated_at
                )
            registered_failure = self._cell_failure(
                registered_grid, costs, static, dynamic, estimated_fire_map
            )
            if registered_failure is not None:
                return self._rejected(
                    exit_item, [registered_failure], evaluated_at,
                    tuple(exit_item.approach_position_world), registered_grid,
                )
        approach = self._resolve_approach(
            exit_item, start_grid, planning_costs, static, dynamic,
            estimated_fire_map
        )
        if approach is None:
            return self._rejected(
                exit_item, [ExitRejectionReason.NO_APPROACH_CELL], evaluated_at
            )
        approach_world, approach_grid, astar_result = approach
        if not astar_result.path:
            return self._rejected(
                exit_item, [ExitRejectionReason.NO_PATH], evaluated_at,
                approach_world, approach_grid,
            )

        path_grid = tuple(astar_result.path)
        path_world = tuple(self.metadata.grid_to_world(*item) for item in path_grid)
        path_length = self._path_length(path_grid)
        risk_cost = self._risk_cost(path_grid, planning_costs)
        average_risk_cost = (
            risk_cost / path_length
            if path_length > 1e-12 else
            self._max_cell_risk_cost(path_grid, planning_costs)
        )
        max_cell_risk_cost = self._max_cell_risk_cost(
            path_grid, planning_costs
        )
        path_temp = self._finite_max(estimated_fire_map.temperature_c, path_grid)
        path_co = self._finite_max(estimated_fire_map.co_ppm, path_grid)
        observed = np.asarray([
            estimated_fire_map.observed_mask[row, col] for col, row in path_grid
        ], dtype=bool)
        unknown_ratio = float((~observed).sum() / len(path_grid))
        exit_temp, exit_co = self._exit_neighborhood_values(
            exit_item.position_world, estimated_fire_map
        )

        if not math.isfinite(risk_cost) and self.config.reject_invalid_cost:
            reasons.append(ExitRejectionReason.INVALID_COST)
        elif self._risk_threshold_exceeded(
            risk_cost, average_risk_cost, max_cell_risk_cost
        ):
            reasons.append(ExitRejectionReason.PATH_RISK_COST_EXCEEDED)
        if self.config.reject_path_over_threshold:
            if (
                (path_temp is not None and path_temp >= self.temperature_blocked_c)
                or (exit_temp is not None and exit_temp >= self.temperature_blocked_c)
            ):
                reasons.append(ExitRejectionReason.TEMPERATURE_LIMIT_EXCEEDED)
            if (
                (path_co is not None and path_co >= self.co_blocked_ppm)
                or (exit_co is not None and exit_co >= self.co_blocked_ppm)
            ):
                reasons.append(ExitRejectionReason.CO_LIMIT_EXCEEDED)
        return ExitEvaluation(
            exit_item.exit_id, exit_item.status.value,
            tuple(exit_item.position_world), approach_world, approach_grid,
            True, not reasons, path_world, path_grid, path_length, risk_cost,
            path_temp, path_co, exit_temp, exit_co, unknown_ratio,
            tuple(dict.fromkeys(reasons)), float(evaluated_at),
            tuple(astar_result.reference_waypoint_ids),
        )

    def _validate_arrays(self, cost_map, static, dynamic, estimated):
        expected = (self.metadata.height, self.metadata.width)
        arrays = (
            np.asarray(cost_map, dtype=float), np.asarray(static, dtype=bool),
            np.asarray(dynamic, dtype=bool),
        )
        if any(item.shape != expected for item in arrays):
            raise ValueError(f"exit evaluation map shape must be {expected}")
        if estimated.shape != expected:
            raise ValueError(f"estimated fire map shape must be {expected}")
        return arrays

    def _resolve_approach(self, exit_item, start, costs, static, dynamic, estimated):
        candidates = []
        registered_approach = exit_item.approach_position_world is not None
        if registered_approach:
            try:
                grid = self.metadata.world_to_grid(*exit_item.approach_position_world)
            except ValueError:
                return None
            candidates.append((tuple(exit_item.approach_position_world), grid))
        else:
            try:
                center = self.metadata.world_to_grid(*exit_item.position_world)
            except ValueError:
                return None
            radius = max(1, int(math.ceil(
                self.config.approach_search_radius_m / self.metadata.resolution_m
            )))
            for row in range(center[1] - radius, center[1] + radius + 1):
                for col in range(center[0] - radius, center[0] + radius + 1):
                    if not self.metadata.is_grid_position_in_bounds(col, row):
                        continue
                    distance = math.hypot(col - center[0], row - center[1])
                    if 0 < distance <= radius:
                        candidates.append((self.metadata.grid_to_world(col, row), (col, row)))
            candidates.sort(key=lambda item: (
                math.hypot(item[1][0] - center[0], item[1][1] - center[1]),
                item[1][1], item[1][0],
            ))
        best = None
        for world, grid in candidates:
            if self._cell_failure(grid, costs, static, dynamic, estimated) is not None:
                continue
            result = self.path_planner(costs, start, grid)
            if registered_approach:
                return world, grid, result
            if result.path and (best is None or result.total_cost < best[2].total_cost):
                best = (world, grid, result)
        return best

    def _cell_failure(self, grid, costs, static, dynamic, estimated):
        col, row = grid
        if not self.metadata.is_grid_position_in_bounds(col, row):
            return ExitRejectionReason.OUT_OF_MAP
        if static[row, col]:
            return ExitRejectionReason.STATIC_OBSTACLE
        if dynamic[row, col]:
            return ExitRejectionReason.DYNAMIC_OBSTACLE
        if estimated.temperature_c[row, col] >= self.temperature_blocked_c:
            return ExitRejectionReason.TEMPERATURE_LIMIT_EXCEEDED
        if estimated.co_ppm[row, col] >= self.co_blocked_ppm:
            return ExitRejectionReason.CO_LIMIT_EXCEEDED
        if not np.isfinite(costs[row, col]):
            return ExitRejectionReason.INVALID_COST
        return None

    def _path_length(self, path):
        return float(sum(
            math.hypot(b[0] - a[0], b[1] - a[1]) * self.metadata.resolution_m
            for a, b in zip(path, path[1:])
        ))

    def _risk_cost(self, path, costs):
        total = 0.0
        for start, end in zip(path, path[1:]):
            distance = math.hypot(end[0] - start[0], end[1] - start[1]) * self.metadata.resolution_m
            first = max(0.0, float(costs[start[1], start[0]]) - self.base_cost)
            second = max(0.0, float(costs[end[1], end[0]]) - self.base_cost)
            total += distance * 0.5 * (first + second)
        return float(total)

    def _max_cell_risk_cost(self, path, costs):
        values = [
            max(0.0, float(costs[row, col]) - self.base_cost)
            for col, row in path
        ]
        return float(max(values, default=0.0))

    def _risk_threshold_exceeded(self, accumulated, average, maximum):
        thresholds = (
            (accumulated, self.config.dangerous_accumulated_risk_cost),
            (average, self.config.dangerous_average_risk_cost),
            (maximum, self.config.dangerous_max_cell_risk_cost),
        )
        return any(
            limit is not None and value >= limit
            for value, limit in thresholds
        )

    @staticmethod
    def _finite_max(layer, path):
        values = np.asarray([layer[row, col] for col, row in path], dtype=float)
        values = values[np.isfinite(values)]
        return None if not values.size else float(values.max())

    def _exit_neighborhood_values(self, position_world, estimated):
        center = self.metadata.world_to_grid(*position_world)
        radius = int(math.ceil(
            self.config.exit_neighborhood_radius_m / self.metadata.resolution_m
        ))
        cells = []
        for row in range(center[1] - radius, center[1] + radius + 1):
            for col in range(center[0] - radius, center[0] + radius + 1):
                if not self.metadata.is_grid_position_in_bounds(col, row):
                    continue
                if math.hypot(col - center[0], row - center[1]) * self.metadata.resolution_m <= self.config.exit_neighborhood_radius_m + 1e-9:
                    cells.append((col, row))
        return (
            self._finite_max(estimated.temperature_c, cells),
            self._finite_max(estimated.co_ppm, cells),
        )

    @staticmethod
    def _rejected(exit_item, reasons, evaluated_at, approach_world=None, approach_grid=None):
        return ExitEvaluation(
            exit_item.exit_id, exit_item.status.value,
            tuple(exit_item.position_world), approach_world, approach_grid,
            False, False, tuple(), tuple(), None, None, None, None, None, None,
            None, tuple(dict.fromkeys(reasons)), float(evaluated_at),
            tuple(),
        )
