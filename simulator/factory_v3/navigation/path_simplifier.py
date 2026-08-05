"""Safe post-processing for cell-wise A* paths.

Planner nodes are ``(col,row)`` and NumPy layers are ``[row,col]``.  The
simplifier never reads FDS Ground Truth; only the current costmap, estimated
fire observations, and static/dynamic obstacle snapshots are accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Sequence

import numpy as np


class SegmentRejectionReason(Enum):
    OUT_OF_MAP = "out_of_map"
    STATIC_OBSTACLE = "static_obstacle"
    DYNAMIC_OBSTACLE = "dynamic_obstacle"
    CORNER_CUTTING = "corner_cutting"
    TEMPERATURE_LIMIT_EXCEEDED = "temperature_limit_exceeded"
    CO_LIMIT_EXCEEDED = "co_limit_exceeded"
    INVALID_COST = "invalid_cost"
    NEGATIVE_COST = "negative_cost"
    UNKNOWN_RATIO_EXCEEDED = "unknown_ratio_exceeded"
    RISK_INCREASE_EXCEEDED = "risk_increase_exceeded"
    AVERAGE_COST_EXCEEDED = "average_cost_exceeded"
    MAXIMUM_COST_EXCEEDED = "maximum_cost_exceeded"
    ACCUMULATED_COST_EXCEEDED = "accumulated_cost_exceeded"
    CLEARANCE_VIOLATION = "clearance_violation"


@dataclass(frozen=True)
class PathRiskConfig:
    compare_with_original_subpath: bool = True
    max_risk_increase_ratio: float = 1.05
    max_unknown_ratio: float = 1.0
    max_accumulated_cost: float | None = None
    max_average_cost: float | None = None
    max_cell_cost: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.compare_with_original_subpath, bool):
            raise TypeError("compare_with_original_subpath must be bool")
        if self.max_risk_increase_ratio < 0:
            raise ValueError("max_risk_increase_ratio must be non-negative")
        if not 0 <= self.max_unknown_ratio <= 1:
            raise ValueError("max_unknown_ratio must be in [0,1]")
        for name in (
            "max_accumulated_cost", "max_average_cost", "max_cell_cost"
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number or null")


@dataclass(frozen=True)
class PathValidationSettings:
    reject_nan: bool = True
    reject_inf: bool = True
    reject_negative_cost: bool = True
    validate_temperature: bool = True
    validate_co: bool = True
    validate_dynamic_obstacles: bool = True

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        disabled = [
            name for name in self.__dataclass_fields__
            if not getattr(self, name)
        ]
        if disabled:
            raise ValueError(
                "path safety validation cannot be disabled: " + ",".join(disabled)
            )


@dataclass(frozen=True)
class PathSimplificationConfig:
    enabled: bool = True
    extract_direction_changes: bool = True
    enable_safe_shortcuts: bool = True
    line_traversal: str = "supercover"
    preserve_start: bool = True
    preserve_goal: bool = True
    prevent_corner_cutting: bool = True
    use_inflated_costmap: bool = True
    fallback_to_original_path: bool = True
    robot_clearance_m: float = 0.0
    risk: PathRiskConfig = PathRiskConfig()
    validation: PathValidationSettings = PathValidationSettings()

    def __post_init__(self) -> None:
        for name in (
            "enabled", "extract_direction_changes", "enable_safe_shortcuts",
            "preserve_start", "preserve_goal", "prevent_corner_cutting",
            "use_inflated_costmap", "fallback_to_original_path",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.line_traversal != "supercover":
            raise ValueError("line_traversal must be 'supercover'")
        if not self.preserve_start or not self.preserve_goal:
            raise ValueError("start and goal preservation are mandatory")
        if not self.prevent_corner_cutting:
            raise ValueError("prevent_corner_cutting must remain true")
        if not self.use_inflated_costmap:
            raise ValueError("use_inflated_costmap must remain true")
        if self.robot_clearance_m < 0:
            raise ValueError("robot_clearance_m must be non-negative")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None):
        values = dict(values or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown path_simplification settings: {sorted(unknown)}")
        risk = PathRiskConfig(**dict(values.pop("risk", {}) or {}))
        validation = PathValidationSettings(
            **dict(values.pop("validation", {}) or {})
        )
        return cls(risk=risk, validation=validation, **values)


@dataclass(frozen=True)
class SegmentSafetyResult:
    safe: bool
    start_grid: tuple[int, int]
    end_grid: tuple[int, int]
    traversed_cells: tuple[tuple[int, int], ...]
    distance_m: float
    accumulated_risk_cost: float
    average_risk_cost: float
    maximum_cell_cost: float
    max_temperature_c: float | None
    max_co_ppm: float | None
    unknown_ratio: float
    first_rejected_cell: tuple[int, int] | None
    rejection_reasons: tuple[SegmentRejectionReason, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["start_grid"] = list(self.start_grid)
        result["end_grid"] = list(self.end_grid)
        result["traversed_cells"] = [list(item) for item in self.traversed_cells]
        result["first_rejected_cell"] = (
            None if self.first_rejected_cell is None
            else list(self.first_rejected_cell)
        )
        result["rejection_reasons"] = [item.value for item in self.rejection_reasons]
        return result


@dataclass(frozen=True)
class PathSimplificationResult:
    success: bool
    original_path_grid: tuple[tuple[int, int], ...]
    corner_path_grid: tuple[tuple[int, int], ...]
    simplified_path_grid: tuple[tuple[int, int], ...]
    waypoints_world: tuple[tuple[float, float], ...]
    original_point_count: int
    corner_point_count: int
    simplified_point_count: int
    original_length_m: float
    simplified_length_m: float
    original_risk_cost: float
    simplified_risk_cost: float
    max_temperature_c: float | None
    max_co_ppm: float | None
    unknown_ratio: float
    used_costmap_revision: int | None
    fallback_used: bool
    failure_reason: str | None
    rejected_shortcuts: tuple[SegmentSafetyResult, ...] = ()

    @property
    def reduction_ratio(self) -> float:
        if self.original_point_count == 0:
            return 0.0
        return 1.0 - self.simplified_point_count / self.original_point_count

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "original_path_grid", "corner_path_grid", "simplified_path_grid"
        ):
            result[key] = [list(item) for item in getattr(self, key)]
        result["waypoints_world"] = [list(item) for item in self.waypoints_world]
        result["rejected_shortcuts"] = [
            item.to_dict() for item in self.rejected_shortcuts
        ]
        result["reduction_ratio"] = self.reduction_ratio
        return result


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _deduplicate(path_grid: Sequence[tuple[int, int]]):
    output = []
    for point in path_grid:
        normalized = (int(point[0]), int(point[1]))
        if not output or output[-1] != normalized:
            output.append(normalized)
    return tuple(output)


def extract_direction_change_points(
    path_grid: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Keep start, normalized-direction changes, and goal deterministically."""
    path = _deduplicate(path_grid)
    if len(path) <= 2:
        return path
    output = [path[0]]
    previous_direction = (
        _sign(path[1][0] - path[0][0]),
        _sign(path[1][1] - path[0][1]),
    )
    for index in range(1, len(path) - 1):
        direction = (
            _sign(path[index + 1][0] - path[index][0]),
            _sign(path[index + 1][1] - path[index][1]),
        )
        if direction != previous_direction:
            output.append(path[index])
        previous_direction = direction
    output.append(path[-1])
    return tuple(output)


def cells_touched_by_segment(
    start_grid: tuple[int, int], end_grid: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Conservative 2-D supercover from cell centre to cell centre.

    At an exact grid-corner crossing both side-adjacent cells are included
    before the diagonal cell. This is deliberately more conservative than
    ordinary Bresenham and exposes potential corner cutting.
    """
    x, y = int(start_grid[0]), int(start_grid[1])
    end_x, end_y = int(end_grid[0]), int(end_grid[1])
    if (x, y) == (end_x, end_y):
        return ((x, y),)
    dx, dy = end_x - x, end_y - y
    step_x, step_y = _sign(dx), _sign(dy)
    t_delta_x = math.inf if dx == 0 else 1.0 / abs(dx)
    t_delta_y = math.inf if dy == 0 else 1.0 / abs(dy)
    t_max_x = math.inf if dx == 0 else 0.5 / abs(dx)
    t_max_y = math.inf if dy == 0 else 0.5 / abs(dy)
    output = [(x, y)]

    def append(cell):
        if cell not in output:
            output.append(cell)

    while (x, y) != (end_x, end_y):
        if abs(t_max_x - t_max_y) <= 1e-12:
            append((x + step_x, y))
            append((x, y + step_y))
            x += step_x
            y += step_y
            append((x, y))
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            x += step_x
            append((x, y))
            t_max_x += t_delta_x
        else:
            y += step_y
            append((x, y))
            t_max_y += t_delta_y
    return tuple(output)


def _corner_crossing_sides(start_grid, end_grid):
    """Return side-cell pairs touched when a segment crosses exact corners."""
    x, y = start_grid
    end_x, end_y = end_grid
    dx, dy = end_x - x, end_y - y
    step_x, step_y = _sign(dx), _sign(dy)
    t_delta_x = math.inf if dx == 0 else 1.0 / abs(dx)
    t_delta_y = math.inf if dy == 0 else 1.0 / abs(dy)
    t_max_x = math.inf if dx == 0 else 0.5 / abs(dx)
    t_max_y = math.inf if dy == 0 else 0.5 / abs(dy)
    pairs = []
    while (x, y) != (end_x, end_y):
        if abs(t_max_x - t_max_y) <= 1e-12:
            pairs.append(((x + step_x, y), (x, y + step_y)))
            x += step_x
            y += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            x += step_x
            t_max_x += t_delta_x
        else:
            y += step_y
            t_max_y += t_delta_y
    return tuple(pairs)


class SafePathSimplifier:
    def __init__(self, map_metadata, config: PathSimplificationConfig | None = None):
        self.metadata = map_metadata
        self.config = config or PathSimplificationConfig()

    def _validate_arrays(self, costmap, static, dynamic, estimated):
        expected = (self.metadata.height, self.metadata.width)
        arrays = (
            np.asarray(costmap, dtype=float), np.asarray(static, dtype=bool),
            np.asarray(dynamic, dtype=bool),
        )
        if any(item.shape != expected for item in arrays):
            raise ValueError(f"path simplification map shape must be {expected}")
        if estimated.shape != expected:
            raise ValueError(f"estimated fire map shape must be {expected}")
        return arrays

    def _clearance_failure(self, cell, static, dynamic):
        radius = int(math.ceil(
            self.config.robot_clearance_m / self.metadata.resolution_m
        ))
        if radius <= 0:
            return None
        col, row = cell
        for y in range(row - radius, row + radius + 1):
            for x in range(col - radius, col + radius + 1):
                if math.hypot(x - col, y - row) > radius + 1e-9:
                    continue
                if not self.metadata.is_grid_position_in_bounds(x, y):
                    return SegmentRejectionReason.CLEARANCE_VIOLATION
                if static[y, x] or dynamic[y, x]:
                    return SegmentRejectionReason.CLEARANCE_VIOLATION
        return None

    @staticmethod
    def _finite_max(layer, cells):
        values = [float(layer[row, col]) for col, row in cells]
        finite = [value for value in values if math.isfinite(value)]
        return max(finite) if finite else None

    def evaluate_segment(
        self, start_grid, end_grid, *, costmap, static_obstacle_map,
        dynamic_obstacle_map, estimated_fire_map,
        reference: SegmentSafetyResult | None = None,
    ) -> SegmentSafetyResult:
        costs, static, dynamic = self._validate_arrays(
            costmap, static_obstacle_map, dynamic_obstacle_map,
            estimated_fire_map,
        )
        start = (int(start_grid[0]), int(start_grid[1]))
        end = (int(end_grid[0]), int(end_grid[1]))
        cells = cells_touched_by_segment(start, end)
        reasons: list[SegmentRejectionReason] = []
        first_rejected_cell = None
        valid_cells = []
        cell_costs = []
        for cell in cells:
            col, row = cell
            if not self.metadata.is_grid_position_in_bounds(col, row):
                reasons.append(SegmentRejectionReason.OUT_OF_MAP)
                first_rejected_cell = first_rejected_cell or cell
                continue
            valid_cells.append(cell)
            if static[row, col]:
                reasons.append(SegmentRejectionReason.STATIC_OBSTACLE)
                first_rejected_cell = first_rejected_cell or cell
            if self.config.validation.validate_dynamic_obstacles and dynamic[row, col]:
                reasons.append(SegmentRejectionReason.DYNAMIC_OBSTACLE)
                first_rejected_cell = first_rejected_cell or cell
            clearance = self._clearance_failure(cell, static, dynamic)
            if clearance is not None:
                reasons.append(clearance)
                first_rejected_cell = first_rejected_cell or cell
            cost = float(costs[row, col])
            if math.isnan(cost) and self.config.validation.reject_nan:
                reasons.append(SegmentRejectionReason.INVALID_COST)
                first_rejected_cell = first_rejected_cell or cell
            elif math.isinf(cost) and self.config.validation.reject_inf:
                reasons.append(SegmentRejectionReason.INVALID_COST)
                first_rejected_cell = first_rejected_cell or cell
            elif cost < 0 and self.config.validation.reject_negative_cost:
                reasons.append(SegmentRejectionReason.NEGATIVE_COST)
                first_rejected_cell = first_rejected_cell or cell
            elif math.isfinite(cost):
                cell_costs.append(cost)
            if (
                self.config.validation.validate_temperature
                and estimated_fire_map.observed_mask[row, col]
                and math.isfinite(float(estimated_fire_map.temperature_c[row, col]))
                and estimated_fire_map.temperature_c[row, col]
                >= estimated_fire_map.temperature_blocked_c
            ):
                reasons.append(SegmentRejectionReason.TEMPERATURE_LIMIT_EXCEEDED)
                first_rejected_cell = first_rejected_cell or cell
            if (
                self.config.validation.validate_co
                and estimated_fire_map.observed_mask[row, col]
                and math.isfinite(float(estimated_fire_map.co_ppm[row, col]))
                and estimated_fire_map.co_ppm[row, col]
                >= estimated_fire_map.co_blocked_ppm
            ):
                reasons.append(SegmentRejectionReason.CO_LIMIT_EXCEEDED)
                first_rejected_cell = first_rejected_cell or cell

        # Exact diagonal corner crossings include both side cells in the
        # supercover. Mark the more specific reason when either side is blocked.
        if self.config.prevent_corner_cutting:
            for adjacent in _corner_crossing_sides(start, end):
                if any(
                    not self.metadata.is_grid_position_in_bounds(col, row)
                    or static[row, col] or dynamic[row, col]
                    for col, row in adjacent
                ):
                    reasons.append(SegmentRejectionReason.CORNER_CUTTING)
                    first_rejected_cell = first_rejected_cell or adjacent[0]

        distance = math.hypot(
            end[0] - start[0], end[1] - start[1]
        ) * self.metadata.resolution_m
        average = float(np.mean(cell_costs)) if cell_costs else math.inf
        maximum = float(max(cell_costs)) if cell_costs else math.inf
        # Cost is integrated over metric distance, matching weighted A*'s
        # distance-weighted meaning while remaining comparable across cell counts.
        accumulated = average * distance if math.isfinite(average) else math.inf
        observed_count = sum(
            bool(estimated_fire_map.observed_mask[row, col])
            for col, row in valid_cells
        )
        unknown_ratio = (
            1.0 - observed_count / len(valid_cells) if valid_cells else 1.0
        )
        if unknown_ratio > self.config.risk.max_unknown_ratio + 1e-12:
            reasons.append(SegmentRejectionReason.UNKNOWN_RATIO_EXCEEDED)
        limits = (
            (self.config.risk.max_accumulated_cost, accumulated,
             SegmentRejectionReason.ACCUMULATED_COST_EXCEEDED),
            (self.config.risk.max_average_cost, average,
             SegmentRejectionReason.AVERAGE_COST_EXCEEDED),
            (self.config.risk.max_cell_cost, maximum,
             SegmentRejectionReason.MAXIMUM_COST_EXCEEDED),
        )
        for limit, value, reason in limits:
            if limit is not None and value > limit + 1e-12:
                reasons.append(reason)
        if reference is not None and self.config.risk.compare_with_original_subpath:
            ratio = self.config.risk.max_risk_increase_ratio
            if (
                average > reference.average_risk_cost * ratio + 1e-12
                or accumulated > reference.accumulated_risk_cost * ratio + 1e-12
                or maximum > reference.maximum_cell_cost * ratio + 1e-12
            ):
                reasons.append(SegmentRejectionReason.RISK_INCREASE_EXCEEDED)
        max_temp = self._finite_max(estimated_fire_map.temperature_c, valid_cells)
        max_co = self._finite_max(estimated_fire_map.co_ppm, valid_cells)
        unique_reasons = tuple(dict.fromkeys(reasons))
        return SegmentSafetyResult(
            not unique_reasons, start, end, cells, distance, accumulated,
            average, maximum, max_temp, max_co, unknown_ratio,
            first_rejected_cell, unique_reasons,
        )

    def _evaluate_path(
        self, path, *, costmap, static, dynamic, estimated,
    ) -> SegmentSafetyResult:
        if len(path) == 1:
            return self.evaluate_segment(
                path[0], path[0], costmap=costmap,
                static_obstacle_map=static, dynamic_obstacle_map=dynamic,
                estimated_fire_map=estimated,
            )
        segments = [
            self.evaluate_segment(
                start, end, costmap=costmap, static_obstacle_map=static,
                dynamic_obstacle_map=dynamic, estimated_fire_map=estimated,
            ) for start, end in zip(path, path[1:])
        ]
        cells = []
        reasons = []
        for segment in segments:
            for cell in segment.traversed_cells:
                if not cells or cells[-1] != cell:
                    cells.append(cell)
            reasons.extend(segment.rejection_reasons)
        distance = sum(item.distance_m for item in segments)
        accumulated = sum(item.accumulated_risk_cost for item in segments)
        average = accumulated / distance if distance > 0 else segments[0].average_risk_cost
        maximum = max(item.maximum_cell_cost for item in segments)
        temperatures = [item.max_temperature_c for item in segments if item.max_temperature_c is not None]
        co_values = [item.max_co_ppm for item in segments if item.max_co_ppm is not None]
        unknown = sum(
            not bool(estimated.observed_mask[row, col]) for col, row in cells
            if self.metadata.is_grid_position_in_bounds(col, row)
        ) / len(cells) if cells else 1.0
        return SegmentSafetyResult(
            not reasons, path[0], path[-1], tuple(cells), distance,
            accumulated, average, maximum,
            max(temperatures) if temperatures else None,
            max(co_values) if co_values else None, unknown,
            next((item.first_rejected_cell for item in segments
                  if item.first_rejected_cell is not None), None),
            tuple(dict.fromkeys(reasons)),
        )

    def validate_path(
        self, path_grid, *, costmap, static_obstacle_map,
        dynamic_obstacle_map, estimated_fire_map,
    ) -> SegmentSafetyResult:
        """Validate a complete path using the same conservative shortcut rules."""
        path = _deduplicate(path_grid)
        if not path:
            raise ValueError("path_grid must not be empty")
        costs, static, dynamic = self._validate_arrays(
            costmap, static_obstacle_map, dynamic_obstacle_map,
            estimated_fire_map,
        )
        return self._evaluate_path(
            path, costmap=costs, static=static, dynamic=dynamic,
            estimated=estimated_fire_map,
        )

    @staticmethod
    def _corner_positions(original, corners):
        positions = []
        search_from = 0
        for corner in corners:
            index = original.index(corner, search_from)
            positions.append(index)
            search_from = index + 1
        return positions

    def simplify(
        self, path_grid, *, costmap, static_obstacle_map,
        dynamic_obstacle_map, estimated_fire_map, costmap_revision=None,
        start_world=None, goal_world=None,
    ) -> PathSimplificationResult:
        original = _deduplicate(path_grid)
        if not original:
            return self._failure(original, costmap_revision, "empty_path")
        costs, static, dynamic = self._validate_arrays(
            costmap, static_obstacle_map, dynamic_obstacle_map,
            estimated_fire_map,
        )
        original_eval = self._evaluate_path(
            original, costmap=costs, static=static, dynamic=dynamic,
            estimated=estimated_fire_map,
        )
        if not original_eval.safe:
            return self._failure(
                original, costmap_revision,
                "original_path_unsafe:" + ",".join(
                    item.value for item in original_eval.rejection_reasons
                ), original_eval,
            )
        corners = (
            extract_direction_change_points(original)
            if self.config.enabled and self.config.extract_direction_changes
            else original
        )
        simplified = [corners[0]]
        rejected = []
        corner_positions = self._corner_positions(original, corners)
        anchor = 0
        while anchor < len(corners) - 1:
            chosen = anchor + 1
            if self.config.enabled and self.config.enable_safe_shortcuts:
                for candidate in range(len(corners) - 1, anchor, -1):
                    reference_path = original[
                        corner_positions[anchor]:corner_positions[candidate] + 1
                    ]
                    reference = self._evaluate_path(
                        reference_path, costmap=costs, static=static,
                        dynamic=dynamic, estimated=estimated_fire_map,
                    )
                    shortcut = self.evaluate_segment(
                        corners[anchor], corners[candidate], costmap=costs,
                        static_obstacle_map=static,
                        dynamic_obstacle_map=dynamic,
                        estimated_fire_map=estimated_fire_map,
                        reference=reference,
                    )
                    if shortcut.safe:
                        chosen = candidate
                        break
                    rejected.append(shortcut)
            simplified.append(corners[chosen])
            anchor = chosen
        simplified = tuple(simplified)
        final_eval = self._evaluate_path(
            simplified, costmap=costs, static=static, dynamic=dynamic,
            estimated=estimated_fire_map,
        )
        fallback = not self.config.enabled
        failure_reason = "simplification_disabled;used_original_path" if fallback else None
        if not final_eval.safe:
            if not self.config.fallback_to_original_path:
                return self._failure(
                    original, costmap_revision, "simplified_path_unsafe",
                    original_eval, corners, tuple(rejected),
                )
            simplified = original
            final_eval = original_eval
            fallback = True
            failure_reason = "simplified_path_unsafe;used_original_path"
        waypoints = [self.metadata.grid_to_world(*item) for item in simplified]
        if waypoints and start_world is not None:
            waypoints[0] = (float(start_world[0]), float(start_world[1]))
        if waypoints and goal_world is not None:
            waypoints[-1] = (float(goal_world[0]), float(goal_world[1]))
        return PathSimplificationResult(
            True, original, corners, simplified, tuple(waypoints),
            len(original), len(corners), len(simplified),
            original_eval.distance_m, final_eval.distance_m,
            original_eval.accumulated_risk_cost,
            final_eval.accumulated_risk_cost,
            final_eval.max_temperature_c, final_eval.max_co_ppm,
            final_eval.unknown_ratio,
            None if costmap_revision is None else int(costmap_revision),
            fallback, failure_reason, tuple(rejected),
        )

    @staticmethod
    def _failure(
        original, revision, reason, evaluation=None, corners=(), rejected=(),
    ):
        return PathSimplificationResult(
            False, tuple(original), tuple(corners), tuple(), tuple(),
            len(original), len(corners), 0,
            0.0 if evaluation is None else evaluation.distance_m, 0.0,
            math.inf if evaluation is None else evaluation.accumulated_risk_cost,
            math.inf, None, None, 1.0,
            None if revision is None else int(revision), False, str(reason),
            tuple(rejected),
        )
