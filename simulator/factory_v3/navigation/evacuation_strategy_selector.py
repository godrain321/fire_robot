"""Sensor-belief based final evacuation route policy.

Coordinates are explicit: world paths contain ``(x, y)`` metres and grid
paths contain ``(col, row)``.  Ground Truth is intentionally absent from every
public API in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any

import numpy as np

from navigation.exit_switching import is_opposite_direction
from planner.a_star import weighted_a_star


def _line_cells(start, end):
    """Yield an inclusive Bresenham segment in ``(col,row)`` order."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx - dy
    while True:
        yield x0, y0
        if (x0, y0) == (x1, y1):
            return
        twice = 2 * error
        if twice > -dy:
            error -= dy
            x0 += sx
        if twice < dx:
            error += dx
            y0 += sy


class HazardKnowledgeState(Enum):
    NO_FIRE_INFORMATION = "no_fire_information"
    FIRE_INFORMATION_AVAILABLE = "fire_information_available"


class EvacuationStrategy(Enum):
    SAFE_EXIT_PLANNING = "safe_exit_planning"
    NEAREST_REACHABLE_EXIT = "nearest_reachable_exit"
    RETURN_VIA_TRAVEL_HISTORY = "return_via_travel_history"
    REPLAN_TO_ENTRANCE = "replan_to_entrance"
    REPLAN_TO_ALTERNATIVE_EXIT = "replan_to_alternative_exit"
    NO_SAFE_ROUTE = "no_safe_route"


class RouteFailureReason(Enum):
    NO_ENTRY_INFORMATION = "no_entry_information"
    RETURN_PATH_UNAVAILABLE = "return_path_unavailable"
    ENTRANCE_PATH_UNAVAILABLE = "entrance_path_unavailable"
    NO_SAFE_EXIT = "no_safe_exit"
    INVALID_COST = "invalid_cost"
    OUT_OF_MAP = "out_of_map"


@dataclass(frozen=True)
class HazardKnowledgeDecision:
    state: HazardKnowledgeState
    reasons: tuple[str, ...]
    evaluated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reasons": list(self.reasons),
            "evaluated_at": self.evaluated_at,
        }


@dataclass(frozen=True)
class EvacuationRouteDecision:
    success: bool
    strategy: EvacuationStrategy
    target_exit_id: str | None
    target_position_world: tuple[float, float] | None
    path_world: tuple[tuple[float, float], ...]
    path_grid: tuple[tuple[int, int], ...]
    hazard_knowledge: HazardKnowledgeDecision
    failure_reason: RouteFailureReason | None
    reasons: tuple[str, ...]
    created_at: float
    costmap_revision: int
    evacuation_plan: Any | None = None
    return_plan: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["strategy"] = self.strategy.value
        result["hazard_knowledge"] = self.hazard_knowledge.to_dict()
        result["failure_reason"] = (
            None if self.failure_reason is None else self.failure_reason.value
        )
        result["target_position_world"] = (
            None if self.target_position_world is None
            else list(self.target_position_world)
        )
        result["path_world"] = [list(point) for point in self.path_world]
        result["path_grid"] = [list(point) for point in self.path_grid]
        result["evacuation_plan"] = (
            None if self.evacuation_plan is None
            else self.evacuation_plan.to_dict()
        )
        result["return_plan"] = (
            None if self.return_plan is None else self.return_plan.to_dict()
        )
        return result


@dataclass(frozen=True)
class EvacuationRouteSelectionConfig:
    enabled: bool = True
    no_fire_information_strategy: str = "nearest_reachable_exit"
    fire_information_strategy: str = "evaluate_all_exits"
    prefer_original_entrance_on_return_failure: bool = True
    evaluate_alternative_exits_on_entrance_failure: bool = True

    def __post_init__(self) -> None:
        for name in (
            "enabled", "prefer_original_entrance_on_return_failure",
            "evaluate_alternative_exits_on_entrance_failure",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.no_fire_information_strategy != "nearest_reachable_exit":
            raise ValueError("unsupported no-fire-information strategy")
        if self.fire_information_strategy != "evaluate_all_exits":
            raise ValueError("unsupported fire-information strategy")

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown evacuation_route_selection settings: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class PathValidationConfig:
    validate_on_costmap_revision: bool = True
    validate_on_dynamic_obstacle_event: bool = True
    validate_on_hazard_update: bool = True
    stop_before_replanning: bool = True

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if not self.stop_before_replanning:
            raise ValueError("stop_before_replanning must remain true for safety")

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown path_validation settings: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class ReplanningConfig:
    enabled: bool = True
    max_replan_attempts: int = 5
    cooldown_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        if (
            isinstance(self.max_replan_attempts, bool)
            or not isinstance(self.max_replan_attempts, int)
            or self.max_replan_attempts < 0
        ):
            raise ValueError("max_replan_attempts must be a non-negative integer")
        if isinstance(self.cooldown_seconds, bool) or self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown replanning settings: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class HazardKnowledgeConfig:
    """Thresholds for declaring meaningful sensor-derived fire evidence."""

    temperature_elevated_c: float = 35.0
    co_elevated_ppm: float = 100.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown hazard_knowledge settings: {sorted(unknown)}")
        return cls(**values)


class HazardKnowledgeTracker:
    """Sticky mission-scoped knowledge derived only from estimated observations."""

    def __init__(self, *, temperature_elevated_c: float, co_elevated_ppm: float):
        if temperature_elevated_c < 0 or co_elevated_ppm < 0:
            raise ValueError("elevated hazard thresholds must be non-negative")
        self.temperature_elevated_c = float(temperature_elevated_c)
        self.co_elevated_ppm = float(co_elevated_ppm)
        self._fire_information_seen = False
        self._persistent_reasons: set[str] = set()

    def evaluate(
        self, estimated_fire_map, *, evaluated_at: float,
        fire_related_event: str | None = None,
    ) -> HazardKnowledgeDecision:
        observed = np.asarray(estimated_fire_map.observed_mask, dtype=bool)
        temp = np.asarray(estimated_fire_map.temperature_c, dtype=float)
        co = np.asarray(estimated_fire_map.co_ppm, dtype=float)
        reasons: set[str] = set()
        if np.any(observed & np.isfinite(temp) & (temp > self.temperature_elevated_c)):
            reasons.add("elevated_temperature_observed")
        if np.any(observed & np.isfinite(co) & (co > self.co_elevated_ppm)):
            reasons.add("elevated_co_observed")
        if np.any(estimated_fire_map.blocked_mask):
            reasons.add("hazard_blocked_cell_observed")
        if fire_related_event:
            reasons.add(str(fire_related_event))
        if reasons:
            self._fire_information_seen = True
            self._persistent_reasons.update(reasons)
        if self._fire_information_seen:
            return HazardKnowledgeDecision(
                HazardKnowledgeState.FIRE_INFORMATION_AVAILABLE,
                tuple(sorted(self._persistent_reasons)), float(evaluated_at),
            )
        return HazardKnowledgeDecision(
            HazardKnowledgeState.NO_FIRE_INFORMATION,
            ("no_valid_fire_observation_history",), float(evaluated_at),
        )

    def reset(self) -> None:
        self._fire_information_seen = False
        self._persistent_reasons.clear()


class EvacuationStrategySelector:
    """Choose initial and fallback routes without reading Ground Truth."""

    def __init__(
        self, map_metadata, hazard_tracker, return_planner, evacuation_planner,
        config: EvacuationRouteSelectionConfig | None = None,
    ) -> None:
        self.map_metadata = map_metadata
        self.hazard_tracker = hazard_tracker
        self.return_planner = return_planner
        self.evacuation_planner = evacuation_planner
        self.config = config or EvacuationRouteSelectionConfig()

    @staticmethod
    def _effective_cost(cost_map, static_map, dynamic_map, estimated_map):
        result = np.asarray(cost_map, dtype=float).copy()
        result[np.asarray(static_map, dtype=bool)] = np.inf
        result[np.asarray(dynamic_map, dtype=bool)] = np.inf
        result[np.asarray(estimated_map.blocked_mask, dtype=bool)] = np.inf
        return result

    def select_initial_route(
        self, *, world_state, travel_history, start_position_world, cost_map,
        costmap_revision: int, created_at: float, victim_position_world=None,
    ) -> EvacuationRouteDecision:
        hazard = self.hazard_tracker.evaluate(
            world_state.estimated_fire_map, evaluated_at=created_at,
        )
        if hazard.state is HazardKnowledgeState.FIRE_INFORMATION_AVAILABLE:
            return self._plan_exits(
                world_state,
                victim_position_world or start_position_world,
                cost_map, hazard,
                costmap_revision, created_at,
                EvacuationStrategy.SAFE_EXIT_PLANNING,
            )
        return self._plan_nearest_reachable_exit(
            world_state, victim_position_world or start_position_world,
            cost_map, hazard, costmap_revision, created_at,
        )

    def _plan_nearest_reachable_exit(
        self, world, start_world, cost_map, hazard, revision, created_at,
        *, excluded_exit_ids=(),
    ) -> EvacuationRouteDecision:
        """Select the shortest currently reachable exit when no fire is known.

        Unknown cells keep their finite partial-costmap penalty.  This mode does
        not claim that they are safe; it deliberately gathers observations while
        moving and the active route is rechecked after every belief revision.
        """
        return self._plan_exits(
            world, start_world, cost_map, hazard, revision, created_at,
            EvacuationStrategy.NEAREST_REACHABLE_EXIT,
            excluded_exit_ids=excluded_exit_ids,
        )

    def replan_after_return_invalidated(
        self, *, world_state, current_position_world, cost_map,
        costmap_revision: int, created_at: float,
    ) -> EvacuationRouteDecision:
        hazard = self.hazard_tracker.evaluate(
            world_state.estimated_fire_map, evaluated_at=created_at,
            fire_related_event="active_return_path_invalidated",
        )
        entry = world_state.mission_entry_position_world
        if entry is not None and self.config.prefer_original_entrance_on_return_failure:
            entrance = self._plan_to_position(
                world_state, current_position_world, entry, cost_map,
                hazard, costmap_revision, created_at,
            )
            if entrance.success:
                return entrance
        if self.config.evaluate_alternative_exits_on_entrance_failure:
            return self._plan_exits(
                world_state, current_position_world, cost_map, hazard,
                costmap_revision, created_at,
                EvacuationStrategy.REPLAN_TO_ALTERNATIVE_EXIT,
            )
        return self._failure(
            hazard, RouteFailureReason.ENTRANCE_PATH_UNAVAILABLE, created_at,
            costmap_revision, ("entrance replanning failed",),
        )

    def replan_to_safe_exit(
        self, *, world_state, current_position_world, cost_map,
        costmap_revision: int, created_at: float, excluded_exit_ids=(),
        risk_first: bool = False,
    ) -> EvacuationRouteDecision:
        hazard = self.hazard_tracker.evaluate(
            world_state.estimated_fire_map, evaluated_at=created_at,
            fire_related_event="active_evacuation_path_invalidated",
        )
        return self._plan_exits(
            world_state, current_position_world, cost_map, hazard,
            costmap_revision, created_at,
            EvacuationStrategy.REPLAN_TO_ALTERNATIVE_EXIT,
            excluded_exit_ids=excluded_exit_ids,
            risk_first=risk_first,
        )

    def replan_to_opposite_exit(
        self, *, world_state, current_position_world, direction_world, cost_map,
        costmap_revision: int, created_at: float, current_exit_id: str,
        minimum_direction_difference_deg: float,
    ) -> EvacuationRouteDecision:
        """Evaluate only safe exits in the robot's rear world-space sector."""
        candidate_ids = {
            item.exit_id for item in world_state.exits.values()
            if item.exit_id != current_exit_id
            and is_opposite_direction(
                direction_world, current_position_world,
                item.position_world,
                minimum_difference_deg=minimum_direction_difference_deg,
            )
        }
        hazard = self.hazard_tracker.evaluate(
            world_state.estimated_fire_map, evaluated_at=created_at,
            fire_related_event="sustained_route_cost_increase",
        )
        if not candidate_ids:
            return self._failure(
                hazard, RouteFailureReason.NO_SAFE_EXIT, created_at,
                costmap_revision, ("no opposite-direction exit candidate",),
            )
        return self._plan_exits(
            world_state, current_position_world, cost_map, hazard,
            costmap_revision, created_at,
            EvacuationStrategy.REPLAN_TO_ALTERNATIVE_EXIT,
            candidate_exit_ids=candidate_ids,
        )

    def _plan_to_position(
        self, world, start_world, goal_world, cost_map, hazard, revision, created_at,
    ):
        try:
            start = self.map_metadata.world_to_grid(*start_world)
            goal = self.map_metadata.world_to_grid(*goal_world)
        except ValueError:
            return self._failure(
                hazard, RouteFailureReason.OUT_OF_MAP, created_at, revision,
                ("entrance or current position is outside map",),
            )
        effective = self._effective_cost(
            cost_map, world.static_obstacle_map,
            world.dynamic_obstacle_mask(), world.estimated_fire_map,
        )
        result = weighted_a_star(effective, start, goal)
        if not result.path:
            return self._failure(
                hazard, RouteFailureReason.ENTRANCE_PATH_UNAVAILABLE,
                created_at, revision, (result.reason,),
            )
        path_world = tuple(
            self.map_metadata.grid_to_world(col, row) for col, row in result.path
        )
        return EvacuationRouteDecision(
            True, EvacuationStrategy.REPLAN_TO_ENTRANCE,
            world.mission_entry_id, tuple(goal_world), path_world,
            tuple(result.path), hazard, None,
            ("history return invalidated; safe A* route to original entrance",),
            float(created_at), int(revision),
        )

    def _plan_exits(
        self, world, start_world, cost_map, hazard, revision, created_at, strategy,
        *, excluded_exit_ids=(), candidate_exit_ids=None, risk_first=False,
    ):
        dynamic = world.dynamic_obstacle_mask()
        effective = self._effective_cost(
            cost_map, world.static_obstacle_map, dynamic,
            world.estimated_fire_map,
        )
        excluded = {str(item) for item in excluded_exit_ids}
        allowed = (
            None if candidate_exit_ids is None
            else {str(item) for item in candidate_exit_ids}
        )
        plan = self.evacuation_planner.plan(
            (
                item for item in world.exits.values()
                if item.exit_id not in excluded
                and (allowed is None or item.exit_id in allowed)
            ),
            start_world, cost_map=effective,
            static_obstacle_map=world.static_obstacle_map,
            dynamic_obstacle_map=dynamic,
            estimated_fire_map=world.estimated_fire_map,
            created_at=created_at,
            risk_first=risk_first,
        )
        world.record_exit_evaluations(plan)
        if not plan.success:
            return self._failure(
                hazard, RouteFailureReason.NO_SAFE_EXIT, created_at, revision,
                (plan.failure_reason.value,), evacuation_plan=plan,
            )
        return EvacuationRouteDecision(
            True, strategy, plan.selected_exit_id,
            plan.selected_approach_position_world, plan.path_world,
            plan.path_grid, hazard, None,
            (plan.selection_reason,), float(created_at), int(revision),
            evacuation_plan=plan,
        )

    @staticmethod
    def _failure(
        hazard, reason, created_at, revision, reasons, *,
        evacuation_plan=None, return_plan=None,
    ):
        return EvacuationRouteDecision(
            False, EvacuationStrategy.NO_SAFE_ROUTE, None, None, tuple(),
            tuple(), hazard, reason, tuple(reasons), float(created_at),
            int(revision), evacuation_plan, return_plan,
        )

    def validate_grid_path(
        self, path_grid, *, start_index: int, cost_map, world_state,
    ) -> tuple[bool, tuple[int, int] | None, str | None]:
        effective = self._effective_cost(
            cost_map, world_state.static_obstacle_map,
            world_state.dynamic_obstacle_mask(), world_state.estimated_fire_map,
        )
        remaining = tuple(path_grid)[max(0, int(start_index)):]
        cells = []
        if len(remaining) == 1:
            cells = [remaining[0]]
        for start, end in zip(remaining, remaining[1:]):
            for cell in _line_cells(start, end):
                if not cells or cells[-1] != cell:
                    cells.append(cell)
        dynamic_mask = world_state.dynamic_obstacle_mask()
        def cell_failure(cell):
            col, row = cell
            if not self.map_metadata.is_grid_position_in_bounds(col, row):
                return "out_of_map"
            if not math.isfinite(float(effective[row, col])):
                if world_state.static_obstacle_map[row, col]:
                    reason = "static_obstacle_on_path"
                elif dynamic_mask[row, col]:
                    reason = "dynamic_obstacle_on_path"
                elif world_state.estimated_fire_map.blocked_mask[row, col]:
                    temp = world_state.estimated_fire_map.temperature_c[row, col]
                    reason = (
                        "temperature_limit_exceeded"
                        if np.isfinite(temp)
                        and temp >= world_state.estimated_fire_map.temperature_blocked_c
                        else "co_limit_exceeded"
                    )
                else:
                    reason = "invalid_cost"
                return reason
            return None

        for index, cell in enumerate(cells):
            reason = cell_failure(cell)
            if reason is not None:
                return False, cell, reason
            if index == 0:
                continue
            previous = cells[index - 1]
            dx, dy = cell[0] - previous[0], cell[1] - previous[1]
            if abs(dx) == 1 and abs(dy) == 1:
                for adjacent in (
                    (previous[0] + dx, previous[1]),
                    (previous[0], previous[1] + dy),
                ):
                    reason = cell_failure(adjacent)
                    if reason is not None:
                        return False, adjacent, reason
        return True, None, None
