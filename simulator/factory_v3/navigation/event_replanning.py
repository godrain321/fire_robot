"""Event-driven replan policy for robot-belief paths.

Grid nodes are ``(col, row)`` while NumPy maps are indexed ``[row, col]``.
This module intentionally accepts no Ground Truth inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
import math
from typing import Any, Mapping, Sequence

import numpy as np

from world.entities import ExitStatus


class ReplanPriority(IntEnum):
    NONE = 0
    PERIODIC = 10
    SAFER_EXIT = 20
    VICTIM_FOLLOW = 30
    EXIT_INVALID = 40
    HAZARD_BLOCKED = 50
    PATH_BLOCKED = 60


class ReplanReason(Enum):
    NONE = "none"
    DYNAMIC_OBSTACLE_ON_PATH = "dynamic_obstacle_on_path"
    PATH_CELL_BLOCKED = "path_cell_blocked"
    PATH_TEMPERATURE_BLOCKED = "path_temperature_blocked"
    PATH_CO_BLOCKED = "path_co_blocked"
    EXIT_BLOCKED = "exit_blocked"
    EXIT_UNSAFE_FIRE = "exit_unsafe_fire"
    VICTIM_FOLLOW_FAILURE = "victim_follow_failure"
    SAFER_EXIT_AVAILABLE = "safer_exit_available"
    PERIODIC_REEVALUATION = "periodic_reevaluation"
    DISTANCE_REEVALUATION = "distance_reevaluation"
    PATH_INVALID = "path_invalid"


@dataclass(frozen=True)
class ReplanDecision:
    required: bool
    immediate_stop: bool
    invalidate_current_path: bool
    reason: ReplanReason
    priority: ReplanPriority
    affected_cell_grid: tuple[int, int] | None = None
    alternative_exit_id: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reason"] = self.reason.value
        result["priority"] = int(self.priority)
        return result


@dataclass(frozen=True)
class EventReplanningConfig:
    enabled: bool = True
    periodic_enabled: bool = True
    periodic_interval_s: float = 5.0
    periodic_travel_distance_m: float = 2.0
    safer_exit_enabled: bool = True
    minimum_cost_improvement_ratio: float = 0.15
    switch_cooldown_s: float = 5.0
    maximum_follow_distance_m: float = 2.5
    follow_progress_timeout_s: float = 3.0
    temperature_block_c: float = 60.0
    temperature_release_c: float = 55.0
    co_block_ppm: float = 1600.0
    co_release_ppm: float = 1400.0
    release_confirmation_observations: int = 3
    minimum_replan_interval_s: float = 0.2
    ignore_same_reason_same_revision: bool = True

    def __post_init__(self) -> None:
        for name in (
            "enabled", "periodic_enabled", "safer_exit_enabled",
            "ignore_same_reason_same_revision",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in (
            "periodic_interval_s", "periodic_travel_distance_m",
            "switch_cooldown_s", "follow_progress_timeout_s",
            "minimum_replan_interval_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.periodic_enabled and self.periodic_interval_s <= 0:
            raise ValueError("periodic_interval_s must be positive when enabled")
        if not 0.0 <= self.minimum_cost_improvement_ratio < 1.0:
            raise ValueError("minimum_cost_improvement_ratio must be in [0,1)")
        if self.maximum_follow_distance_m <= 0:
            raise ValueError("maximum_follow_distance_m must be positive")
        if self.temperature_release_c >= self.temperature_block_c:
            raise ValueError("temperature_release_c must be below temperature_block_c")
        if self.co_release_ppm >= self.co_block_ppm:
            raise ValueError("co_release_ppm must be below co_block_ppm")
        if (
            isinstance(self.release_confirmation_observations, bool)
            or not isinstance(self.release_confirmation_observations, int)
            or self.release_confirmation_observations < 1
        ):
            raise ValueError("release_confirmation_observations must be >= 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None):
        raw = dict(values or {})
        allowed_top = {
            "enabled", "max_replan_attempts", "cooldown_seconds",
            "periodic_reevaluation", "safer_exit", "victim_follow",
            "hysteresis", "duplicate_suppression",
        }
        unknown = set(raw) - allowed_top
        if unknown:
            raise ValueError(f"unknown replanning settings: {sorted(unknown)}")
        periodic = _section(raw, "periodic_reevaluation", {
            "enabled", "interval_s", "travel_distance_m",
        })
        safer = _section(raw, "safer_exit", {
            "enabled", "minimum_cost_improvement_ratio", "switch_cooldown_s",
        })
        victim = _section(raw, "victim_follow", {
            "maximum_follow_distance_m", "progress_timeout_s",
            "use_follow_wait_before_replan",
        })
        if victim.get("use_follow_wait_before_replan", True) is not True:
            raise ValueError("use_follow_wait_before_replan must remain true")
        hysteresis = _section(raw, "hysteresis", {
            "temperature_block_c", "temperature_release_c", "co_block_ppm",
            "co_release_ppm", "block_confirmation_observations",
            "release_confirmation_observations",
        })
        block_count = hysteresis.get("block_confirmation_observations", 1)
        if isinstance(block_count, bool) or not isinstance(block_count, int) or block_count < 1:
            raise ValueError("block_confirmation_observations must be >= 1")
        duplicate = _section(raw, "duplicate_suppression", {
            "minimum_replan_interval_s", "ignore_same_reason_same_revision",
        })
        return cls(
            enabled=raw.get("enabled", True),
            periodic_enabled=periodic.get("enabled", True),
            periodic_interval_s=periodic.get("interval_s", 5.0),
            periodic_travel_distance_m=periodic.get("travel_distance_m", 2.0),
            safer_exit_enabled=safer.get("enabled", True),
            minimum_cost_improvement_ratio=safer.get(
                "minimum_cost_improvement_ratio", 0.15
            ),
            switch_cooldown_s=safer.get("switch_cooldown_s", 5.0),
            maximum_follow_distance_m=victim.get("maximum_follow_distance_m", 2.5),
            follow_progress_timeout_s=victim.get("progress_timeout_s", 3.0),
            temperature_block_c=hysteresis.get("temperature_block_c", 60.0),
            temperature_release_c=hysteresis.get("temperature_release_c", 55.0),
            co_block_ppm=hysteresis.get("co_block_ppm", 1600.0),
            co_release_ppm=hysteresis.get("co_release_ppm", 1400.0),
            release_confirmation_observations=hysteresis.get(
                "release_confirmation_observations", 3
            ),
            minimum_replan_interval_s=duplicate.get("minimum_replan_interval_s", 0.2),
            ignore_same_reason_same_revision=duplicate.get(
                "ignore_same_reason_same_revision", True
            ),
        )


def _section(raw, name, allowed):
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"replanning.{name} must be a mapping")
    value = dict(value)
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown replanning.{name} settings: {sorted(unknown)}")
    return value


class EventReplanningPolicy:
    """Select at most one highest-priority event for each update."""

    def __init__(self, config: EventReplanningConfig):
        self.config = config
        self.last_processed_costmap_revision: int | None = None
        self.last_replan_reason = ReplanReason.NONE
        self.last_replan_time = -math.inf
        self.last_replan_robot_pose: tuple[float, float] | None = None
        self.last_selected_exit_id: str | None = None
        self.last_exit_evaluation_revision: int | None = None
        self._last_switch_time = -math.inf
        self._last_switch_pair: tuple[int, str, str] | None = None
        self._temperature_latched = False
        self._co_latched = False
        self._temperature_release_count = 0
        self._co_release_count = 0

    def evaluate(
        self, *, current_path: Sequence[tuple[int, int]],
        current_costmap: np.ndarray, costmap_revision: int,
        dynamic_obstacle_map: np.ndarray | None = None,
        temperature_map: np.ndarray | None = None,
        co_map: np.ndarray | None = None,
        temperature_observed_mask: np.ndarray | None = None,
        co_observed_mask: np.ndarray | None = None,
        exit_statuses: Mapping[str, ExitStatus] | None = None,
        current_exit_id: str | None = None,
        robot_pose: tuple[float, float] = (0.0, 0.0), elapsed_time: float = 0.0,
        current_exit_cost: float | None = None,
        alternative_exit_costs: Mapping[str, float] | None = None,
        victim_follow_active: bool = False,
        victim_follow_distance_m: float | None = None,
        victim_progress_stalled: bool = False,
        victim_path_blocked: bool = False,
    ) -> ReplanDecision:
        if not self.config.enabled:
            return self._none()
        costmap = np.asarray(current_costmap, dtype=float)
        if costmap.ndim != 2:
            raise ValueError("current_costmap must be a 2-D [row,col] array")
        if dynamic_obstacle_map is not None:
            dynamic_obstacle_map = np.asarray(dynamic_obstacle_map, dtype=bool)
            if dynamic_obstacle_map.shape != costmap.shape:
                raise ValueError(
                    "dynamic_obstacle_map must match current_costmap shape"
                )
        path = tuple((int(col), int(row)) for col, row in current_path)
        self._validate_pose(robot_pose)
        candidates: list[ReplanDecision] = []
        for cell in path:
            col, row = cell
            if row < 0 or col < 0 or row >= costmap.shape[0] or col >= costmap.shape[1]:
                candidates.append(self._decision(
                    ReplanReason.PATH_INVALID, ReplanPriority.PATH_BLOCKED,
                    cell, "remaining path leaves costmap",
                ))
                break
            if dynamic_obstacle_map is not None and bool(dynamic_obstacle_map[row, col]):
                candidates.append(self._decision(
                    ReplanReason.DYNAMIC_OBSTACLE_ON_PATH,
                    ReplanPriority.PATH_BLOCKED, cell,
                ))
                break
            if not np.isfinite(costmap[row, col]):
                candidates.append(self._decision(
                    ReplanReason.PATH_CELL_BLOCKED, ReplanPriority.PATH_BLOCKED, cell,
                ))
                break
        hazard = self._hazard_decision(
            path, costmap.shape, temperature_map, co_map,
            temperature_observed_mask, co_observed_mask,
        )
        if hazard is not None:
            candidates.append(hazard)
        status = None if current_exit_id is None else (exit_statuses or {}).get(current_exit_id)
        if status is ExitStatus.BLOCKED:
            candidates.append(self._decision(
                ReplanReason.EXIT_BLOCKED, ReplanPriority.EXIT_INVALID,
                detail=f"exit={current_exit_id}",
            ))
        elif status is ExitStatus.DANGEROUS:
            candidates.append(self._decision(
                ReplanReason.EXIT_UNSAFE_FIRE, ReplanPriority.EXIT_INVALID,
                detail=f"exit={current_exit_id}",
            ))
        if victim_follow_active and (
            victim_path_blocked or victim_progress_stalled
            or victim_follow_distance_m is not None
            and victim_follow_distance_m > self.config.maximum_follow_distance_m
        ):
            candidates.append(ReplanDecision(
                True, True, bool(victim_path_blocked),
                ReplanReason.VICTIM_FOLLOW_FAILURE,
                ReplanPriority.VICTIM_FOLLOW,
                detail="follow_wait_before_replan",
            ))
        safer = self._safer_exit(
            current_exit_id, current_exit_cost, alternative_exit_costs,
            exit_statuses or {}, int(costmap_revision), float(elapsed_time),
        )
        if safer is not None:
            candidates.append(safer)
        periodic = self._periodic(robot_pose, elapsed_time)
        if periodic is not None:
            candidates.append(periodic)
        if not candidates:
            return self._none()
        decision = max(candidates, key=lambda item: int(item.priority))
        if self._suppressed(decision, int(costmap_revision), float(elapsed_time)):
            return self._none()
        return decision

    def mark_processed(
        self, decision: ReplanDecision, *, costmap_revision: int,
        elapsed_time: float, robot_pose: tuple[float, float],
        selected_exit_id: str | None,
    ) -> None:
        self.last_processed_costmap_revision = int(costmap_revision)
        self.last_replan_reason = decision.reason
        self.last_replan_time = float(elapsed_time)
        self.last_replan_robot_pose = tuple(map(float, robot_pose))
        self.last_selected_exit_id = selected_exit_id
        if decision.reason is ReplanReason.SAFER_EXIT_AVAILABLE:
            self._last_switch_time = float(elapsed_time)
            self._last_switch_pair = (
                int(costmap_revision), selected_exit_id or "",
                decision.alternative_exit_id or "",
            )

    def mark_reevaluation_complete(
        self, *, elapsed_time: float, robot_pose: tuple[float, float],
        costmap_revision: int,
    ) -> None:
        self.last_replan_time = float(elapsed_time)
        self.last_replan_robot_pose = tuple(map(float, robot_pose))
        self.last_exit_evaluation_revision = int(costmap_revision)

    def _hazard_decision(self, path, shape, temp, co, temp_seen, co_seen):
        temp_values = self._observed_values(path, shape, temp, temp_seen)
        co_values = self._observed_values(path, shape, co, co_seen)
        if any(value >= self.config.temperature_block_c for _, value in temp_values):
            self._temperature_latched = True
            self._temperature_release_count = 0
            cell = next(cell for cell, value in temp_values if value >= self.config.temperature_block_c)
            return self._decision(
                ReplanReason.PATH_TEMPERATURE_BLOCKED,
                ReplanPriority.HAZARD_BLOCKED, cell,
            )
        self._update_release("temperature", temp_values, self.config.temperature_release_c)
        if self._temperature_latched:
            cell = temp_values[0][0] if temp_values else None
            return self._decision(
                ReplanReason.PATH_TEMPERATURE_BLOCKED,
                ReplanPriority.HAZARD_BLOCKED, cell,
                "temperature hysteresis latch remains active",
            )
        if any(value >= self.config.co_block_ppm for _, value in co_values):
            self._co_latched = True
            self._co_release_count = 0
            cell = next(cell for cell, value in co_values if value >= self.config.co_block_ppm)
            return self._decision(
                ReplanReason.PATH_CO_BLOCKED, ReplanPriority.HAZARD_BLOCKED, cell,
            )
        self._update_release("co", co_values, self.config.co_release_ppm)
        if self._co_latched:
            cell = co_values[0][0] if co_values else None
            return self._decision(
                ReplanReason.PATH_CO_BLOCKED,
                ReplanPriority.HAZARD_BLOCKED, cell,
                "CO hysteresis latch remains active",
            )
        return None

    def _update_release(self, kind, values, threshold):
        latch_name = f"_{kind}_latched"
        count_name = f"_{kind}_release_count"
        if not getattr(self, latch_name) or not values:
            return
        if all(value < threshold for _, value in values):
            count = getattr(self, count_name) + 1
            setattr(self, count_name, count)
            if count >= self.config.release_confirmation_observations:
                setattr(self, latch_name, False)
                setattr(self, count_name, 0)
        else:
            setattr(self, count_name, 0)

    @staticmethod
    def _observed_values(path, shape, values, observed):
        if values is None or observed is None:
            return []
        values = np.asarray(values, dtype=float)
        observed = np.asarray(observed, dtype=bool)
        if values.shape != shape or observed.shape != shape:
            raise ValueError("belief hazard maps must match current_costmap shape")
        result = []
        for col, row in path:
            if 0 <= row < shape[0] and 0 <= col < shape[1] and observed[row, col]:
                value = float(values[row, col])
                if math.isfinite(value):
                    result.append(((col, row), value))
        return result

    def _safer_exit(self, current_id, current_cost, alternatives, statuses, revision, now):
        if (
            not self.config.safer_exit_enabled or current_id is None
            or current_cost is None or not math.isfinite(float(current_cost))
            or float(current_cost) <= 0 or now - self._last_switch_time < self.config.switch_cooldown_s
        ):
            return None
        valid = []
        for exit_id, cost in (alternatives or {}).items():
            if exit_id == current_id or statuses.get(exit_id) in (
                ExitStatus.BLOCKED, ExitStatus.DANGEROUS,
            ):
                continue
            if math.isfinite(float(cost)) and float(cost) >= 0:
                valid.append((float(cost), str(exit_id)))
        if not valid:
            return None
        new_cost, new_id = min(valid)
        improvement = (float(current_cost) - new_cost) / float(current_cost)
        pair = (revision, current_id, new_id)
        if improvement + 1e-12 < self.config.minimum_cost_improvement_ratio:
            return None
        if pair == self._last_switch_pair:
            return None
        return ReplanDecision(
            True, False, True, ReplanReason.SAFER_EXIT_AVAILABLE,
            ReplanPriority.SAFER_EXIT, alternative_exit_id=new_id,
            detail=f"cost improvement={improvement:.3f}",
        )

    def _periodic(self, pose, now):
        if not self.config.periodic_enabled:
            return None
        if self.last_replan_robot_pose is None:
            self.last_replan_robot_pose = tuple(map(float, pose))
            self.last_replan_time = float(now)
            return None
        if now - self.last_replan_time >= self.config.periodic_interval_s:
            return ReplanDecision(
                True, False, False, ReplanReason.PERIODIC_REEVALUATION,
                ReplanPriority.PERIODIC,
            )
        if self.config.periodic_travel_distance_m > 0 and math.dist(
            tuple(map(float, pose)), self.last_replan_robot_pose
        ) >= self.config.periodic_travel_distance_m:
            return ReplanDecision(
                True, False, False, ReplanReason.DISTANCE_REEVALUATION,
                ReplanPriority.PERIODIC,
            )
        return None

    def _suppressed(self, decision, revision, now):
        emergency = decision.priority >= ReplanPriority.HAZARD_BLOCKED
        same = (
            self.config.ignore_same_reason_same_revision
            and self.last_processed_costmap_revision == revision
            and self.last_replan_reason is decision.reason
        )
        if same:
            return True
        return (
            not emergency
            and now - self.last_replan_time < self.config.minimum_replan_interval_s
        )

    @staticmethod
    def _validate_pose(pose):
        if len(pose) != 2 or not all(math.isfinite(float(value)) for value in pose):
            raise ValueError("robot_pose must be finite world (x,y)")

    @staticmethod
    def _decision(reason, priority, cell=None, detail=None):
        return ReplanDecision(True, True, True, reason, priority, cell, detail=detail)

    @staticmethod
    def _none():
        return ReplanDecision(
            False, False, False, ReplanReason.NONE, ReplanPriority.NONE
        )
