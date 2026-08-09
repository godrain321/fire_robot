"""Current-pose exit exploration using the existing cost-aware A* evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from world.entities import ExitStatus


class ExplorationPhase(Enum):
    INITIAL = "initial"
    RESUMED_AFTER_EVACUATION = "resumed_after_evacuation"
    REPLAN_AFTER_MAP_CHANGE = "replan_after_map_change"


@dataclass(frozen=True)
class ExplorationConfig:
    enabled: bool = True
    start_after_evacuation: bool = True
    use_current_robot_pose_as_start: bool = True
    teleport_to_entrance_on_start: bool = False
    select_exit_by_astar_path_length: bool = True
    reselect_after_each_exit_check: bool = True
    resume_after_victim_evacuation: bool = True
    replan_from_current_pose_on_resume: bool = True
    return_to_entrance_when_complete: bool = False
    return_entrance_exit_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "enabled", "start_after_evacuation", "use_current_robot_pose_as_start",
            "teleport_to_entrance_on_start", "select_exit_by_astar_path_length",
            "reselect_after_each_exit_check", "resume_after_victim_evacuation",
            "replan_from_current_pose_on_resume", "return_to_entrance_when_complete",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.teleport_to_entrance_on_start:
            raise ValueError("exploration must never teleport to the entrance")
        if not self.use_current_robot_pose_as_start:
            raise ValueError("exploration must use the current robot pose")
        if not self.select_exit_by_astar_path_length:
            raise ValueError("exploration requires A* path-length selection")
        if not self.replan_from_current_pose_on_resume:
            raise ValueError("exploration resume must replan from current pose")
        if self.return_entrance_exit_id is not None and not str(
            self.return_entrance_exit_id
        ).strip():
            raise ValueError("return_entrance_exit_id cannot be empty")
        if self.return_to_entrance_when_complete and self.return_entrance_exit_id is None:
            raise ValueError(
                "return_to_entrance_when_complete requires return_entrance_exit_id"
            )

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown exploration settings: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class ExplorationPlan:
    success: bool
    phase: ExplorationPhase
    start_position_world: tuple[float, float]
    target_exit_id: str | None
    target_position_world: tuple[float, float] | None
    path_grid: tuple[tuple[int, int], ...]
    path_world: tuple[tuple[float, float], ...]
    exit_path_lengths_m: tuple[tuple[str, float], ...]
    costmap_revision: int
    created_at: float
    failure_reason: str | None
    exit_failure_reasons: tuple[tuple[str, tuple[str, ...]], ...] = tuple()
    evacuation_plan: Any | None = None

    def to_dict(self):
        result = asdict(self)
        result["phase"] = self.phase.value
        result["evacuation_plan"] = (
            None if self.evacuation_plan is None
            else self.evacuation_plan.to_dict()
        )
        return result


class ExplorationManager:
    def __init__(self, evacuation_planner, config: ExplorationConfig):
        self.evacuation_planner = evacuation_planner
        self.config = config

    def plan_next_exit(
        self, world_state, current_position_world, *, cost_map,
        costmap_revision: int, created_at: float, phase: ExplorationPhase,
    ) -> ExplorationPlan:
        start = tuple(map(float, current_position_world))
        unchecked_exits = world_state.get_unchecked_exits()
        unknown_exits = tuple(
            item for item in unchecked_exits
            if item.status not in (
                ExitStatus.BLOCKED, ExitStatus.DANGEROUS,
                ExitStatus.DANGER_EXPECTED,
            )
        )
        if not unchecked_exits:
            return ExplorationPlan(
                False, phase, start, None, None, tuple(), tuple(), tuple(),
                int(costmap_revision), float(created_at),
                "no_unchecked_exits", tuple(), None,
            )
        if not unknown_exits:
            failures = tuple(
                (item.exit_id, (item.status.value,)) for item in unchecked_exits
            )
            return ExplorationPlan(
                False, phase, start, None, None, tuple(), tuple(), tuple(),
                int(costmap_revision), float(created_at),
                "unchecked_exits_blocked_or_unsafe", failures, None,
            )
        dynamic = world_state.dynamic_obstacle_mask()
        effective = cost_map.copy()
        effective[world_state.static_obstacle_map] = float("inf")
        effective[dynamic] = float("inf")
        effective[world_state.estimated_fire_map.blocked_mask] = float("inf")
        plan = self.evacuation_planner.plan(
            unknown_exits, start, cost_map=effective,
            static_obstacle_map=world_state.static_obstacle_map,
            dynamic_obstacle_map=dynamic,
            estimated_fire_map=world_state.estimated_fire_map,
            created_at=created_at,
        )
        world_state.record_exit_evaluations(plan)
        world_state.clear_active_evacuation_plan()
        lengths = tuple(sorted(
            (item.exit_id, item.path_length_m)
            for item in plan.all_evaluations if item.path_length_m is not None
        ))
        failures = tuple(sorted(
            (
                item.exit_id,
                tuple(reason.value for reason in item.rejection_reasons)
                or (("no_path",) if not item.reachable else tuple()),
            )
            for item in plan.all_evaluations if not item.accepted
        ))
        world_state.record_exploration_reachability(
            plan.all_evaluations, costmap_revision=costmap_revision,
            evaluated_at=created_at,
        )
        if not plan.success:
            return ExplorationPlan(
                False, phase, start, None, None, tuple(), tuple(), lengths,
                int(costmap_revision), float(created_at),
                plan.failure_reason.value, failures, plan,
            )
        return ExplorationPlan(
            True, phase, start, plan.selected_exit_id,
            plan.selected_approach_position_world, plan.path_grid,
            plan.path_world, lengths, int(costmap_revision),
            float(created_at), None, failures, plan,
        )
