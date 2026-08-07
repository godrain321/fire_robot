"""Integrated registry for factory_v3 entities and planner-grid map state."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math
from typing import Any

import numpy as np

from .entities import (
    DynamicObstacle, DynamicObstacleShape, DynamicObstacleStatus,
    Exit, ExitCheckRecord, ExitStatus, ExitVisitStatus,
    ExplorationInterruption, Victim, VictimStatus,
)
from .fire_maps import EstimatedFireMap, GroundTruthFireMap, MapMetadata


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def _enum(enum_type, value, field_name):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"invalid {field_name} {value!r}; expected one of: {allowed}") from exc


class WorldState:
    """Own entity state without performing sensing, planning, or rendering."""

    def __init__(
        self,
        map_metadata: MapMetadata,
        static_obstacle_map,
        *,
        ground_truth_fire_map: GroundTruthFireMap | None = None,
        estimated_fire_map: EstimatedFireMap | None = None,
        temperature_blocked_c: float = 60.0,
        co_blocked_ppm: float = 1600.0,
    ) -> None:
        self.map_metadata = map_metadata
        static = np.asarray(static_obstacle_map, dtype=bool)
        expected = (map_metadata.height, map_metadata.width)
        if static.shape != expected:
            raise ValueError(f"static obstacle shape={static.shape}, expected={expected}")
        self.static_obstacle_map = static.copy()
        self.static_obstacle_map.setflags(write=False)
        # Defaults to the planner layer; the simulator may replace this with
        # its non-inflated SLAM occupancy through the explicit setter below.
        self.known_occupancy_map = static.copy()
        self.known_occupancy_map.setflags(write=False)
        self.dynamic_obstacles: dict[str, DynamicObstacle] = {}
        self.exits: dict[str, Exit] = {}
        self.victims: dict[str, Victim] = {}
        self.ground_truth_fire_map = ground_truth_fire_map or GroundTruthFireMap(
            map_metadata,
            temperature_blocked_c=temperature_blocked_c,
            co_blocked_ppm=co_blocked_ppm,
        )
        self.estimated_fire_map = estimated_fire_map or EstimatedFireMap(
            map_metadata,
            temperature_blocked_c=temperature_blocked_c,
            co_blocked_ppm=co_blocked_ppm,
        )
        if self.ground_truth_fire_map is self.estimated_fire_map:
            raise ValueError("ground truth and estimated fire maps must be separate objects")
        if self.ground_truth_fire_map.metadata != map_metadata or self.estimated_fire_map.metadata != map_metadata:
            raise ValueError("fire map metadata must match WorldState metadata")
        self.simulation_time = 0.0
        self.robot_position_world: tuple[float, float] | None = None
        self.travel_history = None
        self.active_return_plan = None
        self.latest_exit_evaluations: dict[str, Any] = {}
        self.exit_evaluation_history: list[tuple[Any, ...]] = []
        self.active_evacuation_plan = None
        self.mission_start_position_world: tuple[float, float] | None = None
        self.mission_entry_id: str | None = None
        self.mission_entry_position_world: tuple[float, float] | None = None
        self.hazard_knowledge_decision = None
        self.fire_localization_result = None
        self.latest_exit_blockage_results: dict[str, Any] = {}
        self.last_perception_replan_reason: str | None = None
        self.active_route_decision = None
        self.active_route_valid = False
        self.active_route_invalid_reason: str | None = None
        self.active_route_blocked_grid: tuple[int, int] | None = None
        self.active_route_created_at: float | None = None
        self.active_route_last_validated_at: float | None = None
        self.active_route_costmap_revision: int | None = None
        self.costmap_revision = 0
        self.environment_revision = 0
        self.route_replan_count = 0
        self.final_route_failure_reason: str | None = None
        self.active_path_simplification = None
        self.path_simplification_history: list[Any] = []
        self.current_target_exit_id: str | None = None
        self.current_target_selection_reason: str | None = None
        self.current_target_is_usable = False
        self.exit_path_lengths_m: dict[str, float] = {}
        self.remaining_route_cost: dict[str, float] | None = None
        self.route_cost_baseline: float | None = None
        self.route_cost_history: list[Any] = []
        self.consecutive_route_cost_increases = 0
        self.route_costmap_revision: int | None = None
        self.exit_switch_occurred = False
        self.previous_target_exit_id: str | None = None
        self.last_exit_switch_reason: str | None = None
        self.last_exit_switch_at: float | None = None
        self.exit_switch_cooldown_until: float | None = None
        self.last_exit_switch_validation: str | None = None
        self.initial_robot_pose_world: tuple[float, float, float] | None = None
        self.exploration_entry_pose_world: tuple[float, float, float] | None = None
        self.exploration_resume_pose_world: tuple[float, float, float] | None = None
        self.robot_grid_position: tuple[int, int] | None = None
        self.current_exploration_target_exit_id: str | None = None
        self.exploration_plan_start_world: tuple[float, float] | None = None
        self.exploration_exit_path_lengths_m: dict[str, float] = {}
        self.exploration_costmap_revision: int | None = None
        self.exploration_environment_revision: int | None = None
        self.exploration_phase: str | None = None
        self.exploration_return_target_exit_id: str | None = None
        self.exploration_return_to_entrance_enabled = False
        self.active_exploration_plan = None
        self.exit_visit_status: dict[str, ExitVisitStatus] = {}
        self.exit_check_history: list[ExitCheckRecord] = []
        self.exit_reachability_by_revision: dict[str, dict[str, Any]] = {}
        self.exploration_interruptions: list[ExplorationInterruption] = []
        self.exploration_stalled = False
        self.exploration_stall_reason: str | None = None
        self.active_following_victim_id: str | None = None
        self.victim_following_controller = None
        self.victim_following_config = None
        self.victim_following_events: list[dict[str, Any]] = []

    @classmethod
    def from_scenario(cls, scenario: dict[str, Any], grid_map, config) -> "WorldState":
        """Load the existing factory_v3 YAML schema without duplicating coordinates."""
        metadata = MapMetadata.from_grid_map(grid_map)
        world = cls(
            metadata,
            np.asarray(grid_map.occupancy, dtype=bool),
            temperature_blocked_c=config.temperature_blocked,
            co_blocked_ppm=config.co_blocked,
        )
        for item in scenario.get("exits", []):
            marker = item.get("marker")
            approach = item.get("approach")
            if marker is None:
                raise ValueError(f"exit {item.get('id')!r} requires marker")
            world.add_exit(Exit(
                exit_id=str(item["id"]),
                position_world=(marker["x"], marker["y"]),
                approach_position_world=(approach["x"], approach["y"])
                if approach is not None else None,
                status=_enum(ExitStatus, item.get("initial_status", "unknown"), "exit status"),
            ))
        for item in scenario.get("humans", []):
            world.add_victim(Victim(
                victim_id=str(item["id"]),
                position_world=(item["x"], item["y"]),
                status=_enum(VictimStatus, item.get("initial_status", "undetected"), "victim status"),
            ))
        for item in scenario.get("dynamic_obstacles", []):
            world.add_dynamic_obstacle(DynamicObstacle(
                obstacle_id=str(item["id"]),
                position_world=tuple(item["position_world"]),
                shape=_enum(DynamicObstacleShape, item.get("shape", "point"), "obstacle shape"),
                size_m=tuple(item.get("size_m", (0.0, 0.0))),
                status=_enum(DynamicObstacleStatus, item.get("initial_status", "unknown"), "obstacle status"),
                source=str(item.get("source", "config")),
                confidence=float(item.get("confidence", 0.0)),
            ))
        world.validate_all_entities()
        return world

    def set_simulation_time(self, sim_time: float) -> None:
        if not math.isfinite(float(sim_time)) or float(sim_time) < 0:
            raise ValueError("simulation time must be finite and non-negative")
        self.simulation_time = float(sim_time)

    def attach_travel_history(self, travel_history) -> None:
        if travel_history.map_metadata != self.map_metadata:
            raise ValueError("travel history metadata must match WorldState")
        self.travel_history = travel_history

    def set_mission_entry(self, entry_id: str, position_world) -> None:
        self.validate_position(position_world, label="mission entry")
        self.mission_entry_id = str(entry_id)
        self.mission_entry_position_world = (
            float(position_world[0]), float(position_world[1])
        )
        if self.mission_start_position_world is None:
            self.mission_start_position_world = self.mission_entry_position_world

    def set_initial_robot_pose(self, position_world, yaw_rad: float) -> None:
        col, row = self.validate_position(position_world, label="initial robot pose")
        if not math.isfinite(float(yaw_rad)):
            raise ValueError("initial robot yaw must be finite")
        self.initial_robot_pose_world = (
            float(position_world[0]), float(position_world[1]), float(yaw_rad)
        )
        self.mission_start_position_world = self.initial_robot_pose_world[:2]
        self.robot_position_world = self.initial_robot_pose_world[:2]
        self.robot_grid_position = (col, row)

    def record_exploration_plan(self, plan, *, yaw_rad: float) -> None:
        if not plan.success:
            raise ValueError("cannot activate an unsuccessful exploration plan")
        pose = (
            float(plan.start_position_world[0]),
            float(plan.start_position_world[1]), float(yaw_rad),
        )
        if plan.phase.value == "initial" and self.exploration_entry_pose_world is None:
            self.exploration_entry_pose_world = pose
        elif plan.phase.value == "resumed_after_evacuation":
            self.exploration_resume_pose_world = pose
        self.current_exploration_target_exit_id = plan.target_exit_id
        self.exploration_plan_start_world = tuple(plan.start_position_world)
        self.exploration_exit_path_lengths_m = dict(plan.exit_path_lengths_m)
        self.exploration_costmap_revision = int(plan.costmap_revision)
        self.exploration_environment_revision = self.environment_revision
        self.exploration_phase = plan.phase.value
        self.active_exploration_plan = plan

    def configure_exploration_return(
        self, *, enabled: bool, entrance_exit_id: str | None,
    ) -> None:
        if enabled and entrance_exit_id is None:
            raise ValueError("enabled exploration return requires an entrance exit ID")
        if entrance_exit_id is not None:
            self.get_exit(entrance_exit_id)
        self.exploration_return_to_entrance_enabled = bool(enabled)
        self.exploration_return_target_exit_id = entrance_exit_id

    def clear_active_exploration_plan(self) -> None:
        self.active_exploration_plan = None
        self.current_exploration_target_exit_id = None

    def get_unchecked_exits(self) -> tuple[Exit, ...]:
        return tuple(
            item for item in self.exits.values()
            if self.exit_visit_status[item.exit_id] is ExitVisitStatus.UNCHECKED
        )

    def record_exploration_reachability(
        self, evaluations, *, costmap_revision: int, evaluated_at: float,
    ) -> None:
        for item in evaluations:
            self.exit_reachability_by_revision[item.exit_id] = {
                "reachable": bool(item.reachable),
                "accepted": bool(item.accepted),
                "reasons": [reason.value for reason in item.rejection_reasons],
                "costmap_revision": int(costmap_revision),
                "evaluated_at": float(evaluated_at),
            }

    def record_exit_check(
        self, exit_id: str, status: ExitStatus, *, sim_time: float,
        costmap_revision: int, reason: str,
    ) -> ExitCheckRecord:
        self.update_exit_status(
            exit_id, status, sim_time=sim_time,
            reason=reason if status in (ExitStatus.BLOCKED, ExitStatus.DANGEROUS)
            else None,
        )
        self.exit_visit_status[exit_id] = ExitVisitStatus.CHECKED
        record = ExitCheckRecord(
            exit_id, ExitVisitStatus.CHECKED, status, float(sim_time),
            int(costmap_revision), str(reason),
        )
        self.exit_check_history.append(record)
        return record

    def record_exploration_interruption(
        self, *, sim_time: float, reason: str, victim_id: str,
        robot_pose_world, target_exit_id, active_path_grid,
        costmap_revision: int,
    ) -> ExplorationInterruption:
        record = ExplorationInterruption(
            float(sim_time), str(reason), str(victim_id),
            tuple(map(float, robot_pose_world)),
            None if target_exit_id is None else str(target_exit_id),
            tuple((int(col), int(row)) for col, row in active_path_grid),
            int(costmap_revision), True,
        )
        self.exploration_interruptions.append(record)
        return record

    def mark_exploration_stalled(self, reason: str) -> None:
        self.exploration_stalled = True
        self.exploration_stall_reason = str(reason)

    def clear_exploration_stall(self) -> None:
        self.exploration_stalled = False
        self.exploration_stall_reason = None

    def update_costmap_revision(self, revision: int) -> None:
        if isinstance(revision, bool) or int(revision) < self.costmap_revision:
            raise ValueError("costmap revision must be monotonic and non-negative")
        self.costmap_revision = int(revision)

    def set_active_route_decision(self, decision) -> None:
        if not decision.success:
            raise ValueError("cannot activate an unsuccessful route decision")
        self.active_route_decision = decision
        self.hazard_knowledge_decision = decision.hazard_knowledge
        self.active_route_valid = True
        self.active_route_invalid_reason = None
        self.active_route_blocked_grid = None
        self.active_route_created_at = decision.created_at
        self.active_route_last_validated_at = decision.created_at
        self.active_route_costmap_revision = decision.costmap_revision
        self.final_route_failure_reason = None

    def invalidate_active_route(self, reason: str, blocked_grid=None) -> None:
        self.active_route_valid = False
        self.active_route_invalid_reason = str(reason)
        self.active_route_blocked_grid = (
            None if blocked_grid is None
            else (int(blocked_grid[0]), int(blocked_grid[1]))
        )

    def clear_active_route(self) -> None:
        self.active_route_decision = None
        self.active_route_valid = False
        self.active_path_simplification = None

    def record_path_simplification(self, result) -> None:
        if not result.success:
            raise ValueError("cannot activate a failed path simplification")
        self.active_path_simplification = result
        self.path_simplification_history.append(result)

    def clear_path_simplification(self) -> None:
        self.active_path_simplification = None

    def record_route_validation(self, *, sim_time: float, costmap_revision: int) -> None:
        self.active_route_last_validated_at = float(sim_time)
        self.active_route_costmap_revision = int(costmap_revision)

    def record_robot_position(
        self, position_world, *, sim_time: float | None = None,
        is_returning: bool = False, force_reason=None,
    ):
        self.validate_position(position_world, label="robot")
        self.robot_position_world = (
            float(position_world[0]), float(position_world[1])
        )
        self.robot_grid_position = self.map_metadata.world_to_grid(
            *self.robot_position_world
        )
        if self.travel_history is None:
            raise RuntimeError("no TravelHistory is attached")
        return self.travel_history.record_position(
            self.robot_position_world,
            recorded_at=self.simulation_time if sim_time is None else sim_time,
            is_returning=is_returning,
            force_reason=force_reason,
        )

    def get_travel_history(self):
        if self.travel_history is None:
            raise RuntimeError("no TravelHistory is attached")
        return self.travel_history

    def set_active_return_plan(self, plan) -> None:
        if not plan.success:
            raise ValueError("cannot activate an unsuccessful return plan")
        self.active_return_plan = plan

    def clear_active_return_plan(self) -> None:
        self.active_return_plan = None

    def record_exit_evaluations(self, evacuation_plan) -> None:
        evaluations = tuple(evacuation_plan.all_evaluations)
        self.latest_exit_evaluations = {
            item.exit_id: item for item in evaluations
        }
        for evaluation in evaluations:
            exit_item = self.get_exit(evaluation.exit_id)
            exit_item.last_checked_at = evaluation.evaluated_at
            exit_item.temperature_c = evaluation.exit_temperature_c
            exit_item.co_ppm = evaluation.exit_co_ppm
            exit_item.path_cost = evaluation.accumulated_risk_cost
            exit_item.metadata["last_evaluation"] = evaluation.to_dict()
        self.exit_evaluation_history.append(evaluations)
        self.active_evacuation_plan = (
            evacuation_plan if evacuation_plan.success else None
        )

    def record_exit_selection(
        self, exit_id: str, *, reason: str, costmap_revision: int,
        path_lengths_m=None,
    ) -> None:
        exit_item = self.get_exit(exit_id)
        self.current_target_exit_id = exit_item.exit_id
        self.current_target_selection_reason = str(reason)
        self.current_target_is_usable = exit_item.status is ExitStatus.USABLE
        self.route_costmap_revision = int(costmap_revision)
        self.exit_path_lengths_m = {
            str(key): float(value) for key, value in dict(path_lengths_m or {}).items()
        }

    def record_route_cost(self, sample, *, baseline, consecutive: int) -> None:
        self.remaining_route_cost = {
            "accumulated": float(sample.accumulated_cost),
            "average": float(sample.average_cost),
            "maximum": float(sample.maximum_cost),
        }
        self.route_cost_baseline = None if baseline is None else float(baseline)
        self.route_cost_history.append(sample)
        self.consecutive_route_cost_increases = int(consecutive)
        self.route_costmap_revision = int(sample.costmap_revision)

    def record_exit_switch(
        self, *, previous_exit_id: str, new_exit_id: str, reason: str,
        sim_time: float, cooldown_seconds: float, validation_result: str,
    ) -> None:
        self.exit_switch_occurred = True
        self.previous_target_exit_id = str(previous_exit_id)
        self.current_target_exit_id = str(new_exit_id)
        self.last_exit_switch_reason = str(reason)
        self.last_exit_switch_at = float(sim_time)
        self.exit_switch_cooldown_until = float(sim_time) + float(cooldown_seconds)
        self.last_exit_switch_validation = str(validation_result)

    def exit_switch_is_cooling_down(self, sim_time: float) -> bool:
        return (
            self.exit_switch_cooldown_until is not None
            and float(sim_time) < self.exit_switch_cooldown_until
        )

    def clear_active_evacuation_plan(self) -> None:
        self.active_evacuation_plan = None

    def validate_position(self, position_world, *, require_free: bool = True, label: str = "position") -> tuple[int, int]:
        x, y = float(position_world[0]), float(position_world[1])
        col, row = self.map_metadata.world_to_grid(x, y)
        if require_free and self.static_obstacle_map[row, col]:
            raise ValueError(f"{label} {(x, y)} is inside a static obstacle at {(col, row)}")
        return col, row

    def set_known_occupancy_map(self, occupancy_map) -> None:
        values = np.asarray(occupancy_map, dtype=bool)
        if values.shape != self.static_obstacle_map.shape:
            raise ValueError("known occupancy map shape mismatch")
        self.known_occupancy_map = values.copy()
        self.known_occupancy_map.setflags(write=False)

    def add_exit(self, exit_item: Exit) -> None:
        if exit_item.exit_id in self.exits:
            raise ValueError(f"duplicate exit_id: {exit_item.exit_id}")
        self.validate_position(exit_item.position_world, require_free=False, label=f"exit {exit_item.exit_id}")
        if exit_item.approach_position_world is not None:
            self.validate_position(exit_item.approach_position_world, label=f"exit approach {exit_item.exit_id}")
        self.exits[exit_item.exit_id] = exit_item
        self.exit_visit_status[exit_item.exit_id] = ExitVisitStatus.UNCHECKED

    def get_exit(self, exit_id: str) -> Exit:
        try:
            return self.exits[exit_id]
        except KeyError as exc:
            raise KeyError(f"unknown exit_id: {exit_id}") from exc

    def update_exit_status(self, exit_id: str, status: ExitStatus, **context) -> None:
        exit_item = self.get_exit(exit_id)
        previous = exit_item.status
        exit_item.update_status(status, sim_time=context.pop("sim_time", self.simulation_time), **context)
        if status is not previous:
            self.environment_revision += 1
        if self.current_target_exit_id == exit_id:
            self.current_target_is_usable = status is ExitStatus.USABLE

    def record_exit_blockage_result(self, result) -> None:
        if result.exit_id not in self.exits:
            raise KeyError(f"unknown exit_id: {result.exit_id}")
        self.latest_exit_blockage_results[result.exit_id] = result

    def add_victim(self, victim: Victim) -> None:
        if victim.victim_id in self.victims:
            raise ValueError(f"duplicate victim_id: {victim.victim_id}")
        self.validate_position(victim.position_world, label=f"victim {victim.victim_id}")
        self.victims[victim.victim_id] = victim

    def get_victim(self, victim_id: str) -> Victim:
        try:
            return self.victims[victim_id]
        except KeyError as exc:
            raise KeyError(f"unknown victim_id: {victim_id}") from exc

    def update_victim_status(self, victim_id: str, status: VictimStatus, **context) -> None:
        victim = self.get_victim(victim_id)
        victim.update_status(status, sim_time=context.pop("sim_time", self.simulation_time))
        for key, value in context.items():
            if not hasattr(victim, key):
                raise ValueError(f"unknown victim field: {key}")
            setattr(victim, key, value)

    def attach_victim_following(self, controller) -> None:
        if controller.metadata != self.map_metadata:
            raise ValueError("victim following metadata must match WorldState")
        self.victim_following_controller = controller
        self.victim_following_config = controller.config

    def start_victim_following(self, victim_id: str) -> None:
        victim = self.get_victim(victim_id)
        if victim.movable is not True:
            raise ValueError("only a movable victim may start following")
        self.active_following_victim_id = victim_id
        victim.current_grid_position = self.map_metadata.world_to_grid(
            *victim.position_world
        )

    def update_victim_following(self, update, *, sim_time: float) -> None:
        if self.active_following_victim_id is None:
            raise RuntimeError("no active following victim")
        victim = self.get_victim(self.active_following_victim_id)
        col, row = self.validate_position(
            update.position_world, label=f"victim {victim.victim_id} follow position"
        )
        victim.position_world = tuple(update.position_world)
        victim.current_grid_position = (col, row)
        victim.follow_target_world = update.target_world
        victim.follow_distance_m = float(update.robot_victim_distance_m)
        victim.follow_wait_active = update.state.value == "follow_wait"
        victim.follow_wait_started_at = (
            None if self.victim_following_controller is None
            else self.victim_following_controller.wait_started_at
        )
        victim.follow_reprompt_count = (
            0 if self.victim_following_controller is None
            else self.victim_following_controller.reprompt_count
        )
        victim.follow_failed = update.state.value == "follow_failed"
        victim.follow_failure_reason = (
            None if self.victim_following_controller is None
            else self.victim_following_controller.failure_reason
        )
        if update.event:
            self.victim_following_events.append({
                "event": update.event, "victim_id": victim.victim_id,
                "sim_time": float(sim_time),
                "distance_m": float(update.robot_victim_distance_m),
            })

    def complete_victim_following(self, victim_id: str, *, sim_time: float, reason: str) -> None:
        victim = self.get_victim(victim_id)
        if victim_id != self.active_following_victim_id:
            raise ValueError("victim is not the active follower")
        victim.evacuated_at = float(sim_time)
        victim.evacuation_success_reason = str(reason)
        victim.following_robot = False
        victim.follow_wait_active = False
        self.active_following_victim_id = None

    def clear_victim_following(self) -> None:
        self.active_following_victim_id = None

    def _obstacle_extent(self, obstacle: DynamicObstacle) -> tuple[float, float, float, float]:
        x, y = obstacle.position_world
        if obstacle.shape is DynamicObstacleShape.POINT:
            return x, x, y, y
        if obstacle.shape is DynamicObstacleShape.CIRCLE:
            radius = obstacle.size_m[0] / 2.0
            return x - radius, x + radius, y - radius, y + radius
        return (
            x - obstacle.size_m[0] / 2.0, x + obstacle.size_m[0] / 2.0,
            y - obstacle.size_m[1] / 2.0, y + obstacle.size_m[1] / 2.0,
        )

    def _validate_dynamic_obstacle(self, obstacle: DynamicObstacle) -> None:
        col, row = self.validate_position(
            obstacle.position_world, require_free=False,
            label=f"dynamic obstacle {obstacle.obstacle_id}",
        )
        if self.known_occupancy_map[row, col]:
            raise ValueError(
                f"dynamic obstacle {obstacle.obstacle_id} overlaps known static occupancy"
            )
        x1, x2, y1, y2 = self._obstacle_extent(obstacle)
        for point in ((x1, y1), (x2, y2)):
            if not self.map_metadata.is_world_position_in_bounds(*point):
                raise ValueError(
                    f"dynamic obstacle {obstacle.obstacle_id} extent leaves map; "
                    "out-of-bounds areas are not clipped"
                )

    def add_dynamic_obstacle(self, obstacle: DynamicObstacle, *, sim_time: float | None = None) -> None:
        if obstacle.obstacle_id in self.dynamic_obstacles:
            raise ValueError(f"duplicate obstacle_id: {obstacle.obstacle_id}")
        self._validate_dynamic_obstacle(obstacle)
        if sim_time is not None:
            obstacle.update(sim_time=sim_time)
        self.dynamic_obstacles[obstacle.obstacle_id] = obstacle
        self.environment_revision += 1

    def get_dynamic_obstacle(self, obstacle_id: str) -> DynamicObstacle:
        try:
            return self.dynamic_obstacles[obstacle_id]
        except KeyError as exc:
            raise KeyError(f"unknown obstacle_id: {obstacle_id}") from exc

    def update_dynamic_obstacle(self, obstacle_id: str, **changes) -> None:
        obstacle = self.get_dynamic_obstacle(obstacle_id)
        snapshot = (
            obstacle.position_world, obstacle.status, obstacle.confidence,
            obstacle.first_seen_at, obstacle.last_seen_at,
            obstacle.observation_count,
        )
        semantic_before = (
            obstacle.position_world, obstacle.status, obstacle.confidence,
        )
        obstacle.update(sim_time=changes.pop("sim_time", self.simulation_time), **changes)
        try:
            self._validate_dynamic_obstacle(obstacle)
        except Exception:
            (
                obstacle.position_world, obstacle.status, obstacle.confidence,
                obstacle.first_seen_at, obstacle.last_seen_at,
                obstacle.observation_count,
            ) = snapshot
            raise
        semantic_after = (
            obstacle.position_world, obstacle.status, obstacle.confidence,
        )
        if semantic_after != semantic_before:
            self.environment_revision += 1

    def clear_dynamic_obstacle(self, obstacle_id: str, *, sim_time: float | None = None) -> None:
        self.update_dynamic_obstacle(
            obstacle_id, status=DynamicObstacleStatus.CLEARED,
            sim_time=self.simulation_time if sim_time is None else sim_time,
        )

    def dynamic_obstacle_mask(self) -> np.ndarray:
        mask = np.zeros_like(self.static_obstacle_map, dtype=bool)
        for obstacle in self.get_active_dynamic_obstacles():
            center_col, center_row = self.map_metadata.world_to_grid(
                *obstacle.position_world
            )
            mask[center_row, center_col] = True
            if obstacle.shape is DynamicObstacleShape.POINT:
                continue
            x1, x2, y1, y2 = self._obstacle_extent(obstacle)
            for row in range(self.map_metadata.height):
                for col in range(self.map_metadata.width):
                    x, y = self.map_metadata.grid_to_world(col, row)
                    if obstacle.shape is DynamicObstacleShape.CIRCLE:
                        radius = obstacle.size_m[0] / 2.0
                        inside = math.hypot(x - obstacle.position_world[0], y - obstacle.position_world[1]) <= radius
                    else:
                        inside = x1 <= x <= x2 and y1 <= y <= y2
                    if inside:
                        mask[row, col] = True
        return mask

    def combined_obstacle_map(self) -> np.ndarray:
        return self.static_obstacle_map | self.dynamic_obstacle_mask()

    def get_active_dynamic_obstacles(self) -> tuple[DynamicObstacle, ...]:
        return tuple(item for item in self.dynamic_obstacles.values() if item.status in (DynamicObstacleStatus.ACTIVE, DynamicObstacleStatus.MOVING))

    def get_unrescued_victims(self) -> tuple[Victim, ...]:
        return tuple(item for item in self.victims.values() if not item.rescued and item.status is not VictimStatus.REPORTED)

    def get_usable_exits(self) -> tuple[Exit, ...]:
        return tuple(item for item in self.exits.values() if item.status is ExitStatus.USABLE)

    def validate_all_entities(self) -> None:
        for item in self.exits.values():
            self.validate_position(item.position_world, require_free=False, label=f"exit {item.exit_id}")
            if item.approach_position_world is not None:
                self.validate_position(item.approach_position_world, label=f"exit approach {item.exit_id}")
        for item in self.victims.values():
            self.validate_position(item.position_world, label=f"victim {item.victim_id}")
        for item in self.dynamic_obstacles.values():
            self._validate_dynamic_obstacle(item)

    def legacy_humans(self) -> list[dict[str, Any]]:
        return [{"id": item.victim_id, "x": item.position_world[0], "y": item.position_world[1]} for item in self.victims.values()]

    def legacy_exits(self) -> list[dict[str, Any]]:
        result = []
        for item in self.exits.values():
            approach = item.approach_position_world or item.position_world
            result.append({
                "id": item.exit_id,
                "marker": {"x": item.position_world[0], "y": item.position_world[1]},
                "approach": {"x": approach[0], "y": approach[1]},
            })
        return result

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "map_metadata": self.map_metadata,
            "simulation_time": self.simulation_time,
            "robot_position_world": self.robot_position_world,
            "travel_history": None if self.travel_history is None else self.travel_history.to_dict(),
            "active_return_plan": None if self.active_return_plan is None else self.active_return_plan.to_dict(),
            "latest_exit_evaluations": self.latest_exit_evaluations,
            "exit_evaluation_history": self.exit_evaluation_history,
            "active_evacuation_plan": None if self.active_evacuation_plan is None else self.active_evacuation_plan.to_dict(),
            "mission_start_position_world": self.mission_start_position_world,
            "mission_entry_id": self.mission_entry_id,
            "mission_entry_position_world": self.mission_entry_position_world,
            "hazard_knowledge_decision": self.hazard_knowledge_decision,
            "fire_localization_result": self.fire_localization_result,
            "latest_exit_blockage_results": self.latest_exit_blockage_results,
            "last_perception_replan_reason": self.last_perception_replan_reason,
            "active_route_decision": self.active_route_decision,
            "active_route_valid": self.active_route_valid,
            "active_route_invalid_reason": self.active_route_invalid_reason,
            "active_route_blocked_grid": self.active_route_blocked_grid,
            "active_route_created_at": self.active_route_created_at,
            "active_route_last_validated_at": self.active_route_last_validated_at,
            "active_route_costmap_revision": self.active_route_costmap_revision,
            "costmap_revision": self.costmap_revision,
            "environment_revision": self.environment_revision,
            "route_replan_count": self.route_replan_count,
            "final_route_failure_reason": self.final_route_failure_reason,
            "active_path_simplification": self.active_path_simplification,
            "path_simplification_history": self.path_simplification_history,
            "current_target_exit_id": self.current_target_exit_id,
            "current_target_selection_reason": self.current_target_selection_reason,
            "current_target_is_usable": self.current_target_is_usable,
            "exit_path_lengths_m": self.exit_path_lengths_m,
            "remaining_route_cost": self.remaining_route_cost,
            "route_cost_baseline": self.route_cost_baseline,
            "route_cost_history": self.route_cost_history,
            "consecutive_route_cost_increases": self.consecutive_route_cost_increases,
            "route_costmap_revision": self.route_costmap_revision,
            "exit_switch_occurred": self.exit_switch_occurred,
            "previous_target_exit_id": self.previous_target_exit_id,
            "last_exit_switch_reason": self.last_exit_switch_reason,
            "last_exit_switch_at": self.last_exit_switch_at,
            "exit_switch_cooldown_until": self.exit_switch_cooldown_until,
            "last_exit_switch_validation": self.last_exit_switch_validation,
            "initial_robot_pose_world": self.initial_robot_pose_world,
            "exploration_entry_pose_world": self.exploration_entry_pose_world,
            "exploration_resume_pose_world": self.exploration_resume_pose_world,
            "robot_grid_position": self.robot_grid_position,
            "current_exploration_target_exit_id": self.current_exploration_target_exit_id,
            "exploration_plan_start_world": self.exploration_plan_start_world,
            "exploration_exit_path_lengths_m": self.exploration_exit_path_lengths_m,
            "exploration_costmap_revision": self.exploration_costmap_revision,
            "exploration_environment_revision": self.exploration_environment_revision,
            "exploration_phase": self.exploration_phase,
            "exploration_return_target_exit_id": self.exploration_return_target_exit_id,
            "exploration_return_to_entrance_enabled": self.exploration_return_to_entrance_enabled,
            "active_exploration_plan": self.active_exploration_plan,
            "exit_visit_status": self.exit_visit_status,
            "exit_check_history": self.exit_check_history,
            "exit_reachability_by_revision": self.exit_reachability_by_revision,
            "exploration_interruptions": self.exploration_interruptions,
            "exploration_stalled": self.exploration_stalled,
            "exploration_stall_reason": self.exploration_stall_reason,
            "active_following_victim_id": self.active_following_victim_id,
            "victim_following": (
                None if self.victim_following_controller is None
                else self.victim_following_controller.to_dict()
            ),
            "victim_following_events": self.victim_following_events,
            "static_obstacle_map": self.static_obstacle_map,
            "known_occupancy_map": self.known_occupancy_map,
            "dynamic_obstacles": self.dynamic_obstacles,
            "exits": self.exits,
            "victims": self.victims,
            "ground_truth_fire_map": {
                "temperature_c": self.ground_truth_fire_map.temperature_c,
                "co_ppm": self.ground_truth_fire_map.co_ppm,
                "observed_mask": self.ground_truth_fire_map.observed_mask,
                "last_observed_time": self.ground_truth_fire_map.last_observed_time,
            },
            "estimated_fire_map": {
                "temperature_c": self.estimated_fire_map.temperature_c,
                "co_ppm": self.estimated_fire_map.co_ppm,
                "observed_mask": self.estimated_fire_map.observed_mask,
                "last_observed_time": self.estimated_fire_map.last_observed_time,
                "thermal_fire_evidence": self.estimated_fire_map.thermal_fire_evidence,
                "co_gradient_evidence": self.estimated_fire_map.co_gradient_evidence,
                "combined_fire_evidence": self.estimated_fire_map.combined_fire_evidence,
                "fire_probability": self.estimated_fire_map.fire_probability,
                "fire_observation_count": self.estimated_fire_map.fire_observation_count,
                "fire_last_observed_time": self.estimated_fire_map.fire_last_observed_time,
                "fire_localization_result": self.estimated_fire_map.fire_localization_result,
            },
        }
        result = _json_value(payload)
        json.dumps(result)
        return result
