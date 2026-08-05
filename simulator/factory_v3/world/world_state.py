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
    Exit, ExitStatus, Victim, VictimStatus,
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

    def record_robot_position(
        self, position_world, *, sim_time: float | None = None,
        is_returning: bool = False, force_reason=None,
    ):
        self.validate_position(position_world, label="robot")
        self.robot_position_world = (
            float(position_world[0]), float(position_world[1])
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

    def clear_active_evacuation_plan(self) -> None:
        self.active_evacuation_plan = None

    def validate_position(self, position_world, *, require_free: bool = True, label: str = "position") -> tuple[int, int]:
        x, y = float(position_world[0]), float(position_world[1])
        col, row = self.map_metadata.world_to_grid(x, y)
        if require_free and self.static_obstacle_map[row, col]:
            raise ValueError(f"{label} {(x, y)} is inside a static obstacle at {(col, row)}")
        return col, row

    def add_exit(self, exit_item: Exit) -> None:
        if exit_item.exit_id in self.exits:
            raise ValueError(f"duplicate exit_id: {exit_item.exit_id}")
        self.validate_position(exit_item.position_world, require_free=False, label=f"exit {exit_item.exit_id}")
        if exit_item.approach_position_world is not None:
            self.validate_position(exit_item.approach_position_world, label=f"exit approach {exit_item.exit_id}")
        self.exits[exit_item.exit_id] = exit_item

    def get_exit(self, exit_id: str) -> Exit:
        try:
            return self.exits[exit_id]
        except KeyError as exc:
            raise KeyError(f"unknown exit_id: {exit_id}") from exc

    def update_exit_status(self, exit_id: str, status: ExitStatus, **context) -> None:
        self.get_exit(exit_id).update_status(status, sim_time=context.pop("sim_time", self.simulation_time), **context)

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
        self.validate_position(obstacle.position_world, label=f"dynamic obstacle {obstacle.obstacle_id}")
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
        )
        obstacle.update(sim_time=changes.pop("sim_time", self.simulation_time), **changes)
        try:
            self._validate_dynamic_obstacle(obstacle)
        except Exception:
            (
                obstacle.position_world, obstacle.status, obstacle.confidence,
                obstacle.first_seen_at, obstacle.last_seen_at,
            ) = snapshot
            raise

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
            "static_obstacle_map": self.static_obstacle_map,
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
            },
        }
        result = _json_value(payload)
        json.dumps(result)
        return result
