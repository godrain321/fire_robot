"""Victim motion model driven only by the robot's actual pose history."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np


class FollowState(Enum):
    INACTIVE = "inactive"
    FOLLOWING = "following"
    FOLLOW_WAIT = "follow_wait"
    FOLLOW_FAILED = "follow_failed"
    EVACUATED = "evacuated"


@dataclass(frozen=True)
class VictimFollowingConfig:
    enabled: bool = True
    target_follow_distance_m: float = 1.5
    max_follow_distance_m: float = 2.5
    resume_follow_distance_m: float = 2.0
    victim_speed_mps: float = 0.8
    robot_pose_sample_distance_m: float = 0.2
    follow_wait_timeout_s: float = 10.0
    max_reprompt_count: int = 3
    require_victim_at_exit_for_success: bool = True

    @classmethod
    def from_mapping(cls, values) -> "VictimFollowingConfig":
        values = dict(values or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown victim_following settings: {sorted(unknown)}")
        for name in ("enabled", "require_victim_at_exit_for_success"):
            if name in values and not isinstance(values[name], bool):
                raise TypeError(f"victim_following.{name} must be Boolean")
        return cls(**values)

    def __post_init__(self) -> None:
        positive = {
            "target_follow_distance_m": self.target_follow_distance_m,
            "max_follow_distance_m": self.max_follow_distance_m,
            "resume_follow_distance_m": self.resume_follow_distance_m,
            "victim_speed_mps": self.victim_speed_mps,
            "robot_pose_sample_distance_m": self.robot_pose_sample_distance_m,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_follow_distance_m < self.target_follow_distance_m:
            raise ValueError("max_follow_distance_m must be >= target_follow_distance_m")
        if self.resume_follow_distance_m >= self.max_follow_distance_m:
            raise ValueError("resume_follow_distance_m must be < max_follow_distance_m")
        if self.follow_wait_timeout_s < 0.0:
            raise ValueError("follow_wait_timeout_s must be non-negative")
        if (
            isinstance(self.max_reprompt_count, bool)
            or not isinstance(self.max_reprompt_count, int)
            or self.max_reprompt_count < 0
        ):
            raise ValueError("max_reprompt_count must be a non-negative integer")
        if not self.require_victim_at_exit_for_success:
            raise ValueError(
                "require_victim_at_exit_for_success must remain true; "
                "robot-only completion is not supported"
            )


@dataclass(frozen=True)
class RobotPoseSample:
    x: float
    y: float
    yaw: float
    recorded_at: float
    costmap_revision: int


@dataclass(frozen=True)
class FollowUpdate:
    position_world: tuple[float, float]
    target_world: tuple[float, float] | None
    robot_victim_distance_m: float
    state: FollowState
    robot_should_wait: bool
    moved_distance_m: float
    event: str | None = None


class VictimFollowingController:
    """Move one victim along samples of the path the robot actually traversed."""

    def __init__(self, map_metadata, config: VictimFollowingConfig) -> None:
        self.metadata = map_metadata
        self.config = config
        self.victim_id: str | None = None
        self.position_world: tuple[float, float] | None = None
        self.target_world: tuple[float, float] | None = None
        self.pose_history: list[RobotPoseSample] = []
        self.state = FollowState.INACTIVE
        self.wait_started_at: float | None = None
        self.reprompt_count = 0
        self.failure_reason: str | None = None
        self.catch_up_to_exit = False
        self.total_victim_distance_m = 0.0

    @property
    def active(self) -> bool:
        return self.state in (FollowState.FOLLOWING, FollowState.FOLLOW_WAIT)

    def start(
        self, victim_id: str, victim_position_world, robot_pose_world,
        *, sim_time: float, costmap_revision: int,
    ) -> None:
        if not self.config.enabled:
            raise RuntimeError("victim following is disabled")
        self._validate_world(victim_position_world, "victim")
        self._validate_world(robot_pose_world[:2], "robot")
        self.victim_id = str(victim_id)
        self.position_world = tuple(map(float, victim_position_world))
        self.target_world = self.position_world
        self.pose_history = [RobotPoseSample(
            float(robot_pose_world[0]), float(robot_pose_world[1]),
            float(robot_pose_world[2]), float(sim_time), int(costmap_revision),
        )]
        self.state = FollowState.FOLLOWING
        self.wait_started_at = None
        self.reprompt_count = 0
        self.failure_reason = None
        self.catch_up_to_exit = False
        self.total_victim_distance_m = 0.0

    def record_robot_pose(
        self, robot_pose_world, *, sim_time: float, costmap_revision: int,
    ) -> bool:
        if not self.active:
            return False
        latest = self.pose_history[-1]
        if math.hypot(robot_pose_world[0] - latest.x, robot_pose_world[1] - latest.y) + 1e-12 < self.config.robot_pose_sample_distance_m:
            return False
        self.pose_history.append(RobotPoseSample(
            float(robot_pose_world[0]), float(robot_pose_world[1]),
            float(robot_pose_world[2]), float(sim_time), int(costmap_revision),
        ))
        return True

    def request_exit_catch_up(self) -> None:
        if self.active:
            self.catch_up_to_exit = True

    def update(
        self, robot_position_world, *, dt: float, sim_time: float,
        static_obstacle_map, dynamic_obstacle_map,
    ) -> FollowUpdate:
        if not self.active or self.position_world is None:
            raise RuntimeError("victim following is not active")
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        robot = tuple(map(float, robot_position_world))
        distance_before = math.dist(robot, self.position_world)
        event = None
        if self.state is FollowState.FOLLOWING and distance_before >= self.config.max_follow_distance_m:
            self.state = FollowState.FOLLOW_WAIT
            self.wait_started_at = float(sim_time)
            event = "victim_lagging"

        self.target_world = self._select_target(robot)
        moved = self._move_toward_target(
            self.config.victim_speed_mps * dt,
            np.asarray(static_obstacle_map, dtype=bool),
            np.asarray(dynamic_obstacle_map, dtype=bool),
        )
        self.total_victim_distance_m += moved
        distance_after = math.dist(robot, self.position_world)

        if self.state is FollowState.FOLLOW_WAIT:
            if distance_after <= self.config.resume_follow_distance_m:
                self.state = FollowState.FOLLOWING
                self.wait_started_at = None
                event = "victim_caught_up"
            elif (
                self.wait_started_at is not None
                and sim_time - self.wait_started_at >= self.config.follow_wait_timeout_s
            ):
                if self.reprompt_count < self.config.max_reprompt_count:
                    self.reprompt_count += 1
                    self.wait_started_at = float(sim_time)
                    event = "follow_reprompt"
                else:
                    self.state = FollowState.FOLLOW_FAILED
                    self.failure_reason = "victim did not catch up before follow timeout"
                    event = "follow_failed"

        return FollowUpdate(
            self.position_world, self.target_world, distance_after, self.state,
            self.state in (FollowState.FOLLOW_WAIT, FollowState.FOLLOW_FAILED),
            moved, event,
        )

    def mark_evacuated(self) -> None:
        if not self.active:
            raise RuntimeError("cannot evacuate an inactive follower")
        self.state = FollowState.EVACUATED
        self.wait_started_at = None

    def clear(self) -> None:
        self.victim_id = None
        self.position_world = None
        self.target_world = None
        self.pose_history.clear()
        self.state = FollowState.INACTIVE
        self.wait_started_at = None
        self.reprompt_count = 0
        self.failure_reason = None
        self.catch_up_to_exit = False
        self.total_victim_distance_m = 0.0

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "victim_id": self.victim_id,
            "position_world": self.position_world,
            "target_world": self.target_world,
            "state": self.state.value,
            "pose_history": [asdict(item) for item in self.pose_history],
            "wait_started_at": self.wait_started_at,
            "reprompt_count": self.reprompt_count,
            "failure_reason": self.failure_reason,
            "total_victim_distance_m": self.total_victim_distance_m,
        }

    def _select_target(self, robot) -> tuple[float, float]:
        if self.catch_up_to_exit:
            return robot
        points = [(item.x, item.y) for item in self.pose_history]
        if not points:
            return robot
        remaining = self.config.target_follow_distance_m
        cursor = robot
        for point in reversed(points):
            segment = math.dist(cursor, point)
            if segment >= remaining and segment > 0.0:
                ratio = remaining / segment
                return (
                    cursor[0] + (point[0] - cursor[0]) * ratio,
                    cursor[1] + (point[1] - cursor[1]) * ratio,
                )
            remaining -= segment
            cursor = point
        return points[0]

    def _move_toward_target(self, maximum_step, static, dynamic) -> float:
        target = self.target_world
        start = self.position_world
        distance = math.dist(start, target)
        if distance <= 1e-12 or maximum_step <= 0.0:
            return 0.0
        step = min(maximum_step, distance)
        ratio = step / distance
        candidate = (
            start[0] + (target[0] - start[0]) * ratio,
            start[1] + (target[1] - start[1]) * ratio,
        )
        if not self._segment_is_free(start, candidate, static, dynamic):
            return 0.0
        self.position_world = candidate
        return step

    def _segment_is_free(self, start, end, static, dynamic) -> bool:
        if static.shape != (self.metadata.height, self.metadata.width) or dynamic.shape != static.shape:
            raise ValueError("obstacle map shape does not match victim-following map")
        distance = math.dist(start, end)
        samples = max(1, int(math.ceil(distance / (self.metadata.resolution_m / 2.0))))
        previous = None
        for index in range(samples + 1):
            ratio = index / samples
            point = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
            if not self.metadata.is_world_position_in_bounds(*point):
                return False
            col, row = self.metadata.world_to_grid(*point)
            if static[row, col] or dynamic[row, col]:
                return False
            if previous is not None:
                dc, dr = col - previous[0], row - previous[1]
                if abs(dc) == 1 and abs(dr) == 1:
                    side_a = (previous[0] + dc, previous[1])
                    side_b = (previous[0], previous[1] + dr)
                    if (
                        static[side_a[1], side_a[0]]
                        or dynamic[side_a[1], side_a[0]]
                        or static[side_b[1], side_b[0]]
                        or dynamic[side_b[1], side_b[0]]
                    ):
                        return False
            previous = (col, row)
        return True

    def _validate_world(self, position, label) -> None:
        if not self.metadata.is_world_position_in_bounds(float(position[0]), float(position[1])):
            raise ValueError(f"{label} position is outside the map: {position}")


def evacuation_success_ready(
    *, robot_position_world, victim_position_world, exit_position_world,
    exit_radius_m: float, exit_usable: bool, route_valid: bool,
    victim_moved_by_following: bool,
) -> bool:
    """Return true only when both agents actually reached a validated exit."""
    if exit_radius_m <= 0.0:
        raise ValueError("exit_radius_m must be positive")
    return bool(
        exit_usable and route_valid and victim_moved_by_following
        and math.dist(robot_position_world, exit_position_world) <= exit_radius_m
        and math.dist(victim_position_world, exit_position_world) <= exit_radius_m
    )
