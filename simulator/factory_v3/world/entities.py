"""World-coordinate entities used by the factory_v3 simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any


WorldPosition = tuple[float, float]


def _position(value) -> WorldPosition:
    if len(value) != 2:
        raise ValueError("world position must contain exactly (x, y)")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError("world position must be finite")
    return result


class ExitStatus(Enum):
    UNKNOWN = "unknown"
    USABLE = "usable"
    BLOCKED = "blocked"
    DANGEROUS = "dangerous"


class ExitVisitStatus(Enum):
    """Direct exploration visit state, independent of safety assessment."""

    UNCHECKED = "unchecked"
    CHECKED = "checked"


@dataclass(frozen=True)
class ExitCheckRecord:
    exit_id: str
    visit_status: ExitVisitStatus
    safety_status: ExitStatus
    checked_at: float
    costmap_revision: int
    reason: str


@dataclass(frozen=True)
class ExplorationInterruption:
    interrupted_at: float
    reason: str
    victim_id: str
    robot_pose_world: tuple[float, float, float]
    target_exit_id: str | None
    active_path_grid: tuple[tuple[int, int], ...]
    costmap_revision: int
    resume_required: bool


@dataclass(frozen=True)
class ExitStatusChange:
    previous: ExitStatus
    current: ExitStatus
    sim_time: float | None
    reason: str | None


@dataclass
class Exit:
    exit_id: str
    position_world: WorldPosition
    approach_position_world: WorldPosition | None
    status: ExitStatus = ExitStatus.UNKNOWN
    last_checked_at: float | None = None
    blocked_reason: str | None = None
    danger_reason: str | None = None
    temperature_c: float | None = None
    co_ppm: float | None = None
    path_cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status_history: list[ExitStatusChange] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.exit_id:
            raise ValueError("exit_id must not be empty")
        self.position_world = _position(self.position_world)
        if self.approach_position_world is not None:
            self.approach_position_world = _position(self.approach_position_world)
        if not isinstance(self.status, ExitStatus):
            raise TypeError("status must be ExitStatus")

    def update_status(
        self, status: ExitStatus, *, sim_time: float | None = None,
        reason: str | None = None, **measurements: Any,
    ) -> None:
        if not isinstance(status, ExitStatus):
            raise TypeError("status must be ExitStatus")
        if status is ExitStatus.BLOCKED and not reason:
            raise ValueError("BLOCKED exit requires a reason")
        if status is ExitStatus.DANGEROUS and not reason:
            raise ValueError("DANGEROUS exit requires a reason")
        previous = self.status
        self.status = status
        self.last_checked_at = None if sim_time is None else float(sim_time)
        self.blocked_reason = reason if status is ExitStatus.BLOCKED else None
        self.danger_reason = reason if status is ExitStatus.DANGEROUS else None
        for name in ("temperature_c", "co_ppm", "path_cost"):
            if name in measurements:
                setattr(self, name, float(measurements[name]))
        self.status_history.append(
            ExitStatusChange(previous, status, self.last_checked_at, reason)
        )


class VictimStatus(Enum):
    UNDETECTED = "undetected"
    DETECTED = "detected"
    APPROACHING = "approaching"
    REACHED = "reached"
    WAITING = "waiting"
    MOVABLE = "movable"
    READY_TO_FOLLOW = "ready_to_follow"
    FOLLOWING = "following"
    FOLLOW_WAIT = "follow_wait"
    FOLLOW_FAILED = "follow_failed"
    IMMOBILE = "immobile"
    EVACUATING = "evacuating"
    EVACUATED = "evacuated"
    RESCUED = "rescued"
    REPORTED = "reported"


_VICTIM_TRANSITIONS = {
    VictimStatus.UNDETECTED: {VictimStatus.DETECTED},
    VictimStatus.DETECTED: {VictimStatus.APPROACHING, VictimStatus.REACHED, VictimStatus.IMMOBILE},
    VictimStatus.APPROACHING: {VictimStatus.REACHED, VictimStatus.IMMOBILE},
    VictimStatus.REACHED: {VictimStatus.WAITING, VictimStatus.MOVABLE, VictimStatus.IMMOBILE},
    VictimStatus.WAITING: {VictimStatus.MOVABLE, VictimStatus.IMMOBILE},
    VictimStatus.MOVABLE: {
        VictimStatus.READY_TO_FOLLOW, VictimStatus.EVACUATING,
        VictimStatus.IMMOBILE,
    },
    VictimStatus.READY_TO_FOLLOW: {
        VictimStatus.FOLLOWING, VictimStatus.IMMOBILE,
    },
    VictimStatus.FOLLOWING: {
        VictimStatus.FOLLOW_WAIT, VictimStatus.FOLLOW_FAILED,
        VictimStatus.EVACUATED, VictimStatus.RESCUED, VictimStatus.IMMOBILE,
    },
    VictimStatus.FOLLOW_WAIT: {
        VictimStatus.FOLLOWING, VictimStatus.FOLLOW_FAILED,
        VictimStatus.IMMOBILE,
    },
    VictimStatus.FOLLOW_FAILED: set(),
    VictimStatus.IMMOBILE: {VictimStatus.REPORTED},
    VictimStatus.EVACUATING: {
        VictimStatus.FOLLOWING, VictimStatus.EVACUATED,
        VictimStatus.RESCUED, VictimStatus.IMMOBILE,
    },
    VictimStatus.EVACUATED: set(),
    VictimStatus.RESCUED: set(),
    VictimStatus.REPORTED: set(),
}


@dataclass
class Victim:
    victim_id: str
    position_world: WorldPosition
    status: VictimStatus = VictimStatus.UNDETECTED
    detected_at: float | None = None
    last_seen_at: float | None = None
    distance_from_robot_m: float | None = None
    assigned_exit_id: str | None = None
    following_robot: bool = False
    current_grid_position: tuple[int, int] | None = None
    follow_target_world: WorldPosition | None = None
    follow_distance_m: float | None = None
    follow_wait_active: bool = False
    follow_wait_started_at: float | None = None
    follow_reprompt_count: int = 0
    follow_failed: bool = False
    follow_failure_reason: str | None = None
    evacuated_at: float | None = None
    evacuation_success_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.victim_id:
            raise ValueError("victim_id must not be empty")
        self.position_world = _position(self.position_world)
        if not isinstance(self.status, VictimStatus):
            raise TypeError("status must be VictimStatus")

    @property
    def detected(self) -> bool:
        return self.status is not VictimStatus.UNDETECTED

    @property
    def movable(self) -> bool | None:
        if self.status in (
            VictimStatus.MOVABLE, VictimStatus.READY_TO_FOLLOW,
            VictimStatus.FOLLOWING, VictimStatus.FOLLOW_WAIT,
            VictimStatus.EVACUATING, VictimStatus.EVACUATED,
            VictimStatus.RESCUED,
        ):
            return True
        if self.status in (VictimStatus.IMMOBILE, VictimStatus.REPORTED):
            return False
        return None

    @property
    def rescued(self) -> bool:
        return self.status in (VictimStatus.EVACUATED, VictimStatus.RESCUED)

    def update_status(self, status: VictimStatus, *, sim_time: float | None = None) -> None:
        if not isinstance(status, VictimStatus):
            raise TypeError("status must be VictimStatus")
        if status is not self.status and status not in _VICTIM_TRANSITIONS[self.status]:
            raise ValueError(f"invalid victim transition {self.status.name} -> {status.name}")
        self.status = status
        if status is VictimStatus.DETECTED and self.detected_at is None:
            self.detected_at = None if sim_time is None else float(sim_time)
        if status is not VictimStatus.UNDETECTED:
            self.last_seen_at = None if sim_time is None else float(sim_time)
        self.following_robot = status in (
            VictimStatus.FOLLOWING, VictimStatus.FOLLOW_WAIT,
            VictimStatus.EVACUATING,
        )


class DynamicObstacleStatus(Enum):
    UNKNOWN = "unknown"
    ACTIVE = "active"
    MOVING = "moving"  # Moving carts/people need a distinct future update policy.
    CLEARED = "cleared"


class DynamicObstacleShape(Enum):
    POINT = "point"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"


@dataclass
class DynamicObstacle:
    obstacle_id: str
    position_world: WorldPosition
    shape: DynamicObstacleShape = DynamicObstacleShape.POINT
    size_m: tuple[float, float] = (0.0, 0.0)
    status: DynamicObstacleStatus = DynamicObstacleStatus.UNKNOWN
    first_seen_at: float | None = None
    last_seen_at: float | None = None
    source: str = "unknown"
    confidence: float = 0.0
    observation_count: int = 0
    confirmed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.obstacle_id:
            raise ValueError("obstacle_id must not be empty")
        self.position_world = _position(self.position_world)
        if not isinstance(self.shape, DynamicObstacleShape):
            raise TypeError("shape must be DynamicObstacleShape")
        if not isinstance(self.status, DynamicObstacleStatus):
            raise TypeError("status must be DynamicObstacleStatus")
        if len(self.size_m) != 2 or any(float(v) < 0 for v in self.size_m):
            raise ValueError("size_m must be two non-negative values")
        self.size_m = tuple(float(v) for v in self.size_m)
        if self.shape is DynamicObstacleShape.CIRCLE and self.size_m[0] <= 0:
            raise ValueError("circle size_m[0] must be a positive diameter")
        if self.shape is DynamicObstacleShape.RECTANGLE and any(v <= 0 for v in self.size_m):
            raise ValueError("rectangle width and height must be positive")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = float(self.confidence)
        if self.observation_count < 0:
            raise ValueError("observation_count must be non-negative")
        if type(self.confirmed) is not bool:
            raise TypeError("confirmed must be bool")

    def update(
        self, *, position_world=None, status=None, sim_time=None,
        confidence=None, observation_count=None,
    ) -> None:
        if position_world is not None:
            self.position_world = _position(position_world)
        if status is not None:
            if not isinstance(status, DynamicObstacleStatus):
                raise TypeError("status must be DynamicObstacleStatus")
            if self.status is DynamicObstacleStatus.CLEARED and status is not DynamicObstacleStatus.CLEARED:
                raise ValueError("a cleared obstacle cannot be reactivated; add a new observation ID")
            self.status = status
        if confidence is not None:
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            self.confidence = confidence
        if observation_count is not None:
            observation_count = int(observation_count)
            if observation_count < self.observation_count:
                raise ValueError("observation_count must not decrease")
            self.observation_count = observation_count
        if self.first_seen_at is None and sim_time is not None:
            self.first_seen_at = float(sim_time)
        self.last_seen_at = None if sim_time is None else float(sim_time)
