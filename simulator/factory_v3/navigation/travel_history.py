"""Selective history of positions the robot actually traversed.

World positions use metres ``(x,y)``. Grid positions use ``(col,row)``;
NumPy maps are indexed separately as ``map[row,col]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any


class TravelRecordReason(Enum):
    INITIAL_POSITION = "initial_position"
    NEW_GRID_CELL = "new_grid_cell"
    DISTANCE_THRESHOLD = "distance_threshold"
    DIRECTION_CHANGE = "direction_change"
    WAYPOINT_REACHED = "waypoint_reached"
    DOOR_OR_PASSAGE = "door_or_passage"
    MANUAL_CHECKPOINT = "manual_checkpoint"


@dataclass(frozen=True)
class TravelHistoryConfig:
    enabled: bool = True
    record_on_new_grid_cell: bool = True
    min_record_distance_m: float = 0.4
    direction_change_threshold_deg: float = 35.0
    max_points: int = 10000
    record_during_return: bool = False
    simplify_reverse_path: bool = True

    def __post_init__(self) -> None:
        if self.min_record_distance_m < 0:
            raise ValueError("min_record_distance_m must be non-negative")
        if not 0 < self.direction_change_threshold_deg <= 180:
            raise ValueError("direction_change_threshold_deg must be in (0, 180]")
        if self.max_points <= 0:
            raise ValueError("max_points must be positive")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "TravelHistoryConfig":
        values = dict(values or {})
        allowed = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown travel_history settings: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class TravelPoint:
    position_world: tuple[float, float]
    position_grid: tuple[int, int]
    recorded_at: float
    cumulative_distance_m: float
    reason: TravelRecordReason

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["position_world"] = list(self.position_world)
        result["position_grid"] = list(self.position_grid)
        result["reason"] = self.reason.value
        return result


def _angle_difference(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class TravelHistory:
    """Record only meaningful points after confirmed physical translation."""

    def __init__(self, map_metadata, config: TravelHistoryConfig | None = None) -> None:
        self.map_metadata = map_metadata
        self.config = config or TravelHistoryConfig()
        self._points: list[TravelPoint] = []
        self._total_distance_m = 0.0
        self._last_actual_world: tuple[float, float] | None = None
        self._last_motion_heading: float | None = None

    @property
    def total_distance_m(self) -> float:
        return self._total_distance_m

    def reset(self, position_world, *, recorded_at: float = 0.0) -> TravelPoint:
        self.clear()
        return self.record_initial(position_world, recorded_at=recorded_at)

    def clear(self) -> None:
        self._points.clear()
        self._total_distance_m = 0.0
        self._last_actual_world = None
        self._last_motion_heading = None

    def record_initial(self, position_world, *, recorded_at: float = 0.0) -> TravelPoint:
        if self._points:
            raise ValueError("initial position is already recorded")
        world = self._validate_world(position_world)
        point = TravelPoint(
            world, self.map_metadata.world_to_grid(*world), float(recorded_at),
            0.0, TravelRecordReason.INITIAL_POSITION,
        )
        self._points.append(point)
        self._last_actual_world = world
        return point

    def _validate_world(self, position_world) -> tuple[float, float]:
        if len(position_world) != 2:
            raise ValueError("position_world must contain (x,y)")
        world = (float(position_world[0]), float(position_world[1]))
        self.map_metadata.world_to_grid(*world)  # explicit bounds validation
        return world

    def record_position(
        self, position_world, *, recorded_at: float, is_returning: bool = False,
        force_reason: TravelRecordReason | None = None,
    ) -> TravelPoint | None:
        if not self.config.enabled:
            return None
        if not self._points:
            return self.record_initial(position_world, recorded_at=recorded_at)
        world = self._validate_world(position_world)
        if is_returning and not self.config.record_during_return:
            # Do not append to outbound history, but retain the true current
            # pose so a later search phase cannot accumulate the whole return
            # journey as one artificial movement.
            self._last_actual_world = world
            return None
        assert self._last_actual_world is not None
        step = math.hypot(
            world[0] - self._last_actual_world[0],
            world[1] - self._last_actual_world[1],
        )
        if step <= 1e-12:  # stationary or rotation-only frame
            return None
        heading = math.atan2(
            world[1] - self._last_actual_world[1],
            world[0] - self._last_actual_world[0],
        )
        self._total_distance_m += step
        self._last_actual_world = world
        last_record = self._points[-1]
        distance_from_record = math.hypot(
            world[0] - last_record.position_world[0],
            world[1] - last_record.position_world[1],
        )
        grid = self.map_metadata.world_to_grid(*world)
        reason = force_reason
        if reason is None and self.config.record_on_new_grid_cell and grid != last_record.position_grid:
            reason = TravelRecordReason.NEW_GRID_CELL
        elif reason is None and distance_from_record >= self.config.min_record_distance_m:
            reason = TravelRecordReason.DISTANCE_THRESHOLD
        elif (
            reason is None and self._last_motion_heading is not None
            and distance_from_record >= min(0.05, self.config.min_record_distance_m)
            and math.degrees(_angle_difference(heading, self._last_motion_heading))
            >= self.config.direction_change_threshold_deg
        ):
            reason = TravelRecordReason.DIRECTION_CHANGE
        self._last_motion_heading = heading
        if reason is None:
            return None
        if grid == last_record.position_grid and distance_from_record <= 1e-12:
            return None
        point = TravelPoint(
            world, grid, float(recorded_at), self._total_distance_m, reason
        )
        self._points.append(point)
        if len(self._points) > self.config.max_points:
            # Index 0 is the irreplaceable entry point.
            del self._points[1]
        return point

    def get_points(self) -> tuple[TravelPoint, ...]:
        return tuple(self._points)

    def get_points_world(self) -> tuple[tuple[float, float], ...]:
        return tuple(item.position_world for item in self._points)

    def get_points_grid(self) -> tuple[tuple[int, int], ...]:
        return tuple(item.position_grid for item in self._points)

    def build_reverse_path(self) -> tuple[TravelPoint, ...]:
        return tuple(reversed(self._points))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_distance_m": float(self._total_distance_m),
            "point_count": len(self._points),
            "points": [item.to_dict() for item in self._points],
        }
