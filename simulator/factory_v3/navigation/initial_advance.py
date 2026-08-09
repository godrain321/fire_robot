"""One-time physical advance before the initial exploration plan."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class InitialAdvanceConfig:
    distance_m: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.distance_m, bool) or not math.isfinite(
            float(self.distance_m)
        ):
            raise TypeError("initial advance distance_m must be a finite number")
        if float(self.distance_m) < 0.0:
            raise ValueError("initial advance distance_m must be non-negative")

    @classmethod
    def from_robot_motion(cls, values) -> "InitialAdvanceConfig":
        values = dict(values or {})
        return cls(float(values.get("initial_forward_distance_m", 0.0)))

    def target_world(
        self, start_world: tuple[float, float], yaw_rad: float,
    ) -> tuple[float, float]:
        if not all(math.isfinite(float(item)) for item in (*start_world, yaw_rad)):
            raise ValueError("initial advance pose must be finite")
        return (
            float(start_world[0]) + math.cos(float(yaw_rad)) * self.distance_m,
            float(start_world[1]) + math.sin(float(yaw_rad)) * self.distance_m,
        )
