"""Pre-detection victim walking along explicit safe world waypoints."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ScriptedVictimMotionConfig:
    enabled: bool
    speed_mps: float
    stop_when_detected: bool
    waypoints_world: tuple[tuple[float, float], ...]

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        allowed = {"enabled", "speed_mps", "stop_when_detected", "waypoints_world"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown scripted victim motion settings: {sorted(unknown)}")
        enabled = values.get("enabled", False)
        stop = values.get("stop_when_detected", True)
        if type(enabled) is not bool or type(stop) is not bool:
            raise TypeError("victim scripted-motion flags must be Boolean")
        speed = float(values.get("speed_mps", 0.8))
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("victim scripted-motion speed_mps must be positive")
        points = tuple(
            (float(item[0]), float(item[1]))
            for item in values.get("waypoints_world", ())
        )
        if enabled and len(points) < 2:
            raise ValueError("enabled victim scripted motion requires at least two waypoints")
        if any(not all(math.isfinite(value) for value in point) for point in points):
            raise ValueError("victim scripted-motion waypoints must be finite")
        return cls(enabled, speed, stop, points)


class ScriptedVictimMotionController:
    """Move without teleporting and reject any segment crossing occupancy."""

    def __init__(self, metadata, victim_id, initial_position_world, config):
        self.metadata = metadata
        self.victim_id = str(victim_id)
        self.config = config
        self.position_world = tuple(map(float, initial_position_world))
        self.next_waypoint_index = 1
        self.completed = not config.enabled
        if config.enabled:
            if not math.isclose(self.position_world[0], config.waypoints_world[0][0], abs_tol=1e-9) or not math.isclose(
                self.position_world[1], config.waypoints_world[0][1], abs_tol=1e-9
            ):
                raise ValueError("victim position must equal the first scripted waypoint")
            for point in config.waypoints_world:
                if not metadata.is_world_position_in_bounds(*point):
                    raise ValueError(f"victim scripted waypoint is outside map: {point}")

    def update(self, *, dt, static_obstacle_map, dynamic_obstacle_map):
        if self.completed or dt <= 0.0:
            return 0.0
        static = np.asarray(static_obstacle_map, dtype=bool)
        dynamic = np.asarray(dynamic_obstacle_map, dtype=bool)
        expected = (self.metadata.height, self.metadata.width)
        if static.shape != expected or dynamic.shape != expected:
            raise ValueError("victim scripted-motion obstacle map shape mismatch")
        remaining_step = self.config.speed_mps * float(dt)
        moved = 0.0
        while remaining_step > 1e-12 and not self.completed:
            target = self.config.waypoints_world[self.next_waypoint_index]
            distance = math.dist(self.position_world, target)
            if distance <= 1e-12:
                self._advance_waypoint()
                continue
            step = min(remaining_step, distance)
            ratio = step / distance
            candidate = (
                self.position_world[0] + (target[0] - self.position_world[0]) * ratio,
                self.position_world[1] + (target[1] - self.position_world[1]) * ratio,
            )
            if not self._segment_is_free(self.position_world, candidate, static, dynamic):
                break
            self.position_world = candidate
            moved += step
            remaining_step -= step
            if step >= distance - 1e-12:
                self.position_world = target
                self._advance_waypoint()
        return moved

    def _advance_waypoint(self):
        self.next_waypoint_index += 1
        if self.next_waypoint_index >= len(self.config.waypoints_world):
            self.completed = True

    def _segment_is_free(self, start, end, static, dynamic):
        distance = math.dist(start, end)
        samples = max(1, int(math.ceil(distance / (self.metadata.resolution_m / 2.0))))
        for index in range(samples + 1):
            ratio = index / samples
            x = start[0] + (end[0] - start[0]) * ratio
            y = start[1] + (end[1] - start[1]) * ratio
            if not self.metadata.is_world_position_in_bounds(x, y):
                return False
            col, row = self.metadata.world_to_grid(x, y)
            if static[row, col] or dynamic[row, col]:
                return False
        return True
