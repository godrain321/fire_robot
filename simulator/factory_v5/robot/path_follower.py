"""Simple rotate-then-drive follower for replannable A* paths."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class RobotState:
    x: float
    y: float
    theta: float


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def shortest_rotation_delta(current_theta: float, target_theta: float) -> float:
    """Return the smaller signed turn from current yaw to target yaw.

    Positive is counter-clockwise and negative is clockwise. An exact 180
    degree tie deterministically uses counter-clockwise rotation.
    """
    full_turn = 2.0 * math.pi
    counter_clockwise = (float(target_theta) - float(current_theta)) % full_turn
    clockwise = (float(current_theta) - float(target_theta)) % full_turn
    if clockwise < counter_clockwise:
        return -clockwise
    return counter_clockwise


class ReplannablePathFollower:
    """Follow world waypoints and atomically replace them after replanning."""

    def __init__(self, grid_map, config) -> None:
        self.grid_map = grid_map
        self.config = config
        self.grid_path: list[tuple[int, int]] = []
        self.world_path: list[tuple[float, float]] = []
        self.waypoint_index = 0
        self.escape_nodes: set[tuple[int, int]] = set()

    def set_path(
        self, grid_path, escape_path=(), goal_world=None, world_path=None,
    ) -> None:
        self.grid_path = list(grid_path)
        if world_path is None:
            self.world_path = [
                self.grid_map.grid_to_world(gx, gy) for gx, gy in self.grid_path
            ]
        else:
            self.world_path = [tuple(point) for point in world_path]
            if len(self.world_path) != len(self.grid_path):
                raise ValueError("world_path and grid_path must have equal length")
        if self.world_path and goal_world is not None:
            self.world_path[-1] = tuple(goal_world)
        # A one-node replan means the robot is in the goal grid cell but may
        # still need to drive to the exact world goal position.
        self.waypoint_index = 1 if len(self.world_path) > 1 else 0
        self.escape_nodes = set(escape_path)

    def clear(self) -> None:
        self.set_path([])

    def remaining_grid_path(self) -> list[tuple[int, int]]:
        if not self.grid_path:
            return []
        start = max(0, self.waypoint_index - 1)
        return self.grid_path[start:]

    def update(self, state: RobotState, dt: float, final_cost_map: np.ndarray) -> tuple[float, str]:
        """Advance one time step without entering newly blocked cells."""
        if self.waypoint_index >= len(self.world_path):
            return 0.0, "no active waypoint"

        target_x, target_y = self.world_path[self.waypoint_index]
        target_grid = self.grid_path[self.waypoint_index]
        if (
            not np.isfinite(final_cost_map[target_grid[1], target_grid[0]])
            and target_grid not in self.escape_nodes
        ):
            return 0.0, "next waypoint became blocked"

        dx = target_x - state.x
        dy = target_y - state.y
        distance = math.hypot(dx, dy)
        if distance <= self.config.waypoint_tolerance:
            state.x, state.y = target_x, target_y
            self.waypoint_index += 1
            # The tolerance decides when to snap to the exact waypoint, but
            # the remaining physical displacement still belongs to the actual
            # trajectory and travelled-distance metrics.
            return distance, "waypoint reached"

        target_theta = math.atan2(dy, dx)
        angle_error = shortest_rotation_delta(state.theta, target_theta)
        max_turn = self.config.robot_angular_speed * dt
        if abs(angle_error) > math.radians(1.0):
            state.theta = _wrap_angle(
                state.theta + max(-max_turn, min(max_turn, angle_error))
            )
            return 0.0, "turning"

        state.theta = target_theta
        step = min(self.config.robot_speed * dt, distance)
        next_x = state.x + math.cos(state.theta) * step
        next_y = state.y + math.sin(state.theta) * step
        next_grid = self.grid_map.world_to_grid(next_x, next_y)
        if not self.grid_map.in_bounds(next_grid):
            return 0.0, "proposed motion leaves map"
        if self.grid_map.is_blocked(next_grid):
            return 0.0, "proposed motion enters static obstacle"
        if (
            not np.isfinite(final_cost_map[next_grid[1], next_grid[0]])
            and next_grid not in self.escape_nodes
        ):
            return 0.0, "proposed motion enters blocked belief cell"

        state.x, state.y = next_x, next_y
        if step >= distance - 1e-9:
            state.x, state.y = target_x, target_y
            self.waypoint_index += 1
        return step, "moving"

    def goal_reached(self, state: RobotState, goal_world) -> bool:
        return math.hypot(state.x - goal_world[0], state.y - goal_world[1]) <= self.config.waypoint_tolerance
