import heapq
import math
from dataclasses import dataclass

import numpy as np


_MOVES = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def _octile_distance(node, goal):
    dx = abs(node[0] - goal[0])
    dy = abs(node[1] - goal[1])
    return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)


def _neighbors(grid_map, node):
    x, y = node
    for dx, dy, cost in _MOVES:
        nxt = (x + dx, y + dy)
        if grid_map.is_blocked(nxt):
            continue

        # A diagonal is legal only when both adjacent cardinal cells are free.
        # This prevents the path from cutting through an obstacle corner.
        if dx and dy:
            if grid_map.is_blocked((x + dx, y)):
                continue
            if grid_map.is_blocked((x, y + dy)):
                continue

        yield nxt, cost


def a_star(grid_map, start, goal):
    """Return a list of grid coordinates from start to goal, or None."""
    if grid_map.is_blocked(start) or grid_map.is_blocked(goal):
        return None
    if start == goal:
        return [start]

    frontier = [(0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for nxt, step_cost in _neighbors(grid_map, current):
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + _octile_distance(nxt, goal)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return None

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


@dataclass(frozen=True)
class AStarResult:
    """Weighted A* result with an explicit failure reason."""

    path: list[tuple[int, int]]
    total_cost: float
    reason: str
    escape_path: tuple[tuple[int, int], ...] = ()
    replan_start: tuple[int, int] | None = None


def weighted_a_star(cost_map, start, goal):
    """Plan over ``cost_map[gy, gx]`` using 8-connected weighted A*.

    Each edge costs its metric grid distance multiplied by the average cost of
    its two endpoint cells. This trapezoidal approximation avoids introducing
    a directional bias from charging only the destination cell.
    """
    costs = np.asarray(cost_map, dtype=float)
    if costs.ndim != 2:
        return AStarResult([], math.inf, "cost_map must be a 2-D array")

    height, width = costs.shape

    def in_bounds(node):
        x, y = node
        return 0 <= x < width and 0 <= y < height

    def blocked(node):
        x, y = node
        return not in_bounds(node) or not math.isfinite(float(costs[y, x]))

    if not in_bounds(start):
        return AStarResult([], math.inf, "start is outside the costmap")
    if not in_bounds(goal):
        return AStarResult([], math.inf, "goal is outside the costmap")
    if blocked(start):
        return AStarResult([], math.inf, "start is blocked")
    if blocked(goal):
        return AStarResult([], math.inf, "goal is blocked")
    if start == goal:
        return AStarResult([start], 0.0, "path found")

    finite_costs = costs[np.isfinite(costs)]
    if finite_costs.size == 0 or np.any(finite_costs <= 0.0):
        return AStarResult([], math.inf, "finite traversal costs must be positive")
    minimum_cost = float(finite_costs.min())

    def heuristic(node):
        return math.hypot(goal[0] - node[0], goal[1] - node[1]) * minimum_cost

    frontier = [(heuristic(start), 0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}

    while frontier:
        _, queued_cost, current = heapq.heappop(frontier)
        if queued_cost > cost_so_far[current]:
            continue
        if current == goal:
            break

        x, y = current
        for dx, dy, distance in _MOVES:
            nxt = (x + dx, y + dy)
            if blocked(nxt):
                continue
            if dx and dy and (blocked((x + dx, y)) or blocked((x, y + dy))):
                continue

            current_cost = float(costs[y, x])
            next_cost = float(costs[nxt[1], nxt[0]])
            step_cost = distance * 0.5 * (current_cost + next_cost)
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(
                    frontier, (new_cost + heuristic(nxt), new_cost, nxt)
                )

    if goal not in came_from:
        return AStarResult([], math.inf, "no traversable path from start to goal")

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return AStarResult(path, cost_so_far[goal], "path found")


def weighted_a_star_with_escape(cost_map, start, goal, static_obstacle_map):
    """Escape an infinite-cost start cell, then replan with the costmap.

    Infinite environmental cost at the robot's current cell cannot prevent the
    robot from leaving that cell. The escape phase therefore minimizes metric
    distance while allowing hazardous/unknown cells, but it never crosses a
    physical static obstacle. As soon as a finite-cost cell with a route to the
    goal is reached, normal weighted A* is run from that cell. Escape distance
    is charged at unit base cost because infinite cells have no finite risk
    value that can be integrated meaningfully.
    """
    costs = np.asarray(cost_map, dtype=float)
    static = np.asarray(static_obstacle_map, dtype=bool)
    if costs.ndim != 2:
        return AStarResult([], math.inf, "cost_map must be a 2-D array")
    if static.shape != costs.shape:
        return AStarResult(
            [], math.inf,
            f"static obstacle shape {static.shape} does not match costmap {costs.shape}",
        )

    height, width = costs.shape

    def in_bounds(node):
        x, y = node
        return 0 <= x < width and 0 <= y < height

    if not in_bounds(start):
        return AStarResult([], math.inf, "start is outside the costmap")
    if not in_bounds(goal):
        return AStarResult([], math.inf, "goal is outside the costmap")
    if static[start[1], start[0]]:
        return AStarResult(
            [], math.inf, "start is inside a non-traversable static obstacle"
        )
    if not math.isfinite(float(costs[goal[1], goal[0]])):
        return AStarResult([], math.inf, "goal is blocked")
    if math.isfinite(float(costs[start[1], start[0]])):
        return weighted_a_star(costs, start, goal)

    frontier = [(0.0, start)]
    escape_distance = {start: 0.0}
    came_from = {start: None}

    while frontier:
        distance_so_far, current = heapq.heappop(frontier)
        if distance_so_far > escape_distance[current]:
            continue

        if math.isfinite(float(costs[current[1], current[0]])):
            replanned = weighted_a_star(costs, current, goal)
            if replanned.path:
                escape_path = []
                node = current
                while node is not None:
                    escape_path.append(node)
                    node = came_from[node]
                escape_path.reverse()
                combined = escape_path + replanned.path[1:]
                return AStarResult(
                    path=combined,
                    total_cost=distance_so_far + replanned.total_cost,
                    reason="path found after escaping infinite-cost start region",
                    escape_path=tuple(escape_path),
                    replan_start=current,
                )

        x, y = current
        for dx, dy, movement_distance in _MOVES:
            nxt = (x + dx, y + dy)
            if not in_bounds(nxt) or static[nxt[1], nxt[0]]:
                continue
            # Physical obstacle corners remain impassable during escape too.
            if dx and dy:
                if static[y, x + dx] or static[y + dy, x]:
                    continue
            new_distance = distance_so_far + movement_distance
            if nxt not in escape_distance or new_distance < escape_distance[nxt]:
                escape_distance[nxt] = new_distance
                came_from[nxt] = current
                heapq.heappush(frontier, (new_distance, nxt))

    return AStarResult(
        [], math.inf, "no finite-cost cell reachable without crossing static obstacles"
    )
