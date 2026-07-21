import heapq
import math


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
