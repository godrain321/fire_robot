"""Cost-aware global planning over measured SLAM reference waypoints.

The reference graph reduces the global search space, while every edge remains
validated and weighted using the current robot-belief Costmap. Start and goal
connectors are short cell-wise A* paths because neither is required to coincide
with a surveyed reference point.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any

import numpy as np

from planner.a_star import AStarResult, unweighted_a_star


@dataclass(frozen=True)
class ReferenceWaypointGraphConfig:
    enabled: bool = True
    neighbor_radius_m: float = 1.5
    connector_search_radius_m: float = 3.0
    connector_candidate_count: int = 8
    fallback_to_cell_astar: bool = True
    waypoint_cost_radius_m: float = 0.10
    waypoint_risk_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("enabled", "fallback_to_cell_astar"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in (
            "neighbor_radius_m", "connector_search_radius_m",
            "waypoint_cost_radius_m",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool) or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.waypoint_risk_weight, bool)
            or not math.isfinite(float(self.waypoint_risk_weight))
            or float(self.waypoint_risk_weight) < 0.0
        ):
            raise ValueError("waypoint_risk_weight must be finite and non-negative")
        if (
            isinstance(self.connector_candidate_count, bool)
            or not isinstance(self.connector_candidate_count, int)
            or self.connector_candidate_count < 1
        ):
            raise ValueError("connector_candidate_count must be a positive integer")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"unknown reference_waypoint_graph settings: {sorted(unknown)}"
            )
        return cls(**values)


def _append_path(output, path):
    for cell in path:
        cell = (int(cell[0]), int(cell[1]))
        if not output or output[-1] != cell:
            output.append(cell)


class ReferenceWaypointGraphPlanner:
    def __init__(self, metadata, waypoints, config=None):
        self.metadata = metadata
        self.config = config or ReferenceWaypointGraphConfig()
        self.waypoints = tuple(waypoints)
        if self.config.enabled and len(self.waypoints) < 2:
            raise ValueError("reference waypoint graph requires at least two nodes")
        grids = [item.factory_grid for item in self.waypoints]
        if len(set(grids)) != len(grids):
            raise ValueError("reference waypoint graph contains duplicate grid nodes")
        self._candidate_edges = self._build_candidate_edges()

    def _build_candidate_edges(self):
        limit = self.config.neighbor_radius_m + 1e-12
        output = []
        for first in range(len(self.waypoints)):
            a = self.waypoints[first].factory_world
            for second in range(first + 1, len(self.waypoints)):
                b = self.waypoints[second].factory_world
                distance = math.hypot(b[0] - a[0], b[1] - a[1])
                if distance <= limit:
                    output.append((first, second, distance))
        return tuple(output)

    @staticmethod
    def _valid_cell(costs, cell):
        col, row = cell
        return (
            0 <= row < costs.shape[0] and 0 <= col < costs.shape[1]
            and math.isfinite(float(costs[row, col]))
            and float(costs[row, col]) > 0.0
        )

    def _waypoint_cost(self, costs, waypoint):
        radius = max(
            0, int(math.ceil(
                self.config.waypoint_cost_radius_m
                / self.metadata.resolution_m
            )),
        )
        center_col, center_row = waypoint.factory_grid
        values = []
        for row in range(center_row - radius, center_row + radius + 1):
            for col in range(center_col - radius, center_col + radius + 1):
                if (
                    math.hypot(col - center_col, row - center_row)
                    * self.metadata.resolution_m
                    > self.config.waypoint_cost_radius_m + 1e-12
                ):
                    continue
                if not self._valid_cell(costs, (col, row)):
                    return math.inf
                values.append(float(costs[row, col]))
        return max(values) if values else math.inf

    def _edge(
        self, costs, waypoint_costs, first, second, distance, baseline_cost,
    ):
        start = self.waypoints[first].factory_grid
        end = self.waypoints[second].factory_grid
        route = unweighted_a_star(costs, start, end)
        if not route.path:
            return None
        endpoint_risk = max(
            0.0,
            0.5 * (waypoint_costs[first] + waypoint_costs[second])
            - baseline_cost,
        )
        graph_cost = distance * (
            1.0 + self.config.waypoint_risk_weight * endpoint_risk
        )
        return float(graph_cost), tuple(route.path)

    def _connectors(self, costs, endpoint, *, reverse=False):
        candidates = []
        for index, waypoint in enumerate(self.waypoints):
            distance = math.hypot(
                waypoint.factory_grid[0] - endpoint[0],
                waypoint.factory_grid[1] - endpoint[1],
            ) * self.metadata.resolution_m
            if distance <= self.config.connector_search_radius_m + 1e-12:
                candidates.append((distance, waypoint.waypoint_id, index))
        output = {}
        for _, _, index in sorted(candidates)[:self.config.connector_candidate_count]:
            anchor = self.waypoints[index].factory_grid
            result = (
                unweighted_a_star(costs, anchor, endpoint)
                if reverse else unweighted_a_star(costs, endpoint, anchor)
            )
            if result.path:
                output[index] = result
        return output

    def plan(self, cost_map, start, goal):
        costs = np.asarray(cost_map, dtype=float)
        if costs.ndim != 2:
            return AStarResult([], math.inf, "cost_map must be a 2-D array")
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        if start == goal:
            print(f"[Planner] ALREADY_AT_GOAL start={start} goal={goal}")
            return AStarResult(
                [start], 0.0, "ALREADY_AT_GOAL",
                reference_waypoint_ids=tuple(), used_reference_graph=False,
            )
        if not self.config.enabled:
            result = weighted_a_star(costs, start, goal)
            label = "CELL_ASTAR_PATH" if result.path else "TRUE_NO_PATH"
            print(f"[Planner] {label} path_length={len(result.path)}")
            return result
        if not self._valid_cell(costs, start):
            return AStarResult([], math.inf, "start is blocked")
        if not self._valid_cell(costs, goal):
            return AStarResult([], math.inf, "goal is blocked")
        start_links = self._connectors(costs, start)
        if not start_links:
            return self._fallback(costs, start, goal, "start reference connector unavailable")
        goal_links = self._connectors(costs, goal, reverse=True)
        if not goal_links:
            return self._fallback(costs, start, goal, "reference connector unavailable")

        waypoint_costs = tuple(
            self._waypoint_cost(costs, waypoint)
            for waypoint in self.waypoints
        )
        finite_costs = costs[np.isfinite(costs)]
        baseline_cost = float(finite_costs.min())

        adjacency = {index: [] for index in range(len(self.waypoints))}
        edge_paths = {}
        edge_costs = {}
        for first, second, distance in self._candidate_edges:
            if not (
                math.isfinite(waypoint_costs[first])
                and math.isfinite(waypoint_costs[second])
            ):
                continue
            evaluated = self._edge(
                costs, waypoint_costs, first, second, distance, baseline_cost
            )
            if evaluated is None:
                continue
            cost, path = evaluated
            adjacency[first].append((second, cost))
            adjacency[second].append((first, cost))
            edge_paths[(first, second)] = path
            edge_paths[(second, first)] = tuple(reversed(path))
            edge_costs[(first, second)] = cost
            edge_costs[(second, first)] = cost

        frontier = []
        distance_so_far = {}
        parent = {}
        for anchor, connector in start_links.items():
            connector_cost = float(connector.total_cost)
            if connector_cost < distance_so_far.get(anchor, math.inf):
                distance_so_far[anchor] = connector_cost
                parent[anchor] = None
                heapq.heappush(frontier, (connector_cost, anchor))
        best_goal = None
        best_total = math.inf
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            if current_cost > distance_so_far[current] + 1e-12:
                continue
            if current in goal_links:
                total = current_cost + float(goal_links[current].total_cost)
                if total < best_total:
                    best_total = total
                    best_goal = current
            if current_cost >= best_total:
                continue
            for neighbor, edge_cost in adjacency[current]:
                new_cost = current_cost + edge_cost
                if new_cost + 1e-12 < distance_so_far.get(neighbor, math.inf):
                    distance_so_far[neighbor] = new_cost
                    parent[neighbor] = current
                    heapq.heappush(frontier, (new_cost, neighbor))
        if best_goal is None:
            return self._fallback(costs, start, goal, "no safe reference graph route")

        anchors = []
        current = best_goal
        while current is not None:
            anchors.append(current)
            current = parent[current]
        anchors.reverse()

        path = []
        _append_path(path, start_links[anchors[0]].path)
        for first, second in zip(anchors, anchors[1:]):
            _append_path(path, edge_paths[(first, second)])
        _append_path(path, goal_links[anchors[-1]].path)
        total_cost = (
            float(start_links[anchors[0]].total_cost)
            + sum(
                edge_costs[(first, second)]
                for first, second in zip(
                    anchors, anchors[1:]
                )
            )
            + float(goal_links[anchors[-1]].total_cost)
        )
        result = AStarResult(
            path, total_cost, "reference waypoint graph path found",
            reference_waypoint_ids=tuple(
                self.waypoints[index].waypoint_id for index in anchors
            ),
            used_reference_graph=True,
        )
        print(f"[Planner] WAYPOINT_GRAPH_PATH path_length={len(result.path)}")
        return result

    def _fallback(self, costs, start, goal, graph_reason):
        if not self.config.fallback_to_cell_astar:
            return AStarResult([], math.inf, graph_reason)
        print(
            f"[Planner] WAYPOINT_GRAPH_FAILED reason={graph_reason} "
            "-> trying unweighted cell A*"
        )
        print("[Planner] UNWEIGHTED_CELL_ASTAR_FALLBACK")
        result = unweighted_a_star(costs, start, goal)
        if result.path:
            print(
                "[Planner] CELL_ASTAR_PATH "
                f"path_length={len(result.path)}"
            )
        else:
            print(
                "[Planner] TRUE_NO_PATH "
                f"graph_reason={graph_reason} cell_reason={result.reason}"
            )
        return AStarResult(
            result.path, result.total_cost,
            f"{graph_reason}; cell A* fallback: {result.reason}",
            result.escape_path, result.replan_start,
            result.reference_waypoint_ids, False,
        )
