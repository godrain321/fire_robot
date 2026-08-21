"""Mission-specific search planning policy and UNKNOWN-frontier planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from planner.a_star import weighted_a_star


class SearchStage(Enum):
    CHECK_EXITS = "check_exits"
    EXPLORE_FRONTIERS = "explore_frontiers"
    RECHECK_EXITS = "recheck_exits"
    FINAL_EXIT = "final_exit"
    COMPLETE = "complete"


@dataclass(frozen=True)
class SearchPlanningConfig:
    temperature_block_c: float = 80.0
    block_on_co: bool = False
    frontier_connectivity: int = 4
    exit_temperature_radius_m: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature_block_c) or self.temperature_block_c <= 0:
            raise ValueError("search temperature_block_c must be positive and finite")
        if not isinstance(self.block_on_co, bool):
            raise TypeError("search block_on_co must be bool")
        if self.frontier_connectivity not in (4, 8):
            raise ValueError("frontier_connectivity must be 4 or 8")
        if not math.isfinite(self.exit_temperature_radius_m) or self.exit_temperature_radius_m < 0:
            raise ValueError("exit_temperature_radius_m must be finite and non-negative")

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown search_planning settings: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class FrontierPlan:
    success: bool
    target_grid: tuple[int, int] | None
    path_grid: tuple[tuple[int, int], ...]
    total_cost: float | None
    failure_reason: str | None


class SearchFireMapView:
    """Read-only threshold view used by validators and exit evaluation."""

    def __init__(self, estimated, *, temperature_block_c: float, block_on_co: bool):
        self.metadata = estimated.metadata
        self.temperature_c = estimated.temperature_c
        self.co_ppm = estimated.co_ppm
        self.observed_mask = estimated.observed_mask
        self.last_observed_time = estimated.last_observed_time
        self.temperature_blocked_c = float(temperature_block_c)
        self.co_blocked_ppm = (
            float(estimated.co_blocked_ppm) if block_on_co else math.inf
        )

    @property
    def shape(self):
        return self.temperature_c.shape

    @property
    def blocked_mask(self):
        return (
            self.observed_mask
            & (self.temperature_c >= self.temperature_blocked_c)
        )


def frontier_mask(observed_mask, planning_cost_map, *, connectivity: int = 4):
    """Observed traversable cells touching at least one UNKNOWN cell."""
    observed = np.asarray(observed_mask, dtype=bool)
    costs = np.asarray(planning_cost_map, dtype=float)
    if observed.shape != costs.shape:
        raise ValueError("observed mask and planning cost map shape mismatch")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    unknown = ~observed
    adjacent_unknown = np.zeros(observed.shape, dtype=bool)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    height, width = observed.shape
    for dr, dc in offsets:
        src_r = slice(max(0, -dr), min(height, height - dr))
        src_c = slice(max(0, -dc), min(width, width - dc))
        dst_r = slice(max(0, dr), min(height, height + dr))
        dst_c = slice(max(0, dc), min(width, width + dc))
        adjacent_unknown[dst_r, dst_c] |= unknown[src_r, src_c]
    return observed & np.isfinite(costs) & adjacent_unknown


def plan_nearest_frontier(
    planning_cost_map, observed_mask, start_grid,
    *, connectivity: int = 4, excluded_targets=(),
) -> FrontierPlan:
    """Select the nearest reachable frontier by path length, deterministically."""
    costs = np.asarray(planning_cost_map, dtype=float)
    mask = frontier_mask(observed_mask, costs, connectivity=connectivity)
    excluded = {tuple(map(int, item)) for item in excluded_targets}
    # Evaluate one deterministic representative per connected frontier
    # region, not one A* per boundary cell.
    remaining = {
        (int(col), int(row)) for row, col in np.argwhere(mask)
        if (int(col), int(row)) not in excluded
    }
    candidates = []
    adjacency = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        adjacency += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    while remaining:
        seed = min(remaining, key=lambda node: (node[1], node[0]))
        remaining.remove(seed)
        component = [seed]
        stack = [seed]
        while stack:
            col, row = stack.pop()
            for dc, dr in adjacency:
                neighbor = (col + dc, row + dr)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        candidates.append(min(component, key=lambda node: (
            math.hypot(node[0] - start_grid[0], node[1] - start_grid[1]),
            node[1], node[0],
        )))
    candidates.sort(key=lambda node: (
        math.hypot(node[0] - start_grid[0], node[1] - start_grid[1]),
        node[1], node[0],
    ))
    best = None
    # One representative per local vicinity keeps this event-driven search
    # bounded while still validating actual reachability with Weighted A*.
    for target in candidates:
        result = weighted_a_star(costs, start_grid, target)
        if not result.path:
            continue
        length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(result.path, result.path[1:])
        )
        key = (length, result.total_cost, target[1], target[0])
        if best is None or key < best[0]:
            best = (key, target, result)
    if best is None:
        return FrontierPlan(False, None, tuple(), None, "no_reachable_frontier")
    return FrontierPlan(
        True, best[1], tuple(best[2].path), float(best[2].total_cost), None
    )


def representative_exit_temperature_cost(
    temperature_cost_map, temperature_observed_mask, center_grid,
    *, radius_cells: int,
) -> float | None:
    costs = np.asarray(temperature_cost_map, dtype=float)
    observed = np.asarray(temperature_observed_mask, dtype=bool)
    if costs.shape != observed.shape or radius_cells < 0:
        raise ValueError("invalid exit temperature inputs")
    col0, row0 = center_grid
    values = []
    for row in range(row0 - radius_cells, row0 + radius_cells + 1):
        for col in range(col0 - radius_cells, col0 + radius_cells + 1):
            if not (0 <= row < costs.shape[0] and 0 <= col < costs.shape[1]):
                continue
            if math.hypot(col - col0, row - row0) > radius_cells + 1e-9:
                continue
            value = costs[row, col]
            if observed[row, col] and np.isfinite(value):
                values.append(float(value))
    return None if not values else float(np.mean(values))
