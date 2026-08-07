"""Connectivity-aware exit blockage decisions from perceived obstacles."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from planner.a_star import weighted_a_star
from world.entities import ExitStatus


@dataclass(frozen=True)
class ExitBlockageConfig:
    enabled: bool = True
    approach_region_depth_m: float = 1.5
    required_clear_width_m: float = 0.8
    confirmation_observations: int = 2
    release_observations: int = 3
    consider_human_clearance: bool = True
    human_clearance_m: float = 0.2
    exclude_blocked_exit_immediately: bool = True
    ignored_exit_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown exit_blockage settings: {sorted(unknown)}")
        if "ignored_exit_ids" in values:
            raw = values["ignored_exit_ids"]
            if not isinstance(raw, (list, tuple)):
                raise TypeError("ignored_exit_ids must be a list of exit IDs")
            values["ignored_exit_ids"] = tuple(raw)
        return cls(**values)

    def __post_init__(self):
        for name in ("enabled", "consider_human_clearance", "exclude_blocked_exit_immediately"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.approach_region_depth_m < 0.0:
            raise ValueError("approach_region_depth_m must be non-negative")
        if self.required_clear_width_m <= 0.0:
            raise ValueError("required_clear_width_m must be positive")
        if self.human_clearance_m < 0.0:
            raise ValueError("human_clearance_m must be non-negative")
        if self.confirmation_observations < 1 or self.release_observations < 1:
            raise ValueError("exit blockage observation counts must be at least one")
        if any(not isinstance(exit_id, str) or not exit_id for exit_id in self.ignored_exit_ids):
            raise ValueError("ignored_exit_ids must contain non-empty strings")
        if len(set(self.ignored_exit_ids)) != len(self.ignored_exit_ids):
            raise ValueError("ignored_exit_ids must not contain duplicates")


@dataclass(frozen=True)
class ExitBlockageResult:
    exit_id: str
    blocked_geometry: bool
    reachable_goal_exists: bool
    blocked_confirmed: bool
    candidate_cells_grid: tuple[tuple[int, int], ...]
    reachable_cells_grid: tuple[tuple[int, int], ...]
    blocking_cells_grid: tuple[tuple[int, int], ...]
    clear_width_m: float
    observation_count: int
    reason: str
    evaluated_at: float
    environment_revision: int


class ExitBlockageEvaluator:
    def __init__(self, metadata, config=None):
        self.metadata = metadata
        self.config = config or ExitBlockageConfig()
        self._release_counts: dict[str, int] = {}

    def evaluate(
        self, exit_item, current_position_world, *, cost_map,
        static_obstacle_map, dynamic_inflated_map, active_obstacles,
        evaluated_at: float, environment_revision: int,
    ) -> ExitBlockageResult:
        costs = np.asarray(cost_map, dtype=float).copy()
        static = np.asarray(static_obstacle_map, dtype=bool)
        dynamic = np.asarray(dynamic_inflated_map, dtype=bool)
        expected = (self.metadata.height, self.metadata.width)
        if costs.shape != expected or static.shape != expected or dynamic.shape != expected:
            raise ValueError("exit blockage map shape mismatch")
        if exit_item.exit_id in self.config.ignored_exit_ids:
            approach = exit_item.approach_position_world or exit_item.position_world
            candidate = self.metadata.world_to_grid(*approach)
            return ExitBlockageResult(
                exit_item.exit_id, False, True, False,
                (candidate,), (candidate,), (),
                float(self.config.required_clear_width_m), 0,
                "automatic blockage disabled for configured FDS door mesh",
                float(evaluated_at), int(environment_revision),
            )
        start = self.metadata.world_to_grid(*current_position_world)
        approach = exit_item.approach_position_world or exit_item.position_world
        marker = exit_item.position_world
        direction = (marker[0] - approach[0], marker[1] - approach[1])
        norm = math.hypot(*direction)
        if norm <= 1e-12:
            direction = (1.0, 0.0)
        else:
            direction = (direction[0] / norm, direction[1] / norm)
        lateral = (-direction[1], direction[0])
        required_width = self.config.required_clear_width_m + (
            self.config.human_clearance_m
            if self.config.consider_human_clearance else 0.0
        )
        half_span = max(required_width, self.metadata.resolution_m)
        lateral_offsets = np.arange(
            -half_span, half_span + self.metadata.resolution_m * 0.5,
            self.metadata.resolution_m,
        )
        depth = min(
            max(norm - self.metadata.resolution_m, self.metadata.resolution_m),
            self.config.approach_region_depth_m,
        )
        goal_center = (
            approach[0] + direction[0] * depth,
            approach[1] + direction[1] * depth,
        )
        candidates = self._cells_at(goal_center, lateral, lateral_offsets)
        effective = costs.copy()
        effective[static | dynamic] = np.inf
        reachable = []
        for cell in candidates:
            if not np.isfinite(effective[cell[1], cell[0]]):
                continue
            if weighted_a_star(effective, start, cell).path:
                reachable.append(cell)

        samples = np.arange(0.0, depth + self.metadata.resolution_m * 0.5,
                            self.metadata.resolution_m)
        minimum_clear = math.inf
        blocking_cells: set[tuple[int, int]] = set()
        dynamic_closes_open_section = False
        required_cells = max(1, int(math.ceil(required_width / self.metadata.resolution_m)))
        for distance in samples:
            center = (
                approach[0] + direction[0] * distance,
                approach[1] + direction[1] * distance,
            )
            section = self._cells_at(center, lateral, lateral_offsets)
            baseline_run = self._longest_clear_run(section, static)
            combined_run = self._longest_clear_run(section, static | dynamic)
            minimum_clear = min(minimum_clear, combined_run * self.metadata.resolution_m)
            section_dynamic = [cell for cell in section if dynamic[cell[1], cell[0]]]
            if baseline_run >= required_cells and combined_run < required_cells and section_dynamic:
                dynamic_closes_open_section = True
                blocking_cells.update(section_dynamic)

        observation_count = max(
            (item.observation_count for item in active_obstacles
             if item.confirmed and self._near_region(item.position_world, approach, marker, half_span)),
            default=0,
        )
        geometry_blocked = dynamic_closes_open_section and not reachable
        confirmed = geometry_blocked and observation_count >= self.config.confirmation_observations
        if confirmed:
            self._release_counts[exit_item.exit_id] = 0
        elif exit_item.status is ExitStatus.BLOCKED:
            self._release_counts[exit_item.exit_id] = self._release_counts.get(exit_item.exit_id, 0) + 1
        reason = (
            "perceived obstacle closes exit width and no reachable goal candidate"
            if confirmed else
            "exit passage remains reachable" if reachable else
            "blockage evidence below confirmation threshold"
        )
        return ExitBlockageResult(
            exit_item.exit_id, geometry_blocked, bool(reachable), confirmed,
            tuple(candidates), tuple(reachable), tuple(sorted(blocking_cells)),
            0.0 if math.isinf(minimum_clear) else float(minimum_clear),
            observation_count, reason, float(evaluated_at), int(environment_revision),
        )

    def release_confirmed(self, exit_id: str) -> bool:
        return self._release_counts.get(exit_id, 0) >= self.config.release_observations

    def _cells_at(self, center, lateral, offsets):
        cells = []
        for offset in offsets:
            point = (center[0] + lateral[0] * offset,
                     center[1] + lateral[1] * offset)
            if not self.metadata.is_world_position_in_bounds(*point):
                continue
            cell = self.metadata.world_to_grid(*point)
            if cell not in cells:
                cells.append(cell)
        return cells

    @staticmethod
    def _longest_clear_run(cells, blocked):
        longest = current = 0
        for col, row in cells:
            if blocked[row, col]:
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        return longest

    @staticmethod
    def _near_region(point, approach, marker, margin):
        x1, x2 = sorted((approach[0], marker[0]))
        y1, y2 = sorted((approach[1], marker[1]))
        return (x1 - margin <= point[0] <= x2 + margin
                and y1 - margin <= point[1] <= y2 + margin)
