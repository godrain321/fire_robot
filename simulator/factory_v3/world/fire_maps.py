"""Planner-grid fire observations in explicit ``[row, col] == [y, x]`` order.

FDS source arrays remain ``[time,z,y,x]`` or an internal equivalent. They must
be sampled/resampled before entering these 2-D containers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MapMetadata:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution_m: float
    width: int
    height: int
    origin_world: tuple[float, float]
    axis_order: str = "[row,col]=[y,x]"

    def __post_init__(self) -> None:
        if self.resolution_m <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("map resolution and dimensions must be positive")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("map maximum bounds must exceed minimum bounds")
        if tuple(self.origin_world) != (self.x_min, self.y_min):
            raise ValueError("origin_world must equal the lower-left world bound")
        expected_width = int(round((self.x_max - self.x_min) / self.resolution_m)) + 1
        expected_height = int(round((self.y_max - self.y_min) / self.resolution_m)) + 1
        if (self.width, self.height) != (expected_width, expected_height):
            raise ValueError(
                f"metadata dimensions {(self.width, self.height)} do not match "
                f"bounds/resolution {(expected_width, expected_height)}"
            )

    @classmethod
    def from_grid_map(cls, grid_map) -> "MapMetadata":
        return cls(
            grid_map.x_min, grid_map.x_max, grid_map.y_min, grid_map.y_max,
            grid_map.resolution, grid_map.width, grid_map.height,
            (grid_map.x_min, grid_map.y_min),
        )

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        if not self.is_world_position_in_bounds(x, y):
            raise ValueError(f"world position {(x, y)} is outside map bounds")
        return (
            int(round((float(x) - self.x_min) / self.resolution_m)),
            int(round((float(y) - self.y_min) / self.resolution_m)),
        )

    def grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        if not self.is_grid_position_in_bounds(col, row):
            raise ValueError(f"grid position {(col, row)} is outside map bounds")
        return (
            self.x_min + int(col) * self.resolution_m,
            self.y_min + int(row) * self.resolution_m,
        )

    def is_world_position_in_bounds(self, x: float, y: float) -> bool:
        return (
            math.isfinite(float(x)) and math.isfinite(float(y))
            and self.x_min <= float(x) <= self.x_max
            and self.y_min <= float(y) <= self.y_max
        )

    def is_grid_position_in_bounds(self, col: int, row: int) -> bool:
        return 0 <= int(col) < self.width and 0 <= int(row) < self.height


class FireMap:
    """Raw 2-D fire layers; risk-cost calculation remains in partial_costmap."""

    def __init__(self, metadata: MapMetadata, *, temperature_blocked_c: float, co_blocked_ppm: float) -> None:
        self.metadata = metadata
        self.temperature_blocked_c = float(temperature_blocked_c)
        self.co_blocked_ppm = float(co_blocked_ppm)
        shape = (metadata.height, metadata.width)
        self.temperature_c = np.full(shape, np.nan, dtype=float)
        self.co_ppm = np.full(shape, np.nan, dtype=float)
        self.observed_mask = np.zeros(shape, dtype=bool)
        self.last_observed_time = np.full(shape, np.nan, dtype=float)
        self.axis_order = metadata.axis_order

    @property
    def shape(self) -> tuple[int, int]:
        return self.temperature_c.shape

    @property
    def blocked_mask(self) -> np.ndarray:
        return (
            (self.temperature_c >= self.temperature_blocked_c)
            | (self.co_ppm >= self.co_blocked_ppm)
        ) & self.observed_mask

    def update_cell(self, col: int, row: int, *, temperature_c=None, co_ppm=None, sim_time=None) -> None:
        if not self.metadata.is_grid_position_in_bounds(col, row):
            raise ValueError(f"grid position {(col, row)} is outside fire map")
        if temperature_c is None and co_ppm is None:
            raise ValueError("at least one fire observation is required")
        if temperature_c is not None:
            self.temperature_c[row, col] = float(temperature_c)
        if co_ppm is not None:
            self.co_ppm[row, col] = float(co_ppm)
        self.observed_mask[row, col] = True
        self.last_observed_time[row, col] = np.nan if sim_time is None else float(sim_time)

    def replace_layers(self, temperature_c, co_ppm, observed_mask, last_observed_time) -> None:
        arrays = [np.asarray(item) for item in (temperature_c, co_ppm, observed_mask, last_observed_time)]
        if any(item.shape != self.shape for item in arrays):
            raise ValueError(f"fire layer shape must be {self.shape}")
        self.temperature_c = np.asarray(temperature_c, dtype=float).copy()
        self.co_ppm = np.asarray(co_ppm, dtype=float).copy()
        self.observed_mask = np.asarray(observed_mask, dtype=bool).copy()
        self.last_observed_time = np.asarray(last_observed_time, dtype=float).copy()


class GroundTruthFireMap(FireMap):
    """FDS-derived samples used only by virtual sensors and evaluation."""


class EstimatedFireMap(FireMap):
    """Only localized sensor observations available to mapping/planning."""

    def sync_from_belief(self, belief) -> None:
        self.replace_layers(
            belief.temperature_belief_map,
            belief.co_belief_map,
            belief.observed_mask,
            belief.last_observed_time_map,
        )
