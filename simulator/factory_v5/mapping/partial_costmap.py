"""Robot belief costmap updated exclusively from localized sensor observations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from mapping.fire_costmap import FireCostmapConfig
from mapping.grid_map import GridMap


@dataclass(frozen=True)
class PartialCostmapConfig(FireCostmapConfig):
    """All stage-2 mapping, sensing, replanning and motion settings."""

    unknown_penalty: float = 2.0
    unobserved_temperature_penalty: float = 1.0
    unobserved_co_penalty: float = 1.0
    replan_interval_seconds: float = 1.0
    replan_distance: float = 0.0
    sensor_update_interval_seconds: float = 0.25
    robot_speed: float = 0.45
    robot_angular_speed: float = math.radians(45.0)
    waypoint_tolerance: float = 0.08
    gas_update_radius: float = 0.0
    gas_gaussian_sigma: float = 0.5
    selected_fds_start_time: float = 0.0
    simulation_dt: float = 0.1
    render_fps: int = 30
    stale_observation_cost_enabled: bool = True
    stale_observation_grace_period_s: float = 5.0
    stale_observation_cost_per_second: float = 0.05
    stale_observation_maximum_cost: float = 2.0
    stale_observation_apply_to_temperature: bool = True
    stale_observation_apply_to_co: bool = True
    stale_observation_block_cells: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        non_negative = {
            "unknown_penalty": self.unknown_penalty,
            "unobserved_temperature_penalty": self.unobserved_temperature_penalty,
            "unobserved_co_penalty": self.unobserved_co_penalty,
            "replan_distance": self.replan_distance,
            "gas_update_radius": self.gas_update_radius,
            "selected_fds_start_time": self.selected_fds_start_time,
            "stale_observation_grace_period_s": (
                self.stale_observation_grace_period_s
            ),
            "stale_observation_cost_per_second": (
                self.stale_observation_cost_per_second
            ),
            "stale_observation_maximum_cost": (
                self.stale_observation_maximum_cost
            ),
        }
        for name, value in non_negative.items():
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        positive = {
            "replan_interval_seconds": self.replan_interval_seconds,
            "sensor_update_interval_seconds": self.sensor_update_interval_seconds,
            "robot_speed": self.robot_speed,
            "robot_angular_speed": self.robot_angular_speed,
            "waypoint_tolerance": self.waypoint_tolerance,
            "gas_gaussian_sigma": self.gas_gaussian_sigma,
            "simulation_dt": self.simulation_dt,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.render_fps, bool)
            or not isinstance(self.render_fps, int)
            or self.render_fps < 1
        ):
            raise ValueError("render_fps must be a positive integer")
        boolean_fields = (
            "stale_observation_cost_enabled",
            "stale_observation_apply_to_temperature",
            "stale_observation_apply_to_co",
            "stale_observation_block_cells",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a Boolean")
        if self.stale_observation_block_cells:
            raise ValueError(
                "stale observations may add uncertainty cost but may not block cells"
            )


@dataclass(frozen=True)
class BeliefUpdate:
    """Cells changed by one sensor update."""

    changed_cells: frozenset[tuple[int, int]]
    newly_blocked_cells: frozenset[tuple[int, int]]


class PartialFireCostmap:
    """Maintain fire-risk belief without retaining any Ground Truth arrays."""

    def __init__(
        self,
        grid_map: GridMap,
        static_obstacle_map: np.ndarray,
        config: PartialCostmapConfig,
    ) -> None:
        self.grid_map = grid_map
        self.config = config
        shape = (grid_map.height, grid_map.width)
        static = np.asarray(static_obstacle_map, dtype=bool)
        if static.shape != shape:
            raise ValueError(f"static map shape={static.shape}, expected={shape}")

        self.static_obstacle_map = static.copy()
        self.temperature_observed_mask = np.zeros(shape, dtype=bool)
        self.co_observed_mask = np.zeros(shape, dtype=bool)
        self.observed_mask = np.zeros(shape, dtype=bool)
        self.temperature_belief_map = np.full(shape, np.nan, dtype=float)
        self.co_belief_map = np.full(shape, np.nan, dtype=float)
        self.co_confidence_map = np.zeros(shape, dtype=float)
        self.temperature_cost_map = np.zeros(shape, dtype=float)
        self.co_cost_map = np.zeros(shape, dtype=float)
        self.unknown_cost_map = np.zeros(shape, dtype=float)
        self.estimated_fire_cost_map = np.zeros(shape, dtype=float)
        self.stale_observation_cost_map = np.zeros(shape, dtype=float)
        self.dynamic_obstacle_map = np.zeros(shape, dtype=bool)
        self.dynamic_inflated_obstacle_map = np.zeros(shape, dtype=bool)
        self.blocked_mask = static.copy()
        self.final_cost_map = np.full(shape, config.base_cost, dtype=float)
        self.last_observed_time_map = np.full(shape, np.nan, dtype=float)
        self.current_time = 0.0
        # Monotonic sensor-belief revision.  A revision requests inexpensive
        # path validation; it does not by itself force a full replan.
        self.revision = 0
        self.recalculate()

    @property
    def shape(self) -> tuple[int, int]:
        return self.final_cost_map.shape

    def _snapshot_blocked(self) -> np.ndarray:
        return self.blocked_mask.copy()

    def _finish_update(
        self, changed: set[tuple[int, int]], old_blocked: np.ndarray
    ) -> BeliefUpdate:
        self.recalculate()
        if changed:
            self.revision += 1
        new_blocked = self.blocked_mask & ~old_blocked
        newly_blocked = {
            (gx, gy) for gy, gx in np.argwhere(new_blocked)
        }
        return BeliefUpdate(frozenset(changed), frozenset(newly_blocked))

    def update_thermal_observations(
        self,
        temperature_celsius: np.ndarray,
        ray_observations,
        sim_time: float,
    ) -> BeliefUpdate:
        """Project the raw 24x32 sensor array using per-pixel hit locations.

        The MLX90640 has no physical depth output. Stage 2 explicitly uses the
        simulator's selected ray-hit location as localization metadata. The
        numeric temperature always comes from the same pre-visualization array
        sent to the thermal viewer; no screen pixel or colormap is read. For
        duplicate XY hits within a scan, the maximum pixel value is retained.
        """
        temperatures = np.asarray(temperature_celsius, dtype=float)
        if temperatures.ndim != 2:
            raise ValueError(
                f"temperature_celsius must be a 2-D sensor array: {temperatures.shape}"
            )
        old_blocked = self._snapshot_blocked()
        scan_values: dict[tuple[int, int], float] = {}
        for observation in ray_observations:
            if not observation.valid or observation.hit_world_position is None:
                continue
            if not (
                0 <= observation.row < temperatures.shape[0]
                and 0 <= observation.col < temperatures.shape[1]
            ):
                raise ValueError(
                    "Thermal observation pixel index is outside sensor array: "
                    f"({observation.row}, {observation.col}) vs {temperatures.shape}"
                )
            gx, gy = self.grid_map.world_to_grid(
                observation.hit_world_position[0],
                observation.hit_world_position[1],
            )
            node = (gx, gy)
            if not self.grid_map.in_bounds(node):
                continue
            if self.static_obstacle_map[gy, gx]:
                continue
            pixel_temperature = float(
                temperatures[observation.row, observation.col]
            )
            if not np.isfinite(pixel_temperature):
                continue
            current = scan_values.get(node, -math.inf)
            scan_values[node] = max(current, pixel_temperature)

        changed = set(scan_values)
        for (gx, gy), temperature in scan_values.items():
            # Latest-scan replacement is simple and supports time-varying FDS.
            self.temperature_belief_map[gy, gx] = temperature
            self.temperature_observed_mask[gy, gx] = True
            self.last_observed_time_map[gy, gx] = sim_time
        return self._finish_update(changed, old_blocked)

    def update_co_observation(
        self,
        robot_x: float,
        robot_y: float,
        measured_ppm: float,
        sim_time: float,
    ) -> BeliefUpdate:
        """Update the current cell, optionally with a small Gaussian footprint."""
        old_blocked = self._snapshot_blocked()
        center = self.grid_map.world_to_grid(robot_x, robot_y)
        if not self.grid_map.in_bounds(center) or not np.isfinite(measured_ppm):
            return BeliefUpdate(frozenset(), frozenset())

        radius_cells = int(math.ceil(
            self.config.gas_update_radius / self.grid_map.resolution
        ))
        changed: set[tuple[int, int]] = set()
        for gy in range(center[1] - radius_cells, center[1] + radius_cells + 1):
            for gx in range(center[0] - radius_cells, center[0] + radius_cells + 1):
                node = (gx, gy)
                if not self.grid_map.in_bounds(node):
                    continue
                distance = math.hypot(gx - center[0], gy - center[1]) * self.grid_map.resolution
                if distance > self.config.gas_update_radius + 1e-9:
                    continue
                if self.static_obstacle_map[gy, gx]:
                    continue
                confidence = math.exp(
                    -(distance ** 2) / (2.0 * self.config.gas_gaussian_sigma ** 2)
                )
                self.co_belief_map[gy, gx] = float(measured_ppm)
                self.co_confidence_map[gy, gx] = confidence
                self.co_observed_mask[gy, gx] = True
                self.last_observed_time_map[gy, gx] = sim_time
                changed.add(node)
        return self._finish_update(changed, old_blocked)

    def update_estimated_fire_probability(
        self, probability_map, *, cost_weight: float,
        minimum_probability: float,
    ) -> BeliefUpdate:
        """Apply a finite sensor-inferred risk layer without creating blocks."""
        values = np.asarray(probability_map, dtype=float)
        if values.shape != self.shape:
            raise ValueError(f"fire probability shape={values.shape}, expected={self.shape}")
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("fire probabilities must be finite and in [0,1]")
        if cost_weight < 0.0 or not 0.0 <= minimum_probability <= 1.0:
            raise ValueError("invalid estimated fire cost settings")
        old_blocked = self._snapshot_blocked()
        new_cost = np.where(
            values >= minimum_probability, values * float(cost_weight), 0.0
        )
        changed_indices = np.argwhere(~np.isclose(
            new_cost, self.estimated_fire_cost_map, rtol=1e-9, atol=1e-12
        ))
        changed = {(int(col), int(row)) for row, col in changed_indices}
        self.estimated_fire_cost_map = new_cost
        return self._finish_update(changed, old_blocked)

    def update_dynamic_obstacles(
        self, obstacle_map, *, inflation_radius_m: float,
    ) -> BeliefUpdate:
        """Replace sensor-known dynamic occupancy and inflate it for planning."""
        raw = np.asarray(obstacle_map, dtype=bool)
        if raw.shape != self.shape:
            raise ValueError(f"dynamic obstacle shape={raw.shape}, expected={self.shape}")
        if inflation_radius_m < 0.0:
            raise ValueError("dynamic obstacle inflation radius must be non-negative")
        inflated = raw.copy()
        radius_cells = int(math.ceil(
            inflation_radius_m / self.grid_map.resolution
        ))
        if radius_cells:
            for row, col in np.argwhere(raw):
                for dy in range(-radius_cells, radius_cells + 1):
                    for dx in range(-radius_cells, radius_cells + 1):
                        if math.hypot(dx, dy) * self.grid_map.resolution > inflation_radius_m + 1e-12:
                            continue
                        yy, xx = int(row + dy), int(col + dx)
                        if 0 <= yy < self.shape[0] and 0 <= xx < self.shape[1]:
                            inflated[yy, xx] = True
        changed_indices = np.argwhere(
            (raw != self.dynamic_obstacle_map)
            | (inflated != self.dynamic_inflated_obstacle_map)
        )
        if changed_indices.size == 0:
            return BeliefUpdate(frozenset(), frozenset())
        old_blocked = self._snapshot_blocked()
        self.dynamic_obstacle_map = raw.copy()
        self.dynamic_inflated_obstacle_map = inflated
        changed = {(int(col), int(row)) for row, col in changed_indices}
        return self._finish_update(changed, old_blocked)

    def advance_time(self, sim_time: float) -> BeliefUpdate:
        """Refresh bounded uncertainty cost for aging sensor observations."""
        now = float(sim_time)
        if not math.isfinite(now):
            raise ValueError("sim_time must be finite")
        if now < self.current_time - 1e-12:
            raise ValueError("sim_time must not move backwards")
        old_cost = self.stale_observation_cost_map.copy()
        old_blocked = self._snapshot_blocked()
        self.current_time = now
        self.recalculate()
        changed_indices = np.argwhere(
            ~np.isclose(
                old_cost, self.stale_observation_cost_map,
                rtol=1e-9, atol=1e-12,
            )
        )
        changed = {
            (int(col), int(row)) for row, col in changed_indices
        }
        if changed:
            self.revision += 1
        return BeliefUpdate(frozenset(changed), frozenset())

    def recalculate(self) -> None:
        """Rebuild costs using observed values and per-modality uncertainty."""
        cfg = self.config
        self.observed_mask = (
            self.temperature_observed_mask | self.co_observed_mask
        )

        temp_normalized = np.zeros(self.shape, dtype=float)
        temp_normalized[self.temperature_observed_mask] = np.clip(
            (self.temperature_belief_map[self.temperature_observed_mask]
             - cfg.temperature_safe)
            / (cfg.temperature_blocked - cfg.temperature_safe),
            0.0,
            1.0,
        )
        self.temperature_cost_map = (
            cfg.temperature_weight * temp_normalized ** cfg.temperature_power
        )

        co_normalized = np.zeros(self.shape, dtype=float)
        co_normalized[self.co_observed_mask] = np.clip(
            (self.co_belief_map[self.co_observed_mask] - cfg.co_safe)
            / (cfg.co_blocked - cfg.co_safe),
            0.0,
            1.0,
        )
        self.co_cost_map = cfg.co_weight * co_normalized ** cfg.co_power

        neither_observed = ~self.temperature_observed_mask & ~self.co_observed_mask
        self.unknown_cost_map = np.zeros(self.shape, dtype=float)
        self.unknown_cost_map[neither_observed] = cfg.unknown_penalty
        partially_observed = ~neither_observed
        self.unknown_cost_map[
            partially_observed & ~self.temperature_observed_mask
        ] += cfg.unobserved_temperature_penalty
        self.unknown_cost_map[
            partially_observed & ~self.co_observed_mask
        ] += cfg.unobserved_co_penalty

        eligible_for_aging = np.zeros(self.shape, dtype=bool)
        if cfg.stale_observation_apply_to_temperature:
            eligible_for_aging |= self.temperature_observed_mask
        if cfg.stale_observation_apply_to_co:
            eligible_for_aging |= self.co_observed_mask
        self.stale_observation_cost_map = np.zeros(self.shape, dtype=float)
        if cfg.stale_observation_cost_enabled:
            valid_time = eligible_for_aging & np.isfinite(
                self.last_observed_time_map
            )
            age = np.zeros(self.shape, dtype=float)
            age[valid_time] = np.maximum(
                0.0,
                self.current_time - self.last_observed_time_map[valid_time],
            )
            stale_age = np.maximum(
                0.0, age - cfg.stale_observation_grace_period_s
            )
            self.stale_observation_cost_map[valid_time] = np.minimum(
                cfg.stale_observation_maximum_cost,
                stale_age[valid_time]
                * cfg.stale_observation_cost_per_second,
            )

        temperature_blocked = (
            self.temperature_observed_mask
            & (self.temperature_belief_map >= cfg.temperature_blocked)
        )
        co_blocked = (
            self.co_observed_mask & (self.co_belief_map >= cfg.co_blocked)
        )
        self.blocked_mask = (
            self.static_obstacle_map | self.dynamic_inflated_obstacle_map
            | temperature_blocked | co_blocked
        )
        self.final_cost_map = (
            cfg.base_cost
            + self.temperature_cost_map
            + self.co_cost_map
            + self.unknown_cost_map
            + self.estimated_fire_cost_map
            + self.stale_observation_cost_map
        )
        self.final_cost_map[self.blocked_mask] = np.inf
        self._validate_layers()

    def _validate_layers(self) -> None:
        expected = self.shape
        names = (
            "static_obstacle_map", "observed_mask",
            "temperature_observed_mask", "co_observed_mask",
            "temperature_belief_map", "co_belief_map",
            "temperature_cost_map", "co_cost_map", "unknown_cost_map",
            "estimated_fire_cost_map",
            "stale_observation_cost_map",
            "dynamic_obstacle_map", "dynamic_inflated_obstacle_map",
            "blocked_mask", "final_cost_map", "last_observed_time_map",
        )
        mismatches = {
            name: getattr(self, name).shape for name in names
            if getattr(self, name).shape != expected
        }
        if mismatches:
            raise ValueError(f"Belief layer shape mismatch: {mismatches}")


def path_has_new_block(path, newly_blocked_cells, lookahead: int = 12) -> bool:
    """Return whether a newly blocked cell lies in the path lookahead."""
    return bool(set(path[1:1 + lookahead]).intersection(newly_blocked_cells))
