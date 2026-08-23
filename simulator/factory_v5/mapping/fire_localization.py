"""Sensor-only, accumulated fire-source localization for factory_v5.

All 2-D arrays use ``[row, col] == [y, x]``.  Thermal ray samples may carry
FDS ``(x, y, z)`` coordinates, but are converted through :class:`MapMetadata`
before evidence is applied.  Ground Truth arrays are deliberately absent from
this module's API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Iterable

import numpy as np

from world.fire_maps import MapMetadata


class FireEstimateState(Enum):
    UNOBSERVED = "unobserved"
    WEAK_EVIDENCE = "weak_evidence"
    POSSIBLE_FIRE = "possible_fire"
    LIKELY_FIRE = "likely_fire"
    CONFIRMED_FIRE_REGION = "confirmed_fire_region"


class FireSensorType(Enum):
    THERMAL = "thermal"
    CO_GRADIENT = "co_gradient"


@dataclass(frozen=True)
class FireLocalizationConfig:
    enabled: bool = True
    prior_probability: float = 0.05
    possible_probability_threshold: float = 0.40
    likely_probability_threshold: float = 0.65
    confirm_probability_threshold: float = 0.80
    release_probability_threshold: float = 0.70
    thermal_warning_threshold_c: float = 40.0
    thermal_strong_threshold_c: float = 60.0
    thermal_evidence_weight: float = 1.0
    co_rise_threshold_ppm: float = 5.0
    temperature_rise_threshold_c: float = 2.0
    minimum_robot_motion_m: float = 0.2
    co_gradient_evidence_weight: float = 0.4
    minimum_consecutive_co_rises: int = 2
    maximum_observation_interval_s: float = 2.0
    minimum_distinct_observation_poses: int = 2
    minimum_confirmation_observations: int = 3
    distinct_pose_distance_m: float = 0.5
    distinct_heading_deg: float = 15.0
    maximum_evidence_per_update: float = 1.0
    maximum_accumulated_evidence: float = 8.0
    evidence_decay_per_second: float = 0.001
    candidate_probability_threshold: float = 0.50
    maximum_confirmed_region_cells: int = 250
    estimated_fire_cost_weight: float = 0.0
    co_projection_distance_m: float = 4.0
    co_projection_half_angle_deg: float = 15.0

    @classmethod
    def from_mapping(cls, values) -> "FireLocalizationConfig":
        values = {} if values is None else dict(values)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown fire_localization settings: {sorted(unknown)}")
        return cls(**values)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("fire_localization.enabled must be boolean")
        probabilities = (
            self.prior_probability, self.possible_probability_threshold,
            self.likely_probability_threshold, self.confirm_probability_threshold,
            self.release_probability_threshold, self.candidate_probability_threshold,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
            raise ValueError("fire localization probabilities must be in [0,1]")
        if not (
            self.prior_probability < self.possible_probability_threshold
            < self.likely_probability_threshold < self.confirm_probability_threshold
        ):
            raise ValueError("fire probability thresholds must be strictly increasing")
        if self.release_probability_threshold >= self.confirm_probability_threshold:
            raise ValueError("release threshold must be below confirmation threshold")
        if self.thermal_strong_threshold_c <= self.thermal_warning_threshold_c:
            raise ValueError("thermal strong threshold must exceed warning threshold")
        nonnegative = (
            self.thermal_evidence_weight, self.co_gradient_evidence_weight,
            self.evidence_decay_per_second, self.estimated_fire_cost_weight,
        )
        if any(float(value) < 0.0 for value in nonnegative):
            raise ValueError("fire localization weights and decay must be non-negative")
        positive = (
            self.minimum_robot_motion_m, self.maximum_observation_interval_s,
            self.distinct_pose_distance_m, self.maximum_evidence_per_update,
            self.maximum_accumulated_evidence, self.co_projection_distance_m,
        )
        if any(float(value) <= 0.0 for value in positive):
            raise ValueError("fire localization distances/limits must be positive")
        if self.minimum_consecutive_co_rises < 1 or self.minimum_confirmation_observations < 1:
            raise ValueError("minimum observation counts must be at least one")
        if self.minimum_distinct_observation_poses < 1 or self.maximum_confirmed_region_cells < 1:
            raise ValueError("pose and region counts must be at least one")
        if not 0.0 <= self.distinct_heading_deg <= 180.0:
            raise ValueError("distinct_heading_deg must be in [0,180]")
        if not 0.0 <= self.co_projection_half_angle_deg <= 90.0:
            raise ValueError("co projection half angle must be in [0,90]")


@dataclass(frozen=True)
class FireObservationRecord:
    observation_id: str
    simulation_time: float
    robot_pose_world: tuple[float, float, float]
    sensor_type: FireSensorType
    measured_value: float
    direction_world: tuple[float, float] | None
    applied_cells_grid: tuple[tuple[int, int], ...]
    valid: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sensor_type"] = self.sensor_type.value
        return data


@dataclass(frozen=True)
class FireLocalizationResult:
    state: FireEstimateState
    highest_probability: float
    highest_probability_grid: tuple[int, int] | None
    highest_probability_world: tuple[float, float] | None
    candidate_cells_grid: tuple[tuple[int, int], ...]
    candidate_bounding_box_grid: tuple[int, int, int, int] | None
    weighted_center_world: tuple[float, float] | None
    confidence: float
    valid_observation_count: int
    distinct_pose_count: int
    last_updated_at: float | None
    confirmed: bool

    def to_dict(self) -> dict:
        data = asdict(self)
        data["state"] = self.state.value
        return data


class FireLocalizer:
    """Fuse obstacle-aware thermal rays and motion-conditioned CO gradients."""

    def __init__(
        self, metadata: MapMetadata, static_obstacle_map,
        config: FireLocalizationConfig | None = None,
    ) -> None:
        self.metadata = metadata
        self.config = config or FireLocalizationConfig()
        shape = (metadata.height, metadata.width)
        static = np.asarray(static_obstacle_map, dtype=bool)
        if static.shape != shape:
            raise ValueError(f"occupancy shape {static.shape} does not match {shape}")
        self.static_obstacle_map = static.copy()
        self.thermal_fire_evidence = np.zeros(shape, dtype=float)
        self.co_gradient_evidence = np.zeros(shape, dtype=float)
        self.temporal_consistency_evidence = np.zeros(shape, dtype=float)
        self.combined_fire_evidence = np.zeros(shape, dtype=float)
        self.fire_probability = np.full(shape, self.config.prior_probability, dtype=float)
        self.observation_count_map = np.zeros(shape, dtype=np.int32)
        self.last_observed_time_map = np.full(shape, np.nan, dtype=float)
        self._thermal_count_map = np.zeros(shape, dtype=np.int32)
        self._co_count_map = np.zeros(shape, dtype=np.int32)
        self._pose_keys_by_cell: dict[tuple[int, int], set[tuple[int, int, int]]] = {}
        self._observation_ids: set[str] = set()
        self.observation_history: list[FireObservationRecord] = []
        self._last_time: float | None = None
        self._last_co_sample: tuple[float, tuple[float, float, float], float, float] | None = None
        self._co_rise_streak = 0
        self._confirmed = False
        self.latest_result = self.result()

    def _prepare(self, observation_id: str, simulation_time: float) -> float:
        if not observation_id:
            raise ValueError("observation_id must not be empty")
        if observation_id in self._observation_ids:
            raise ValueError(f"duplicate observation ID: {observation_id}")
        now = float(simulation_time)
        if not math.isfinite(now):
            raise ValueError("simulation_time must be finite")
        if self._last_time is not None and now < self._last_time - 1e-12:
            raise ValueError("simulation_time must not move backwards")
        dt = 0.0 if self._last_time is None else now - self._last_time
        if dt > 0.0:
            factor = math.exp(-self.config.evidence_decay_per_second * dt)
            self.thermal_fire_evidence *= factor
            self.co_gradient_evidence *= factor
            self.temporal_consistency_evidence *= factor
        self._last_time = now
        self._observation_ids.add(observation_id)
        return now

    def _pose_key(self, pose) -> tuple[int, int, int]:
        x, y, heading = map(float, pose)
        distance = self.config.distinct_pose_distance_m
        angle = max(self.config.distinct_heading_deg, 1e-6)
        return (round(x / distance), round(y / distance), round(math.degrees(heading) / angle))

    def _world_cell(self, world_position) -> tuple[int, int] | None:
        x, y = float(world_position[0]), float(world_position[1])
        if not self.metadata.is_world_position_in_bounds(x, y):
            return None
        col, row = self.metadata.world_to_grid(x, y)
        return col, row

    def _temperature_strength(self, temperature: float) -> float:
        cfg = self.config
        if temperature < cfg.thermal_warning_threshold_c:
            return 0.0
        span = cfg.thermal_strong_threshold_c - cfg.thermal_warning_threshold_c
        normalized = (temperature - cfg.thermal_warning_threshold_c) / span
        return cfg.thermal_evidence_weight * min(2.0, 0.25 + 0.75 * max(0.0, normalized))

    def add_thermal_observation(
        self, observation_id: str, simulation_time: float,
        robot_pose_world: tuple[float, float, float], ray_observations,
    ) -> FireLocalizationResult:
        now = self._prepare(observation_id, simulation_time)
        pose = tuple(map(float, robot_pose_world))
        frame_delta: dict[tuple[int, int], float] = {}
        hottest = -math.inf
        for ray in ray_observations:
            temperature = float(ray.pixel_temperature)
            if not ray.valid or not math.isfinite(temperature):
                continue
            hottest = max(hottest, temperature)
            strength = self._temperature_strength(temperature)
            if strength <= 0.0:
                continue
            unique_cells: list[tuple[int, int]] = []
            for sample in ray.ray_cells:
                cell = self._world_cell(sample.world_position)
                if cell is not None and (not unique_cells or unique_cells[-1] != cell):
                    unique_cells.append(cell)
            if not unique_cells:
                continue
            # The max-temperature model has no depth.  Spread bounded evidence
            # across the entire visible ray, with a mild far-distance bias;
            # crossing rays reinforce their common cells over later poses.
            weights = np.linspace(0.5, 1.0, len(unique_cells), dtype=float)
            weights /= max(float(weights.sum()), 1e-12)
            for cell, weight in zip(unique_cells, weights):
                frame_delta[cell] = frame_delta.get(cell, 0.0) + strength * float(weight)
        cap = self.config.maximum_evidence_per_update
        pose_key = self._pose_key(pose)
        applied = []
        for (col, row), value in frame_delta.items():
            delta = min(cap, value)
            prior_count = self.observation_count_map[row, col]
            self.thermal_fire_evidence[row, col] += delta
            if prior_count:
                self.temporal_consistency_evidence[row, col] += min(0.25 * delta, cap)
            self.observation_count_map[row, col] += 1
            self._thermal_count_map[row, col] += 1
            self.last_observed_time_map[row, col] = now
            self._pose_keys_by_cell.setdefault((col, row), set()).add(pose_key)
            applied.append((col, row))
        self.observation_history.append(FireObservationRecord(
            observation_id, now, pose, FireSensorType.THERMAL,
            hottest if math.isfinite(hottest) else math.nan,
            (math.cos(pose[2]), math.sin(pose[2])), tuple(sorted(applied)),
            bool(applied),
        ))
        self._recalculate()
        return self.latest_result

    def add_co_observation(
        self, observation_id: str, simulation_time: float,
        robot_pose_world: tuple[float, float, float], co_ppm: float,
        local_temperature_c: float, *, thermal_direction_supported: bool = False,
    ) -> FireLocalizationResult:
        now = self._prepare(observation_id, simulation_time)
        pose = tuple(map(float, robot_pose_world))
        co = float(co_ppm)
        temp = float(local_temperature_c)
        if not math.isfinite(co) or not math.isfinite(temp):
            raise ValueError("CO and local temperature observations must be finite")
        applied: list[tuple[int, int]] = []
        direction = None
        previous = self._last_co_sample
        if previous is not None:
            old_time, old_pose, old_co, old_temp = previous
            dt = now - old_time
            dx, dy = pose[0] - old_pose[0], pose[1] - old_pose[1]
            distance = math.hypot(dx, dy)
            delta_co, delta_temp = co - old_co, temp - old_temp
            co_trend_qualifies = (
                0.0 < dt <= self.config.maximum_observation_interval_s
                and distance >= self.config.minimum_robot_motion_m
                and delta_co >= self.config.co_rise_threshold_ppm
            )
            supported = (
                delta_temp >= self.config.temperature_rise_threshold_c
                or thermal_direction_supported
            )
            self._co_rise_streak = self._co_rise_streak + 1 if co_trend_qualifies else 0
            if co_trend_qualifies and self._co_rise_streak >= self.config.minimum_consecutive_co_rises:
                direction = (dx / distance, dy / distance)
                gradient = delta_co / distance
                normalized = min(2.0, gradient / max(self.config.co_rise_threshold_ppm, 1e-9))
                # A repeated CO-only spatial trend is weak evidence. Thermal
                # or local-temperature agreement increases it substantially.
                support_scale = 1.0 if supported else 0.35
                strength = min(
                    self.config.maximum_evidence_per_update,
                    self.config.co_gradient_evidence_weight * normalized
                    * support_scale,
                )
                applied = self._project_direction(pose, direction, strength, now)
        self._last_co_sample = (now, pose, co, temp)
        self.observation_history.append(FireObservationRecord(
            observation_id, now, pose, FireSensorType.CO_GRADIENT, co,
            direction, tuple(applied), bool(applied),
        ))
        self._recalculate()
        return self.latest_result

    def _project_direction(self, pose, direction, strength, now) -> list[tuple[int, int]]:
        applied: set[tuple[int, int]] = set()
        base_angle = math.atan2(direction[1], direction[0])
        angles = (-self.config.co_projection_half_angle_deg, 0.0,
                  self.config.co_projection_half_angle_deg)
        step = self.metadata.resolution_m * 0.5
        pose_key = self._pose_key(pose)
        for offset in angles:
            theta = base_angle + math.radians(offset)
            for distance in np.arange(step, self.config.co_projection_distance_m + step, step):
                cell = self._world_cell((
                    pose[0] + distance * math.cos(theta),
                    pose[1] + distance * math.sin(theta),
                ))
                if cell is None:
                    break
                col, row = cell
                if self.static_obstacle_map[row, col]:
                    break
                if cell in applied:
                    continue
                attenuation = max(0.2, 1.0 - distance / self.config.co_projection_distance_m)
                self.co_gradient_evidence[row, col] += strength * attenuation
                self.observation_count_map[row, col] += 1
                self._co_count_map[row, col] += 1
                self.last_observed_time_map[row, col] = now
                self._pose_keys_by_cell.setdefault(cell, set()).add(pose_key)
                applied.add(cell)
        return sorted(applied)

    def _recalculate(self) -> None:
        cap = self.config.maximum_accumulated_evidence
        overlap = np.minimum(self.thermal_fire_evidence, self.co_gradient_evidence)
        self.combined_fire_evidence = np.clip(
            self.thermal_fire_evidence + self.co_gradient_evidence
            + self.temporal_consistency_evidence + 0.5 * overlap,
            0.0, cap,
        )
        prior = self.config.prior_probability
        prior_logit = math.log(prior / (1.0 - prior))
        logits = prior_logit + self.combined_fire_evidence
        self.fire_probability = np.clip(1.0 / (1.0 + np.exp(-logits)), 0.0, 1.0)
        self.latest_result = self.result()

    def _candidate_component(self) -> tuple[tuple[int, int], ...]:
        mask = self.fire_probability >= self.config.candidate_probability_threshold
        if not np.any(mask):
            return ()
        highest_row, highest_col = np.unravel_index(np.argmax(self.fire_probability), self.fire_probability.shape)
        if not mask[highest_row, highest_col]:
            return ()
        stack = [(int(highest_col), int(highest_row))]
        seen = set(stack)
        while stack:
            col, row = stack.pop()
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = col + dc, row + dr
                if (nxt not in seen and self.metadata.is_grid_position_in_bounds(*nxt)
                        and mask[nxt[1], nxt[0]]):
                    seen.add(nxt)
                    stack.append(nxt)
        return tuple(sorted(seen, key=lambda cell: (cell[1], cell[0])))

    def result(self) -> FireLocalizationResult:
        observed = self.observation_count_map > 0
        if not np.any(observed):
            return FireLocalizationResult(
                FireEstimateState.UNOBSERVED, self.config.prior_probability,
                None, None, (), None, None, 0.0, 0, 0,
                self._last_time, False,
            )
        row, col = np.unravel_index(np.argmax(self.fire_probability), self.fire_probability.shape)
        highest = float(self.fire_probability[row, col])
        candidate = self._candidate_component()
        pose_count = len(self._pose_keys_by_cell.get((int(col), int(row)), ()))
        observation_count = int(self.observation_count_map[row, col])
        has_thermal = self._thermal_count_map[row, col] > 0
        has_support = self._co_count_map[row, col] > 0 or self.temporal_consistency_evidence[row, col] > 0
        confirmation_ready = (
            highest >= self.config.confirm_probability_threshold
            and observation_count >= self.config.minimum_confirmation_observations
            and pose_count >= self.config.minimum_distinct_observation_poses
            and has_thermal and has_support and candidate
            and len(candidate) <= self.config.maximum_confirmed_region_cells
        )
        if self._confirmed:
            self._confirmed = highest >= self.config.release_probability_threshold
        elif confirmation_ready:
            self._confirmed = True
        if self._confirmed:
            state = FireEstimateState.CONFIRMED_FIRE_REGION
        elif highest >= self.config.likely_probability_threshold:
            state = FireEstimateState.LIKELY_FIRE
        elif highest >= self.config.possible_probability_threshold:
            state = FireEstimateState.POSSIBLE_FIRE
        else:
            state = FireEstimateState.WEAK_EVIDENCE
        bbox = None
        center = None
        if candidate:
            cols = [item[0] for item in candidate]
            rows = [item[1] for item in candidate]
            bbox = (min(cols), min(rows), max(cols), max(rows))
            weights = np.asarray([self.fire_probability[r, c] for c, r in candidate])
            worlds = np.asarray([self.metadata.grid_to_world(c, r) for c, r in candidate])
            center_values = np.average(worlds, axis=0, weights=weights)
            center = (float(center_values[0]), float(center_values[1]))
        return FireLocalizationResult(
            state, highest, (int(col), int(row)),
            self.metadata.grid_to_world(int(col), int(row)), candidate, bbox,
            center, highest, observation_count, pose_count,
            None if not math.isfinite(self.last_observed_time_map[row, col])
            else float(self.last_observed_time_map[row, col]), self._confirmed,
        )

    def to_dict(self) -> dict:
        return {
            "axis_order": "[row,col]=[y,x]",
            "thermal_fire_evidence": self.thermal_fire_evidence.tolist(),
            "co_gradient_evidence": self.co_gradient_evidence.tolist(),
            "temporal_consistency_evidence": self.temporal_consistency_evidence.tolist(),
            "combined_fire_evidence": self.combined_fire_evidence.tolist(),
            "fire_probability": self.fire_probability.tolist(),
            "observation_count_map": self.observation_count_map.tolist(),
            "last_observed_time_map": self.last_observed_time_map.tolist(),
            "latest_result": self.latest_result.to_dict(),
            "observation_history": [item.to_dict() for item in self.observation_history],
        }
