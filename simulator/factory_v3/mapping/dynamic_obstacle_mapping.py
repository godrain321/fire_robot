"""Sensor-derived dynamic obstacle mapping for factory_v3.

The mapper consumes obstacle-occluded thermal rays, not the simulator's
Ground Truth obstacle list.  Ray endpoints are world coordinates and are
converted to planner ``(col, row)`` cells through ``MapMetadata``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from world.entities import (
    DynamicObstacle, DynamicObstacleShape, DynamicObstacleStatus,
)


@dataclass(frozen=True)
class DynamicObstacleMappingConfig:
    enabled: bool = True
    minimum_confirmation_observations: int = 2
    confirmation_timeout_s: float = 1.0
    duplicate_merge_distance_m: float = 0.3
    obstacle_diameter_m: float = 0.2
    obstacle_inflation_radius_m: float = 0.45
    stale_obstacle_timeout_s: float = 10.0
    minimum_confidence: float = 0.6
    ignored_fds_obstacle_ids: tuple[str, ...] = ()
    ignored_fds_mesh_xy_tolerance_m: float = 0.0

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown dynamic_obstacle_mapping settings: {sorted(unknown)}")
        if "ignored_fds_obstacle_ids" in values:
            raw = values["ignored_fds_obstacle_ids"]
            if not isinstance(raw, (list, tuple)):
                raise TypeError("ignored_fds_obstacle_ids must be a list")
            values["ignored_fds_obstacle_ids"] = tuple(raw)
        return cls(**values)

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be bool")
        if self.minimum_confirmation_observations < 1:
            raise ValueError("minimum_confirmation_observations must be at least one")
        for name in (
            "confirmation_timeout_s", "duplicate_merge_distance_m",
            "obstacle_diameter_m", "obstacle_inflation_radius_m",
            "stale_obstacle_timeout_s", "ignored_fds_mesh_xy_tolerance_m",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.obstacle_diameter_m <= 0.0:
            raise ValueError("obstacle_diameter_m must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0,1]")
        if any(
            not isinstance(item, str) or not item
            for item in self.ignored_fds_obstacle_ids
        ):
            raise ValueError(
                "ignored_fds_obstacle_ids must contain non-empty strings"
            )
        if len(set(self.ignored_fds_obstacle_ids)) != len(
            self.ignored_fds_obstacle_ids
        ):
            raise ValueError("ignored_fds_obstacle_ids must not contain duplicates")


@dataclass
class _ObservationTrack:
    position_world: tuple[float, float]
    first_seen_at: float
    last_seen_at: float
    observation_count: int
    source: str
    last_observation_id: str
    obstacle_id: str | None = None


@dataclass(frozen=True)
class DynamicObstacleMappingUpdate:
    observed_positions_world: tuple[tuple[float, float], ...]
    confirmed_obstacle_ids: tuple[str, ...]
    updated_obstacle_ids: tuple[str, ...]
    rejected_known_static_count: int
    map_changed: bool


class DynamicObstacleMapper:
    def __init__(
        self, metadata, static_obstacle_map, config=None,
        *, ignored_fds_bounds_world=(),
    ):
        import numpy as np

        self.metadata = metadata
        self.config = config or DynamicObstacleMappingConfig()
        self.static_obstacle_map = np.asarray(static_obstacle_map, dtype=bool).copy()
        if self.static_obstacle_map.shape != (metadata.height, metadata.width):
            raise ValueError("static obstacle map shape mismatch")
        self._tracks: list[_ObservationTrack] = []
        self._next_id = 1
        self._processed_observation_ids: set[str] = set()
        self.ignored_fds_bounds_world = tuple(
            tuple(map(float, bounds)) for bounds in ignored_fds_bounds_world
        )
        if any(len(bounds) != 6 for bounds in self.ignored_fds_bounds_world):
            raise ValueError("ignored FDS bounds must contain six XB values")

    def process_thermal_rays(
        self, observation_id: str, ray_observations, *, simulation_time: float,
        world_state,
    ) -> DynamicObstacleMappingUpdate:
        if observation_id in self._processed_observation_ids:
            raise ValueError(f"duplicate obstacle observation ID: {observation_id}")
        now = float(simulation_time)
        if not math.isfinite(now):
            raise ValueError("simulation_time must be finite")
        self._processed_observation_ids.add(observation_id)
        endpoints = []
        rejected_static = 0
        for ray in ray_observations:
            if not ray.occluded or not ray.ray_cells:
                continue
            sample = ray.ray_cells[-1]
            point = (float(sample.world_position[0]), float(sample.world_position[1]))
            if not all(math.isfinite(value) for value in point):
                continue
            if not self.metadata.is_world_position_in_bounds(*point):
                continue
            if self._matches_ignored_fds_mesh(sample.world_position):
                continue
            col, row = self.metadata.world_to_grid(*point)
            if self.static_obstacle_map[row, col]:
                rejected_static += 1
                continue
            if not any(
                math.dist(point, existing) <= self.config.duplicate_merge_distance_m
                for existing in endpoints
            ):
                endpoints.append(point)

        confirmed = []
        updated = []
        changed = False
        for point in endpoints:
            track = self._nearest_track(point)
            if track is None or now - track.last_seen_at > self.config.confirmation_timeout_s:
                track = _ObservationTrack(point, now, now, 1, "thermal_occlusion", observation_id)
                self._tracks.append(track)
            elif track.last_observation_id != observation_id:
                count = track.observation_count + 1
                averaged_position = (
                    (track.position_world[0] * track.observation_count + point[0]) / count,
                    (track.position_world[1] * track.observation_count + point[1]) / count,
                )
                # Two individually valid endpoints can straddle a concave wall
                # boundary and produce an average inside known occupancy.  Keep
                # the last known-free anchor unless the averaged position is
                # also a valid free cell.
                if self._is_known_free(averaged_position):
                    track.position_world = averaged_position
                track.observation_count = count
                track.last_seen_at = now
                track.last_observation_id = observation_id
            if (
                track.obstacle_id is None
                and track.observation_count >= self.config.minimum_confirmation_observations
            ):
                obstacle_id = f"perceived_obstacle_{self._next_id:04d}"
                self._next_id += 1
                confidence = max(
                    self.config.minimum_confidence,
                    min(1.0, track.observation_count /
                        self.config.minimum_confirmation_observations),
                )
                obstacle = DynamicObstacle(
                    obstacle_id=obstacle_id,
                    position_world=track.position_world,
                    shape=DynamicObstacleShape.CIRCLE,
                    size_m=(self.config.obstacle_diameter_m,
                            self.config.obstacle_diameter_m),
                    status=DynamicObstacleStatus.ACTIVE,
                    first_seen_at=track.first_seen_at,
                    last_seen_at=now,
                    source=track.source,
                    confidence=confidence,
                    observation_count=track.observation_count,
                    confirmed=True,
                )
                world_state.add_dynamic_obstacle(obstacle, sim_time=now)
                track.obstacle_id = obstacle_id
                confirmed.append(obstacle_id)
                changed = True
            elif track.obstacle_id is not None:
                obstacle = world_state.get_dynamic_obstacle(track.obstacle_id)
                position_changed = math.dist(
                    obstacle.position_world, track.position_world
                ) > self.metadata.resolution_m * 0.5
                world_state.update_dynamic_obstacle(
                    track.obstacle_id,
                    position_world=(track.position_world if position_changed else None),
                    confidence=max(obstacle.confidence, self.config.minimum_confidence),
                    observation_count=track.observation_count,
                    sim_time=now,
                )
                updated.append(track.obstacle_id)
                changed = changed or position_changed
        return DynamicObstacleMappingUpdate(
            tuple(endpoints), tuple(confirmed), tuple(updated), rejected_static,
            changed,
        )

    def _matches_ignored_fds_mesh(self, world_position) -> bool:
        """Match a configured door mesh plus ray-grid XY quantization margin."""
        x, y = float(world_position[0]), float(world_position[1])
        z = float(world_position[2]) if len(world_position) >= 3 else 0.0
        epsilon = 1e-9
        xy_margin = float(self.config.ignored_fds_mesh_xy_tolerance_m)
        return any(
            x1 - xy_margin - epsilon <= x <= x2 + xy_margin + epsilon
            and y1 - xy_margin - epsilon <= y <= y2 + xy_margin + epsilon
            and z1 - epsilon <= z <= z2 + epsilon
            for x1, x2, y1, y2, z1, z2 in self.ignored_fds_bounds_world
        )

    def _nearest_track(self, point):
        candidates = [
            (math.dist(point, track.position_world), track)
            for track in self._tracks
            if math.dist(point, track.position_world)
            <= self.config.duplicate_merge_distance_m
        ]
        return min(candidates, default=(None, None), key=lambda item: item[0])[1]

    def _is_known_free(self, point):
        if not self.metadata.is_world_position_in_bounds(*point):
            return False
        col, row = self.metadata.world_to_grid(*point)
        return not bool(self.static_obstacle_map[row, col])
