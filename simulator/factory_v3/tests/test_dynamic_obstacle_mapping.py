import numpy as np
import pytest

from mapping.dynamic_obstacle_mapping import (
    DynamicObstacleMapper, DynamicObstacleMappingConfig,
)
from mapping.grid_map import GridMap
from mapping.partial_costmap import PartialCostmapConfig, PartialFireCostmap
from sensors.thermal_camera import ThermalRayObservation, ThermalRaySample
from world.fire_maps import MapMetadata
from world.world_state import WorldState


def meta():
    return MapMetadata(0, 4, 0, 4, 1, 5, 5, (0, 0))


def ray(endpoint=(2, 2), *, occluded=True):
    sample = ThermalRaySample(
        (endpoint[0], endpoint[1], 0),
        (float(endpoint[0]), float(endpoint[1]), 0.3), 2.0, 25.0,
    )
    return ThermalRayObservation(
        0, 0, 25.0, (sample,), sample.world_position,
        sample.grid_position, 2.0, True, occluded,
    )


def world(static=None):
    static = np.zeros((5, 5), bool) if static is None else static
    return WorldState(meta(), static)


def test_obstacle_is_hidden_until_confirmed_then_duplicate_observations_merge():
    state = world()
    mapper = DynamicObstacleMapper(
        meta(), state.known_occupancy_map,
        DynamicObstacleMappingConfig(minimum_confirmation_observations=2),
    )
    first = mapper.process_thermal_rays(
        "frame-1", (ray(), ray((2.1, 2))), simulation_time=0, world_state=state
    )
    assert not first.confirmed_obstacle_ids
    assert not state.dynamic_obstacles
    second = mapper.process_thermal_rays(
        "frame-2", (ray(),), simulation_time=0.25, world_state=state
    )
    assert len(second.confirmed_obstacle_ids) == 1
    assert len(state.dynamic_obstacles) == 1
    obstacle = state.get_dynamic_obstacle(second.confirmed_obstacle_ids[0])
    assert obstacle.confirmed
    assert obstacle.observation_count == 2
    environment_revision = state.environment_revision
    mapper.process_thermal_rays(
        "frame-3", (ray(),), simulation_time=0.5, world_state=state
    )
    assert len(state.dynamic_obstacles) == 1
    assert obstacle.observation_count == 3
    assert state.environment_revision == environment_revision


def test_known_static_occlusion_and_invalid_or_unoccluded_rays_are_not_mapped():
    static = np.zeros((5, 5), bool)
    static[2, 2] = True
    state = world(static)
    mapper = DynamicObstacleMapper(
        meta(), static,
        DynamicObstacleMappingConfig(minimum_confirmation_observations=1),
    )
    update = mapper.process_thermal_rays(
        "known", (ray(), ray((3, 3), occluded=False)),
        simulation_time=0, world_state=state,
    )
    assert update.rejected_known_static_count == 1
    assert not state.dynamic_obstacles
    with pytest.raises(ValueError, match="duplicate"):
        mapper.process_thermal_rays(
            "known", (), simulation_time=1, world_state=state
        )


def test_track_average_cannot_move_confirmed_obstacle_into_static_occupancy():
    metadata = MapMetadata(0, 1, 0, 1, 0.1, 11, 11, (0, 0))
    static = np.zeros((11, 11), bool)
    static[5, 5] = True
    state = WorldState(metadata, static)
    mapper = DynamicObstacleMapper(
        metadata, static,
        DynamicObstacleMappingConfig(
            minimum_confirmation_observations=2,
            duplicate_merge_distance_m=0.3,
        ),
    )

    mapper.process_thermal_rays(
        "left", (ray((0.39, 0.5)),), simulation_time=0, world_state=state,
    )
    update = mapper.process_thermal_rays(
        "right", (ray((0.61, 0.5)),), simulation_time=0.1, world_state=state,
    )

    obstacle = state.get_dynamic_obstacle(update.confirmed_obstacle_ids[0])
    col, row = metadata.world_to_grid(*obstacle.position_world)
    assert not static[row, col]
    assert obstacle.position_world == pytest.approx((0.39, 0.5))


def test_dynamic_layer_inflation_and_revision_change_only_when_mask_changes():
    grid = GridMap((0, 4, 0, 4, 0, 1), [], [], 1, 0)
    belief = PartialFireCostmap(
        grid, np.zeros((5, 5), bool),
        PartialCostmapConfig(grid_resolution=1),
    )
    raw = np.zeros((5, 5), bool)
    raw[2, 2] = True
    update = belief.update_dynamic_obstacles(raw, inflation_radius_m=1.0)
    assert update.changed_cells
    assert belief.dynamic_obstacle_map[2, 2]
    assert belief.dynamic_inflated_obstacle_map[2, 1]
    assert not belief.dynamic_obstacle_map[2, 1]
    assert np.isinf(belief.final_cost_map[2, 1])
    revision = belief.revision
    repeated = belief.update_dynamic_obstacles(raw.copy(), inflation_radius_m=1.0)
    assert not repeated.changed_cells
    assert belief.revision == revision


@pytest.mark.parametrize("values", [
    {"minimum_confirmation_observations": 0},
    {"duplicate_merge_distance_m": -1},
    {"obstacle_inflation_radius_m": -1},
    {"minimum_confidence": 2},
])
def test_invalid_mapping_config(values):
    with pytest.raises((ValueError, TypeError)):
        DynamicObstacleMappingConfig(**values)
