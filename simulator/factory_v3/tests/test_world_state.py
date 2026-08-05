import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mapping.fire_costmap import load_factory_geometry
from mapping.grid_map import GridMap
from mapping.partial_costmap import PartialCostmapConfig
from mission.mission_manager import MissionEvent, MissionManager
from world import (
    DynamicObstacle, DynamicObstacleShape, DynamicObstacleStatus,
    Exit, ExitStatus, MapMetadata, Victim, VictimStatus, WorldState,
)

BASE = Path(__file__).resolve().parents[1]


def make_world():
    metadata = MapMetadata(0, 4, 0, 3, 1, 5, 4, (0, 0))
    static = np.zeros((4, 5), dtype=bool)
    static[2, 2] = True
    return WorldState(metadata, static)


def test_registry_duplicates_missing_ids_and_entity_validation():
    world = make_world()
    world.add_exit(Exit("E1", (0, 1), (1, 1)))
    with pytest.raises(ValueError):
        world.add_exit(Exit("E1", (4, 1), (3, 1)))
    with pytest.raises(KeyError):
        world.get_exit("missing")
    with pytest.raises(ValueError):
        world.add_victim(Victim("blocked", (2, 2)))
    with pytest.raises(ValueError):
        world.add_victim(Victim("outside", (5, 1)))
    with pytest.raises(ValueError):
        world.add_exit(Exit("bad-approach", (0, 1), (2, 2)))


def test_dynamic_layer_is_separate_and_out_of_bounds_not_clipped():
    world = make_world()
    obstacle = DynamicObstacle(
        "cart", (1, 1), DynamicObstacleShape.RECTANGLE, (1, 1),
        DynamicObstacleStatus.ACTIVE,
    )
    world.add_dynamic_obstacle(obstacle)
    assert world.dynamic_obstacle_mask().any()
    assert not world.static_obstacle_map[1, 1]
    world.clear_dynamic_obstacle("cart")
    assert not world.dynamic_obstacle_mask().any()
    with pytest.raises(ValueError):
        world.add_dynamic_obstacle(DynamicObstacle(
            "edge", (0, 0), DynamicObstacleShape.CIRCLE, (1, 1)
        ))


def test_point_obstacle_uses_nearest_grid_cell():
    world = make_world()
    world.add_dynamic_obstacle(DynamicObstacle(
        "point", (1.2, 0.8), status=DynamicObstacleStatus.ACTIVE
    ))
    col, row = world.map_metadata.world_to_grid(1.2, 0.8)
    assert world.dynamic_obstacle_mask()[row, col]


def test_queries_validation_and_json_serialization():
    world = make_world()
    world.add_exit(Exit("E1", (0, 1), (1, 1)))
    world.add_victim(Victim("V1", (1, 2)))
    world.update_exit_status("E1", ExitStatus.USABLE)
    world.update_victim_status("V1", VictimStatus.DETECTED)
    assert [item.exit_id for item in world.get_usable_exits()] == ["E1"]
    assert [item.victim_id for item in world.get_unrescued_victims()] == ["V1"]
    world.validate_all_entities()
    json.dumps(world.to_dict())


def test_world_state_and_mission_manager_minimal_link():
    world = make_world()
    world.add_victim(Victim("V1", (1, 1)))
    manager = MissionManager()
    world.update_victim_status("V1", VictimStatus.DETECTED, sim_time=2)
    victim = world.get_victim("V1")
    manager.handle_event(
        MissionEvent.VICTIM_DETECTED,
        victim_id=victim.victim_id,
        victim_position=victim.position_world,
        sim_time=2,
    )
    assert manager.selected_victim_id == victim.victim_id


def test_factory_v3_config_loads_and_bad_enum_is_rejected():
    scenario = yaml.safe_load((BASE / "config/evacuation.yaml").read_text())
    mesh, obstacles, holes = load_factory_geometry(BASE / "factory_v3.fds")
    config = PartialCostmapConfig(
        grid_resolution=scenario["planner"]["grid_resolution_m"],
        inflation_radius=scenario["planner"]["inflation_radius_m"],
    )
    grid = GridMap(mesh, obstacles, holes, config.grid_resolution, config.inflation_radius)
    world = WorldState.from_scenario(scenario, grid, config)
    assert set(world.exits) == {"EXIT1", "EXIT2", "EXIT3"}
    assert set(world.victims) == {"victim_1"}
    assert (world.map_metadata.width, world.map_metadata.height) == (142, 113)
    bad = dict(scenario)
    bad["exits"] = [dict(scenario["exits"][0], initial_status="maybe")]
    with pytest.raises(ValueError, match="invalid exit status"):
        WorldState.from_scenario(bad, grid, config)
