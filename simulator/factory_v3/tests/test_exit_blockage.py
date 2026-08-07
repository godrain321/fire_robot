import numpy as np
import pytest
from pathlib import Path
import yaml

from mapping.fire_costmap import load_factory_geometry, obstacles_for_initial_robot_map
from mapping.grid_map import GridMap
from mapping.partial_costmap import PartialCostmapConfig, PartialFireCostmap
from navigation.exit_blockage import ExitBlockageConfig, ExitBlockageEvaluator
from world.entities import (
    DynamicObstacle, DynamicObstacleShape, DynamicObstacleStatus, Exit,
    ExitStatus,
)
from world.fire_maps import MapMetadata
from world.world_state import WorldState


def setup():
    metadata = MapMetadata(0, 6, 0, 6, 1, 7, 7, (0, 0))
    exit_item = Exit("EXIT", (3, 4), (3, 2), ExitStatus.USABLE)
    costs = np.ones((7, 7), float)
    static = np.zeros((7, 7), bool)
    return metadata, exit_item, costs, static


def obstacle(count=2):
    return DynamicObstacle(
        "obs", (3, 3), DynamicObstacleShape.RECTANGLE, (3, 1),
        DynamicObstacleStatus.ACTIVE, source="thermal_occlusion",
        confidence=1, observation_count=count, confirmed=True,
    )


def test_exit_near_obstacle_is_not_blocked_when_goal_width_remains_open():
    metadata, item, costs, static = setup()
    dynamic = np.zeros_like(static)
    dynamic[3, 3] = True
    evaluator = ExitBlockageEvaluator(
        metadata, ExitBlockageConfig(required_clear_width_m=1)
    )
    result = evaluator.evaluate(
        item, (3, 0), cost_map=costs, static_obstacle_map=static,
        dynamic_inflated_map=dynamic, active_obstacles=(obstacle(),),
        evaluated_at=1, environment_revision=1,
    )
    assert not result.blocked_confirmed
    assert result.reachable_goal_exists


def test_full_exit_width_block_requires_confirmed_observations():
    metadata, item, costs, static = setup()
    dynamic = np.zeros_like(static)
    dynamic[3, 1:6] = True
    evaluator = ExitBlockageEvaluator(
        metadata,
        ExitBlockageConfig(required_clear_width_m=1, confirmation_observations=2),
    )
    weak = evaluator.evaluate(
        item, (3, 0), cost_map=costs, static_obstacle_map=static,
        dynamic_inflated_map=dynamic, active_obstacles=(obstacle(1),),
        evaluated_at=1, environment_revision=1,
    )
    confirmed = evaluator.evaluate(
        item, (3, 0), cost_map=costs, static_obstacle_map=static,
        dynamic_inflated_map=dynamic, active_obstacles=(obstacle(2),),
        evaluated_at=2, environment_revision=2,
    )
    assert weak.blocked_geometry and not weak.blocked_confirmed
    assert confirmed.blocked_confirmed
    assert not confirmed.reachable_goal_exists
    assert confirmed.blocking_cells_grid


def test_open_alternative_goal_prevents_center_cell_false_block():
    metadata, item, costs, static = setup()
    dynamic = np.zeros_like(static)
    dynamic[3, 3] = True
    evaluator = ExitBlockageEvaluator(
        metadata, ExitBlockageConfig(required_clear_width_m=1)
    )
    result = evaluator.evaluate(
        item, (3, 0), cost_map=costs, static_obstacle_map=static,
        dynamic_inflated_map=dynamic, active_obstacles=(obstacle(),),
        evaluated_at=1, environment_revision=1,
    )
    assert len(result.reachable_cells_grid) >= 1
    assert not result.blocked_confirmed


def test_factory_exit1_fallen_rack_geometry_blocks_full_approach():
    base = Path(__file__).resolve().parents[1]
    scenario = yaml.safe_load((base / "config/evacuation.yaml").read_text())
    mesh, obstacles, holes = load_factory_geometry(base / "factory_v3.fds")
    planner_obstacles = obstacles_for_initial_robot_map(obstacles, scenario)
    planner_grid = GridMap(mesh, planner_obstacles, holes, 0.2, 0.45)
    raw_grid = GridMap(
        mesh, planner_obstacles, holes, 0.2, 0,
        include_lower_obstacle_boundary=False,
    )
    config = PartialCostmapConfig(
        grid_resolution=0.2, inflation_radius=0.45,
    )
    state = WorldState.from_scenario(scenario, planner_grid, config)
    state.set_known_occupancy_map(raw_grid.occupancy)
    rack = DynamicObstacle(
        "rack", (9.3, 17.9), DynamicObstacleShape.RECTANGLE, (3.0, 0.2),
        DynamicObstacleStatus.ACTIVE, source="thermal_occlusion",
        confidence=1, observation_count=2, confirmed=True,
    )
    state.add_dynamic_obstacle(rack, sim_time=1)
    belief = PartialFireCostmap(planner_grid, planner_grid.occupancy, config)
    belief.update_dynamic_obstacles(
        state.dynamic_obstacle_mask(), inflation_radius_m=0.45
    )
    evaluator = ExitBlockageEvaluator(
        state.map_metadata,
        ExitBlockageConfig(
            approach_region_depth_m=1.5, required_clear_width_m=0.8,
            confirmation_observations=2, human_clearance_m=0.2,
        ),
    )
    result = evaluator.evaluate(
        state.get_exit("EXIT1"), (13, 16),
        cost_map=belief.final_cost_map,
        static_obstacle_map=state.known_occupancy_map,
        dynamic_inflated_map=belief.dynamic_inflated_obstacle_map,
        active_obstacles=state.get_active_dynamic_obstacles(),
        evaluated_at=1, environment_revision=state.environment_revision,
    )
    assert result.blocked_confirmed
    assert not result.reachable_goal_exists


@pytest.mark.parametrize("values", [
    {"required_clear_width_m": 0},
    {"approach_region_depth_m": -1},
    {"confirmation_observations": 0},
    {"release_observations": 0},
])
def test_invalid_exit_blockage_config(values):
    with pytest.raises((ValueError, TypeError)):
        ExitBlockageConfig(**values)
