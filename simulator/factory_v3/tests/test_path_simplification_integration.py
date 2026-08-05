import json
from pathlib import Path

import numpy as np
import yaml

from mapping.partial_costmap import PartialCostmapConfig
from navigation.path_simplifier import PathSimplificationConfig, SafePathSimplifier
from navigation.return_path_planner import ReturnPathConfig, ReturnPathPlanner
from navigation.travel_history import TravelHistory
from planner.a_star import weighted_a_star
from robot.path_follower import ReplannablePathFollower, RobotState
from world import (
    DynamicObstacle, DynamicObstacleStatus, MapMetadata, WorldState,
)


BASE = Path(__file__).resolve().parents[1]


class GridAdapter:
    def __init__(self, metadata, blocked):
        self.metadata = metadata
        self.blocked = blocked
        self.resolution = metadata.resolution_m
    def grid_to_world(self, col, row):
        return self.metadata.grid_to_world(col, row)
    def world_to_grid(self, x, y):
        return self.metadata.world_to_grid(x, y)
    def in_bounds(self, node):
        return self.metadata.is_grid_position_in_bounds(*node)
    def is_blocked(self, node):
        return not self.in_bounds(node) or self.blocked[node[1], node[0]]


def setup_world():
    metadata = MapMetadata(0, 5, 0, 5, 1, 6, 6, (0, 0))
    world = WorldState(metadata, np.zeros((6, 6), dtype=bool))
    world.estimated_fire_map.replace_layers(
        np.full((6, 6), 20.0), np.zeros((6, 6)),
        np.ones((6, 6), dtype=bool), np.zeros((6, 6)),
    )
    return world, SafePathSimplifier(metadata), np.ones((6, 6))


def test_a_star_remains_cellwise_then_result_is_simplified():
    world, simplifier, costs = setup_world()
    costs[1, 1] = np.inf
    astar = weighted_a_star(costs, (0, 0), (3, 2))
    assert len(astar.path) > 2
    result = simplifier.simplify(
        astar.path, costmap=costs,
        static_obstacle_map=world.static_obstacle_map,
        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
        estimated_fire_map=world.estimated_fire_map,
    )
    assert result.success
    assert result.original_path_grid == tuple(astar.path)
    assert result.simplified_point_count <= result.original_point_count


def test_world_state_stores_structured_simplification_result():
    world, simplifier, costs = setup_world()
    result = simplifier.simplify(
        [(0, 0), (1, 0), (2, 0)], costmap=costs,
        static_obstacle_map=world.static_obstacle_map,
        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
        estimated_fire_map=world.estimated_fire_map,
    )
    world.record_path_simplification(result)
    assert world.active_path_simplification is result
    assert len(world.path_simplification_history) == 1
    json.dumps(world.to_dict())


def test_follower_receives_validated_world_waypoints():
    world, simplifier, costs = setup_world()
    result = simplifier.simplify(
        [(0, 0), (1, 0), (2, 0)], costmap=costs,
        static_obstacle_map=world.static_obstacle_map,
        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
        estimated_fire_map=world.estimated_fire_map,
        start_world=(0.1, 0.0), goal_world=(2.1, 0.0),
    )
    follower = ReplannablePathFollower(
        GridAdapter(world.map_metadata, world.static_obstacle_map),
        PartialCostmapConfig(robot_speed=1, simulation_dt=0.1),
    )
    follower.set_path(
        result.simplified_path_grid, world_path=result.waypoints_world
    )
    assert follower.world_path == list(result.waypoints_world)


def test_costmap_change_invalidates_previous_simplified_path():
    world, simplifier, costs = setup_world()
    result = simplifier.simplify(
        [(0, 0), (1, 0), (2, 0)], costmap=costs,
        static_obstacle_map=world.static_obstacle_map,
        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
        estimated_fire_map=world.estimated_fire_map, costmap_revision=1,
    )
    costs[0, 1] = np.inf
    validation = simplifier.validate_path(
        result.simplified_path_grid, costmap=costs,
        static_obstacle_map=world.static_obstacle_map,
        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
        estimated_fire_map=world.estimated_fire_map,
    )
    assert result.used_costmap_revision == 1
    assert not validation.safe


def test_cleared_dynamic_obstacle_does_not_remain_blocked():
    world, simplifier, costs = setup_world()
    world.add_dynamic_obstacle(DynamicObstacle(
        "cart", (1, 0), status=DynamicObstacleStatus.ACTIVE
    ))
    blocked = simplifier.validate_path(
        [(0, 0), (2, 0)], costmap=costs,
        static_obstacle_map=world.static_obstacle_map,
        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
        estimated_fire_map=world.estimated_fire_map,
    )
    world.clear_dynamic_obstacle("cart", sim_time=1)
    cleared = simplifier.validate_path(
        [(0, 0), (2, 0)], costmap=costs,
        static_obstacle_map=world.static_obstacle_map,
        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
        estimated_fire_map=world.estimated_fire_map,
    )
    assert not blocked.safe and cleared.safe


def test_travel_history_reverse_path_is_simplified_without_mutating_history():
    world, simplifier, costs = setup_world()
    history = TravelHistory(world.map_metadata)
    for time, point in enumerate(((0, 0), (1, 0), (2, 0), (2, 1))):
        history.record_position(point, recorded_at=time)
    before = history.get_points()
    planner = ReturnPathPlanner(
        world.map_metadata, ReturnPathConfig(max_observation_age_s=None)
    )
    plan = planner.create_plan(
        history, (2, 1), cost_map=costs,
        static_obstacle_map=world.static_obstacle_map,
        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
        estimated_fire_map=world.estimated_fire_map, created_at=4,
    )
    result = simplifier.simplify(
        plan.path_grid, costmap=costs,
        static_obstacle_map=world.static_obstacle_map,
        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
        estimated_fire_map=world.estimated_fire_map,
    )
    assert result.success
    assert history.get_points() == before


def test_yaml_path_simplification_configuration_loads():
    scenario = yaml.safe_load((BASE / "config/evacuation.yaml").read_text())
    config = PathSimplificationConfig.from_mapping(scenario["path_simplification"])
    assert config.line_traversal == "supercover"
    assert config.use_inflated_costmap
