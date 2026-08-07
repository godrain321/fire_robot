import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from mapping.fire_costmap import load_factory_geometry
from mapping.grid_map import GridMap
from navigation.exploration_manager import (
    ExplorationConfig, ExplorationManager, ExplorationPhase,
)
from planner.evacuation_planner import EvacuationPlanner
from planner.exit_evaluator import ExitEvaluationConfig, ExitEvaluator
from world import Exit, ExitStatus, MapMetadata, WorldState


BASE = Path(__file__).resolve().parents[1]


def make_system():
    metadata = MapMetadata(0, 6, 0, 6, 1, 7, 7, (0, 0))
    static = np.zeros((7, 7), dtype=bool)
    static[1:6, 4] = True
    world = WorldState(metadata, static)
    world.add_exit(Exit("NEAR_EUCLIDEAN", (5, 3), (5, 3)))
    world.add_exit(Exit("SHORT_ASTAR", (3, 6), (3, 6)))
    evaluator = ExitEvaluator(
        metadata, ExitEvaluationConfig(), temperature_blocked_c=60,
        co_blocked_ppm=1600, base_cost=1,
    )
    manager = ExplorationManager(
        EvacuationPlanner(evaluator), ExplorationConfig()
    )
    return world, manager, np.ones((7, 7), dtype=float)


def test_initial_pose_comes_from_existing_yaml_and_is_not_an_entrance():
    scenario = yaml.safe_load((BASE / "config/evacuation.yaml").read_text())
    assert scenario["robot_start"] == {
        "x": 13.0, "y": 16.0, "yaw_deg": -85.70
    }
    assert "mission_entry_id" not in scenario
    assert scenario["exploration"]["teleport_to_entrance_on_start"] is False


def test_factory_initial_world_pose_grid_and_yaw_are_preserved():
    scenario = yaml.safe_load((BASE / "config/evacuation.yaml").read_text())
    mesh, obstacles, holes = load_factory_geometry(BASE / scenario["fds_file"])
    grid = GridMap(mesh, obstacles, holes, 0.2, 0.45)
    start = scenario["robot_start"]
    col, row = grid.world_to_grid(start["x"], start["y"])
    assert grid.in_bounds((col, row))
    assert not grid.is_blocked((col, row))
    state_yaw = math.radians(start["yaw_deg"])
    assert math.degrees(state_yaw) == pytest.approx(-85.70)


def test_unknown_exit_selection_uses_astar_length_not_euclidean_distance():
    world, manager, costs = make_system()
    plan = manager.plan_next_exit(
        world, (3, 3), cost_map=costs, costmap_revision=1,
        created_at=0, phase=ExplorationPhase.INITIAL,
    )
    assert plan.success
    assert plan.target_exit_id == "SHORT_ASTAR"
    lengths = dict(plan.exit_path_lengths_m)
    assert lengths["SHORT_ASTAR"] < lengths["NEAR_EUCLIDEAN"]


def test_resume_replans_from_actual_completion_pose_and_not_old_start():
    world, manager, costs = make_system()
    first = manager.plan_next_exit(
        world, (3, 3), cost_map=costs, costmap_revision=1,
        created_at=0, phase=ExplorationPhase.INITIAL,
    )
    world.update_exit_status(first.target_exit_id, ExitStatus.USABLE)
    completion_pose = (6, 6)
    resumed = manager.plan_next_exit(
        world, completion_pose, cost_map=costs, costmap_revision=2,
        created_at=10, phase=ExplorationPhase.RESUMED_AFTER_EVACUATION,
    )
    assert resumed.success
    assert resumed.start_position_world == completion_pose
    assert resumed.path_grid[0] == world.map_metadata.world_to_grid(*completion_pose)
    world.record_exploration_plan(resumed, yaw_rad=0.3)
    assert world.exploration_resume_pose_world == (6.0, 6.0, 0.3)
    assert world.initial_robot_pose_world is None
    json.dumps(world.to_dict())


def test_checked_exits_are_not_reselected_and_result_is_deterministic():
    world, manager, costs = make_system()
    world.update_exit_status("SHORT_ASTAR", ExitStatus.USABLE)
    plans = [manager.plan_next_exit(
        world, (3, 3), cost_map=costs, costmap_revision=1,
        created_at=0, phase=ExplorationPhase.INITIAL,
    ) for _ in range(3)]
    assert all(item.target_exit_id == "NEAR_EUCLIDEAN" for item in plans)


def test_world_keeps_initial_exploration_and_entrance_fields_separate():
    world, manager, costs = make_system()
    world.set_initial_robot_pose((3, 3), 0.5)
    world.configure_exploration_return(enabled=False, entrance_exit_id=None)
    plan = manager.plan_next_exit(
        world, (3, 3), cost_map=costs, costmap_revision=4,
        created_at=2, phase=ExplorationPhase.INITIAL,
    )
    world.record_exploration_plan(plan, yaw_rad=0.5)
    assert world.initial_robot_pose_world == (3.0, 3.0, 0.5)
    assert world.exploration_entry_pose_world == (3.0, 3.0, 0.5)
    assert world.mission_entry_position_world is None
    assert world.robot_grid_position == (3, 3)


def test_exploration_config_rejects_teleport_and_invalid_boolean():
    with pytest.raises(ValueError):
        ExplorationConfig(teleport_to_entrance_on_start=True)
    with pytest.raises(ValueError):
        ExplorationConfig(use_current_robot_pose_as_start=False)
    with pytest.raises(ValueError):
        ExplorationConfig(return_to_entrance_when_complete=True)
    with pytest.raises(TypeError):
        ExplorationConfig(enabled="yes")
