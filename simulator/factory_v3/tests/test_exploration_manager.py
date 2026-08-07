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
from mission.mission_manager import MissionEvent, MissionManager, MissionState
from planner.evacuation_planner import EvacuationPlanner
from planner.exit_evaluator import ExitEvaluationConfig, ExitEvaluator
from world import Exit, ExitStatus, ExitVisitStatus, MapMetadata, WorldState


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
    world.record_exit_check(
        "SHORT_ASTAR", ExitStatus.USABLE, sim_time=1,
        costmap_revision=1, reason="confirmed",
    )
    plans = [manager.plan_next_exit(
        world, (3, 3), cost_map=costs, costmap_revision=1,
        created_at=0, phase=ExplorationPhase.INITIAL,
    ) for _ in range(3)]
    assert all(item.target_exit_id == "NEAR_EUCLIDEAN" for item in plans)


def test_safety_status_and_direct_visit_status_are_independent():
    world, _, _ = make_system()
    world.update_exit_status("SHORT_ASTAR", ExitStatus.USABLE)
    assert world.exit_visit_status["SHORT_ASTAR"] is ExitVisitStatus.UNCHECKED
    world.record_exit_check(
        "SHORT_ASTAR", ExitStatus.USABLE, sim_time=2,
        costmap_revision=3, reason="direct check",
    )
    assert world.exit_visit_status["SHORT_ASTAR"] is ExitVisitStatus.CHECKED
    assert world.get_exit("SHORT_ASTAR").status is ExitStatus.USABLE


def test_unreachable_is_revision_scoped_and_retried_after_costmap_change():
    world, manager, costs = make_system()
    blocked = np.full_like(costs, np.inf)
    blocked[3, 3] = 1.0
    failed = manager.plan_next_exit(
        world, (3, 3), cost_map=blocked, costmap_revision=4,
        created_at=1, phase=ExplorationPhase.INITIAL,
    )
    assert not failed.success
    assert failed.failure_reason == "no_safe_exit"
    assert all(
        value["costmap_revision"] == 4
        for value in world.exit_reachability_by_revision.values()
    )
    retried = manager.plan_next_exit(
        world, (3, 3), cost_map=costs, costmap_revision=5,
        created_at=2, phase=ExplorationPhase.REPLAN_AFTER_MAP_CHANGE,
    )
    assert retried.success
    assert world.exit_visit_status[retried.target_exit_id] is ExitVisitStatus.UNCHECKED


def test_victim_interruption_keeps_target_path_and_current_pose():
    world, manager, costs = make_system()
    plan = manager.plan_next_exit(
        world, (3, 3), cost_map=costs, costmap_revision=6,
        created_at=3, phase=ExplorationPhase.INITIAL,
    )
    world.record_exploration_plan(plan, yaw_rad=0.25)
    record = world.record_exploration_interruption(
        sim_time=4, reason="victim_detected", victim_id="V1",
        robot_pose_world=(3.5, 3.0, 0.25),
        target_exit_id=plan.target_exit_id,
        active_path_grid=plan.path_grid[1:], costmap_revision=6,
    )
    assert record.robot_pose_world == (3.5, 3.0, 0.25)
    assert record.target_exit_id == plan.target_exit_id
    assert record.resume_required
    json.dumps(world.to_dict())


def test_exploration_mission_stall_retry_and_completion_states():
    mission = MissionManager()
    mission.handle_event(MissionEvent.EXPLORATION_STALLED, reason="no path")
    assert mission.current_state is MissionState.EXPLORATION_STALLED
    mission.handle_event(MissionEvent.RETRY_REQUESTED)
    assert mission.current_state is MissionState.SEARCH_EXITS
    mission.handle_event(MissionEvent.EXPLORATION_COMPLETED)
    assert mission.current_state is MissionState.EXPLORATION_COMPLETE


def test_three_exits_are_reselected_from_each_actual_arrival_pose():
    metadata = MapMetadata(0, 8, 0, 8, 1, 9, 9, (0, 0))
    world = WorldState(metadata, np.zeros((9, 9), dtype=bool))
    for exit_id, position in (
        ("A", (1, 1)), ("B", (7, 1)), ("C", (7, 7))
    ):
        world.add_exit(Exit(exit_id, position, position))
    evaluator = ExitEvaluator(
        metadata, ExitEvaluationConfig(), temperature_blocked_c=60,
        co_blocked_ppm=1600, base_cost=1,
    )
    manager = ExplorationManager(EvacuationPlanner(evaluator), ExplorationConfig())
    costs = np.ones((9, 9), dtype=float)
    current = (4, 4)
    order = []
    starts = []
    for revision in (1, 2, 3):
        plan = manager.plan_next_exit(
            world, current, cost_map=costs, costmap_revision=revision,
            created_at=revision, phase=ExplorationPhase.INITIAL,
        )
        assert plan.success
        starts.append(plan.start_position_world)
        order.append(plan.target_exit_id)
        world.record_exit_check(
            plan.target_exit_id, ExitStatus.USABLE, sim_time=revision,
            costmap_revision=revision, reason="checked",
        )
        current = plan.target_position_world
    assert len(set(order)) == 3
    assert starts[0] == (4.0, 4.0)
    assert starts[1] != starts[0]
    completed = manager.plan_next_exit(
        world, current, cost_map=costs, costmap_revision=4,
        created_at=4, phase=ExplorationPhase.INITIAL,
    )
    assert not completed.success
    assert completed.failure_reason == "no_unchecked_exits"


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
