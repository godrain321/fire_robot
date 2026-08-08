from pathlib import Path

import numpy as np
import pytest
import yaml

from mapping.partial_costmap import PartialCostmapConfig
from mission.mission_manager import MissionEvent, MissionManager, MissionState
from planner.evacuation_planner import EvacuationPlanner, ExitSelectionConfig
from planner.exit_evaluator import ExitEvaluationConfig, ExitEvaluator
from world import Exit, MapMetadata, Victim, VictimStatus, WorldState

BASE = Path(__file__).resolve().parents[1]


def setup_world():
    metadata = MapMetadata(0, 5, 0, 5, 1, 6, 6, (0, 0))
    world = WorldState(metadata, np.zeros((6, 6), dtype=bool))
    world.add_exit(Exit("E1", (5, 0), (5, 0)))
    world.add_exit(Exit("E2", (5, 5), (5, 5)))
    world.add_victim(Victim("V1", (0, 0)))
    shape = (6, 6)
    world.estimated_fire_map.replace_layers(
        np.full(shape, 20.0), np.zeros(shape), np.ones(shape, dtype=bool),
        np.zeros(shape),
    )
    evaluator = ExitEvaluator(
        metadata, ExitEvaluationConfig(), temperature_blocked_c=60,
        co_blocked_ppm=1600, base_cost=1,
    )
    return world, EvacuationPlanner(evaluator), np.ones(shape)


def ready_manager():
    manager = MissionManager()
    manager.handle_event(MissionEvent.VICTIM_DETECTED, victim_id="V1", victim_position=(0, 0))
    manager.handle_event(MissionEvent.VICTIM_REACHED)
    manager.handle_event(MissionEvent.ANNOUNCEMENT_FINISHED)
    manager.handle_event(MissionEvent.VICTIM_STARTED_MOVING)
    return manager


def test_world_and_mission_success_integration():
    world, planner, costs = setup_world()
    manager = ready_manager()
    manager.handle_event(MissionEvent.EXIT_EVALUATION_REQUESTED)
    plan = planner.plan(
        world.exits.values(), world.get_victim("V1").position_world,
        cost_map=costs, static_obstacle_map=world.static_obstacle_map,
        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
        estimated_fire_map=world.estimated_fire_map, created_at=1,
    )
    world.record_exit_evaluations(plan)
    world.record_exit_evaluations(plan)
    manager.handle_event(
        MissionEvent.SAFE_EXIT_SELECTED, exit_id=plan.selected_exit_id,
        exit_position=plan.selected_approach_position_world,
        selection_reason=plan.selection_reason,
    )
    manager.handle_event(
        MissionEvent.EVACUATION_PLAN_CREATED, path=plan.path_grid,
        exit_id=plan.selected_exit_id,
    )
    assert manager.current_state is MissionState.ESCORT_VICTIM
    assert manager.selected_exit_reason == plan.selection_reason
    assert world.active_evacuation_plan is plan
    assert len(world.latest_exit_evaluations) == 2
    assert len(world.exit_evaluation_history) == 2
    assert world.get_exit(plan.selected_exit_id).last_checked_at == 1


def test_world_and_mission_failure_integration():
    world, planner, costs = setup_world()
    costs[:] = np.inf
    manager = ready_manager()
    manager.handle_event(MissionEvent.EXIT_EVALUATION_REQUESTED)
    plan = planner.plan(
        world.exits.values(), (0, 0), cost_map=costs,
        static_obstacle_map=world.static_obstacle_map,
        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
        estimated_fire_map=world.estimated_fire_map, created_at=1,
    )
    world.record_exit_evaluations(plan)
    manager.handle_event(MissionEvent.NO_SAFE_EXIT_FOUND, reason="all rejected")
    assert manager.current_state is MissionState.NO_SAFE_EXIT
    assert world.active_evacuation_plan is None
    assert plan.selected_exit_id is None


def test_factory_config_and_invalid_settings():
    scenario = yaml.safe_load((BASE / "config/evacuation.yaml").read_text())
    evaluation = ExitEvaluationConfig.from_mapping(scenario["exit_evaluation"])
    selection = ExitSelectionConfig.from_mapping(scenario["exit_selection"])
    assert evaluation.usable_confirmation_max_unknown_ratio == 0.5
    assert evaluation.dangerous_accumulated_risk_cost == 150.0
    assert evaluation.dangerous_average_risk_cost == 10.0
    assert evaluation.dangerous_max_cell_risk_cost == 20.0
    assert selection.primary_key == "path_length_m"
    with pytest.raises(ValueError):
        ExitEvaluationConfig.from_mapping({"unknown_cell_policy": "safe"})
    with pytest.raises(ValueError):
        ExitSelectionConfig.from_mapping({"secondary_key": "exit_id"})


def test_existing_partial_costmap_thresholds_are_reused():
    config = PartialCostmapConfig()
    assert config.temperature_blocked == 60
    assert config.co_blocked == 1600
