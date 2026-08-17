import json

import numpy as np
import pytest

from navigation.evacuation_strategy_selector import (
    EvacuationRouteSelectionConfig, EvacuationStrategy,
    EvacuationStrategySelector, HazardKnowledgeConfig, HazardKnowledgeTracker,
    PathValidationConfig, ReplanningConfig,
)
from navigation.return_path_planner import ReturnPathConfig, ReturnPathPlanner
from navigation.travel_history import TravelHistory
from planner.evacuation_planner import EvacuationPlanner
from planner.exit_evaluator import ExitEvaluationConfig, ExitEvaluator
from world import Exit, MapMetadata, WorldState


def setup_system():
    metadata = MapMetadata(0, 5, 0, 5, 1, 6, 6, (0, 0))
    world = WorldState(metadata, np.zeros((6, 6), dtype=bool))
    world.add_exit(Exit("NEAR", (3, 0), (3, 0)))
    world.add_exit(Exit("FAR", (5, 5), (5, 5)))
    world.set_mission_entry("ENTRY", (0, 0))
    temp = np.full((6, 6), 20.0)
    co = np.zeros((6, 6))
    world.estimated_fire_map.replace_layers(
        temp, co, np.ones((6, 6), dtype=bool), np.full((6, 6), 5.0)
    )
    history = TravelHistory(metadata)
    for time, point in enumerate(((0, 0), (1, 0), (2, 0))):
        history.record_position(point, recorded_at=time)
    world.attach_travel_history(history)
    evaluator = ExitEvaluator(
        metadata, ExitEvaluationConfig(),
        temperature_blocked_c=60, co_blocked_ppm=1600, base_cost=1,
    )
    selector = EvacuationStrategySelector(
        metadata,
        HazardKnowledgeTracker(temperature_elevated_c=35, co_elevated_ppm=100),
        ReturnPathPlanner(metadata, ReturnPathConfig(max_observation_age_s=None)),
        EvacuationPlanner(evaluator),
    )
    return world, history, selector, np.ones((6, 6))


def test_no_fire_information_selects_nearest_reachable_exit_without_history_mutation():
    world, history, selector, costs = setup_system()
    before = history.get_points()
    decision = selector.select_initial_route(
        world_state=world, travel_history=history,
        start_position_world=(2, 0), victim_position_world=(2, 0),
        cost_map=costs, costmap_revision=3, created_at=5,
    )
    assert decision.success
    assert decision.strategy is EvacuationStrategy.NEAREST_REACHABLE_EXIT
    assert decision.target_exit_id == "NEAR"
    assert decision.path_grid[-1] == (3, 0)
    assert history.get_points() == before
    json.dumps(decision.to_dict())


def test_fire_information_rejects_dangerous_near_exit_and_selects_other():
    world, history, selector, costs = setup_system()
    world.estimated_fire_map.update_cell(3, 0, temperature_c=60, sim_time=5)
    decision = selector.select_initial_route(
        world_state=world, travel_history=history,
        start_position_world=(2, 0), victim_position_world=(2, 0),
        cost_map=costs, costmap_revision=4, created_at=5,
    )
    assert decision.success
    assert decision.strategy is EvacuationStrategy.SAFE_EXIT_PLANNING
    assert decision.target_exit_id == "FAR"
    assert len(decision.evacuation_plan.all_evaluations) == 2


def test_invalidated_history_replans_to_original_entry_from_current_position():
    world, _, selector, costs = setup_system()
    decision = selector.replan_after_return_invalidated(
        world_state=world, current_position_world=(2, 0), cost_map=costs,
        costmap_revision=7, created_at=7,
    )
    assert decision.success
    assert decision.strategy is EvacuationStrategy.REPLAN_TO_ENTRANCE
    assert decision.path_grid[0] == (2, 0)
    assert decision.path_grid[-1] == (0, 0)


def test_all_routes_failed_never_selects_default_exit():
    world, _, selector, costs = setup_system()
    costs[:] = np.inf
    decision = selector.replan_after_return_invalidated(
        world_state=world, current_position_world=(2, 0), cost_map=costs,
        costmap_revision=8, created_at=8,
    )
    assert not decision.success
    assert decision.strategy is EvacuationStrategy.NO_SAFE_ROUTE
    assert decision.target_exit_id is None


def test_cost_increase_replan_can_exclude_current_exit():
    world, _, selector, costs = setup_system()
    decision = selector.replan_to_safe_exit(
        world_state=world, current_position_world=(2, 0), cost_map=costs,
        costmap_revision=9, created_at=9, excluded_exit_ids=("NEAR",),
    )
    assert decision.success
    assert decision.target_exit_id == "FAR"


def test_opposite_exit_replan_uses_world_direction_and_excludes_front():
    world, _, selector, costs = setup_system()
    decision = selector.replan_to_opposite_exit(
        world_state=world, current_position_world=(2, 0),
        direction_world=(1, 0), cost_map=costs,
        costmap_revision=10, created_at=10, current_exit_id="NEAR",
        minimum_direction_difference_deg=90,
    )
    assert not decision.success  # FAR is also in the forward half-plane.

    world.add_exit(Exit("BACK", (0, 1), (0, 1)))
    decision = selector.replan_to_opposite_exit(
        world_state=world, current_position_world=(2, 0),
        direction_world=(1, 0), cost_map=costs,
        costmap_revision=11, created_at=11, current_exit_id="NEAR",
        minimum_direction_difference_deg=90,
    )
    assert decision.success
    assert decision.target_exit_id == "BACK"


def test_path_validation_finds_dynamic_obstacle():
    from world import DynamicObstacle, DynamicObstacleStatus
    world, _, selector, costs = setup_system()
    world.add_dynamic_obstacle(DynamicObstacle(
        "box", (1, 0), status=DynamicObstacleStatus.ACTIVE
    ))
    valid, cell, reason = selector.validate_grid_path(
        ((2, 0), (1, 0), (0, 0)), start_index=0,
        cost_map=costs, world_state=world,
    )
    assert not valid and cell == (1, 0)
    assert reason == "dynamic_obstacle_on_path"


def test_settings_reject_invalid_values():
    with pytest.raises(ValueError):
        EvacuationRouteSelectionConfig(
            no_fire_information_strategy="nearest_exit"
        )
    with pytest.raises(ValueError):
        HazardKnowledgeConfig(temperature_elevated_c=-1)
    with pytest.raises(ValueError):
        HazardKnowledgeConfig.from_mapping({"unknown": 1})
    with pytest.raises(ValueError):
        ReplanningConfig(max_replan_attempts=-1)
    with pytest.raises(ValueError):
        ReplanningConfig(cooldown_seconds=-0.1)
    with pytest.raises(ValueError):
        PathValidationConfig.from_mapping({"temperature_block_threshold_c": 60})
    with pytest.raises(ValueError):
        PathValidationConfig(stop_before_replanning=False)
    with pytest.raises(ValueError):
        ReplanningConfig(max_replan_attempts=2.5)
