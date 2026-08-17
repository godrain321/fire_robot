#!/usr/bin/env python3
"""Run Stage-6 route policy examples without FDS or Pygame."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

FACTORY = Path(__file__).resolve().parents[1]
if str(FACTORY) not in sys.path:
    sys.path.insert(0, str(FACTORY))

from navigation.evacuation_strategy_selector import (  # noqa: E402
    EvacuationStrategySelector, HazardKnowledgeTracker,
)
from navigation.return_path_planner import ReturnPathConfig, ReturnPathPlanner  # noqa: E402
from navigation.travel_history import TravelHistory  # noqa: E402
from planner.evacuation_planner import EvacuationPlanner  # noqa: E402
from planner.exit_evaluator import ExitEvaluationConfig, ExitEvaluator  # noqa: E402
from world import (  # noqa: E402
    DynamicObstacle, DynamicObstacleStatus, Exit, MapMetadata, WorldState,
)


def build():
    metadata = MapMetadata(0, 6, 0, 6, 1, 7, 7, (0, 0))
    world = WorldState(metadata, np.zeros((7, 7), dtype=bool))
    world.set_mission_entry("ENTRY", (0, 0))
    world.add_exit(Exit("NEAR", (3, 0), (3, 0)))
    world.add_exit(Exit("SAFE", (6, 6), (6, 6)))
    world.estimated_fire_map.replace_layers(
        np.full((7, 7), 20.0), np.zeros((7, 7)),
        np.ones((7, 7), dtype=bool), np.full((7, 7), 10.0),
    )
    history = TravelHistory(metadata)
    for index, point in enumerate(((0, 0), (1, 0), (2, 0))):
        history.record_position(point, recorded_at=index)
    world.attach_travel_history(history)
    evaluator = ExitEvaluator(
        metadata, ExitEvaluationConfig(),
        temperature_blocked_c=60, co_blocked_ppm=1600, base_cost=1,
    )
    selector = EvacuationStrategySelector(
        metadata, HazardKnowledgeTracker(
            temperature_elevated_c=35, co_elevated_ppm=100
        ),
        ReturnPathPlanner(metadata, ReturnPathConfig(max_observation_age_s=None)),
        EvacuationPlanner(evaluator),
    )
    return world, history, selector, np.ones((7, 7))


def main():
    world, history, selector, costs = build()
    original = history.get_points()
    no_fire = selector.select_initial_route(
        world_state=world, travel_history=history,
        start_position_world=(2, 0), victim_position_world=(2, 0),
        cost_map=costs, costmap_revision=1, created_at=10,
    )
    print("NO FIRE:", no_fire.strategy.value, no_fire.path_grid)

    world.estimated_fire_map.update_cell(3, 0, temperature_c=65, sim_time=11)
    with_fire = selector.select_initial_route(
        world_state=world, travel_history=history,
        start_position_world=(2, 0), victim_position_world=(2, 0),
        cost_map=costs, costmap_revision=2, created_at=11,
    )
    print("FIRE:", with_fire.strategy.value, with_fire.target_exit_id)
    selected = with_fire.evacuation_plan.selected_evaluation
    print("distance/risk:", selected.path_length_m, selected.accumulated_risk_cost)

    world.add_dynamic_obstacle(DynamicObstacle(
        "box", (1, 0), status=DynamicObstacleStatus.ACTIVE
    ))
    valid, blocked, reason = selector.validate_grid_path(
        no_fire.path_grid, start_index=0, cost_map=costs, world_state=world
    )
    print("RETURN INVALIDATED / STOP:", not valid, blocked, reason)
    replacement = selector.replan_after_return_invalidated(
        world_state=world, current_position_world=(2, 0), cost_map=costs,
        costmap_revision=3, created_at=12,
    )
    print("REPLAN:", replacement.strategy.value, replacement.success)
    print("HISTORY UNCHANGED:", history.get_points() == original)
    json.dumps(replacement.to_dict())
    print("JSON: OK")


if __name__ == "__main__":
    main()
