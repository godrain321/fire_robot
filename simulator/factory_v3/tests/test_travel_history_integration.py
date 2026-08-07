from pathlib import Path

import numpy as np
import pytest
import yaml

from mission.mission_manager import MissionEvent, MissionManager, MissionState
from navigation import ReturnPathPlanner, TravelHistory, TravelHistoryConfig
from navigation.return_path_planner import ReturnPathConfig
from world import MapMetadata, WorldState

BASE = Path(__file__).resolve().parents[1]


def test_world_state_tracks_actual_positions_and_active_plan():
    metadata = MapMetadata(0, 3, 0, 3, 1, 4, 4, (0, 0))
    world = WorldState(metadata, np.zeros((4, 4), dtype=bool))
    history = TravelHistory(metadata)
    world.attach_travel_history(history)
    world.record_robot_position((0, 0), sim_time=0)
    world.record_robot_position((1, 0), sim_time=1)
    assert world.robot_position_world == (1, 0)
    assert world.get_travel_history().get_points_grid() == ((0, 0), (1, 0))


def test_mission_return_flow():
    manager = MissionManager()
    manager.handle_event(MissionEvent.VICTIM_DETECTED, victim_id="v", victim_position=(1, 1))
    manager.handle_event(MissionEvent.VICTIM_REACHED)
    manager.handle_event(MissionEvent.ANNOUNCEMENT_FINISHED)
    manager.handle_event(MissionEvent.VICTIM_STARTED_MOVING)
    manager.handle_event(MissionEvent.RETURN_REQUESTED)
    assert manager.current_state is MissionState.PLAN_RETURN_BY_HISTORY
    manager.handle_event(MissionEvent.RETURN_PATH_CREATED, path=[(1, 1), (0, 0)])
    assert manager.current_state is MissionState.RETURNING_BY_HISTORY
    manager.handle_event(MissionEvent.RETURN_PATH_INVALIDATED, reason="fire")
    assert manager.current_state is MissionState.RETURN_PATH_BLOCKED


def test_yaml_configuration_loads_and_rejects_invalid_values():
    scenario = yaml.safe_load((BASE / "config/evacuation.yaml").read_text())
    travel = TravelHistoryConfig.from_mapping(scenario["travel_history"])
    returning = ReturnPathConfig.from_mapping(scenario["return_path"])
    assert travel.enabled and returning.prefer_normal_planner
    with pytest.raises(ValueError):
        TravelHistoryConfig.from_mapping({"max_points": 0})
    with pytest.raises(ValueError):
        ReturnPathConfig.from_mapping({"lookahead_distance_m": -1})
    with pytest.raises(ValueError):
        ReturnPathConfig.from_mapping({"invented": True})
