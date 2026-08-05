import json
from pathlib import Path

import numpy as np
import yaml

from mission.mission_manager import MissionEvent, MissionManager, MissionState
from world import MapMetadata, WorldState
from mapping.partial_costmap import PartialCostmapConfig, PartialFireCostmap


BASE = Path(__file__).resolve().parents[1]


def ready(manager):
    manager.handle_event(
        MissionEvent.VICTIM_DETECTED, victim_id="v", victim_position=(1, 1)
    )
    manager.handle_event(MissionEvent.VICTIM_REACHED)
    manager.handle_event(MissionEvent.ANNOUNCEMENT_FINISHED)
    manager.handle_event(MissionEvent.VICTIM_STARTED_MOVING)
    manager.handle_event(MissionEvent.VICTIM_READY_FOR_EVACUATION)


def test_mission_no_hazard_return_and_invalidated_replan_flow():
    manager = MissionManager()
    ready(manager)
    assert manager.current_state is MissionState.EVALUATING_HAZARD_INFORMATION
    manager.handle_event(MissionEvent.NO_HAZARD_INFORMATION)
    manager.handle_event(MissionEvent.RETURN_PATH_CREATED, path=[(1, 1), (0, 0)])
    assert manager.current_state is MissionState.RETURNING_BY_HISTORY
    manager.handle_event(MissionEvent.RETURN_PATH_INVALIDATED)
    manager.handle_event(MissionEvent.REPLAN_REQUESTED)
    manager.handle_event(MissionEvent.ENTRANCE_ROUTE_CREATED, path=[(1, 1), (0, 0)])
    assert manager.current_state is MissionState.RETURNING_TO_ENTRANCE


def test_mission_hazard_exit_and_no_safe_route_flows():
    success = MissionManager()
    ready(success)
    success.handle_event(MissionEvent.HAZARD_INFORMATION_AVAILABLE)
    success.handle_event(MissionEvent.SAFE_EXIT_SELECTED, exit_id="E2")
    success.handle_event(MissionEvent.EVACUATION_PLAN_CREATED, path=[(1, 1)])
    assert success.current_state is MissionState.ESCORT_VICTIM

    failed = MissionManager()
    ready(failed)
    failed.handle_event(MissionEvent.HAZARD_INFORMATION_AVAILABLE)
    failed.handle_event(MissionEvent.NO_SAFE_ROUTE_FOUND)
    assert failed.current_state is MissionState.NO_SAFE_ROUTE


def test_world_route_metadata_serializes_and_revisions_are_monotonic():
    metadata = MapMetadata(0, 1, 0, 1, 1, 2, 2, (0, 0))
    world = WorldState(metadata, np.zeros((2, 2), dtype=bool))
    world.set_mission_entry("ENTRY", (0, 0))
    world.update_costmap_revision(2)
    payload = world.to_dict()
    assert payload["mission_entry_id"] == "ENTRY"
    assert payload["costmap_revision"] == 2
    json.dumps(payload)


def test_stage6_yaml_sections_load():
    scenario = yaml.safe_load((BASE / "config/evacuation.yaml").read_text())
    assert scenario["mission_entry_id"] == "MISSION_ENTRY"
    assert scenario["evacuation_route_selection"]["fire_information_strategy"] == "evaluate_all_exits"
    assert scenario["replanning"]["max_replan_attempts"] == 5


def test_costmap_revision_changes_only_for_nonempty_sensor_update():
    class Grid:
        width = height = 2
        resolution = 1.0
        def world_to_grid(self, x, y):
            return int(x), int(y)
        def in_bounds(self, node):
            return 0 <= node[0] < 2 and 0 <= node[1] < 2

    belief = PartialFireCostmap(
        Grid(), np.zeros((2, 2), dtype=bool), PartialCostmapConfig()
    )
    assert belief.revision == 0
    belief.update_co_observation(0, 0, 10, 1)
    assert belief.revision == 1
