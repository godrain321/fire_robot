import json

import numpy as np
import pytest

from mission.mission_manager import (
    InvalidTransitionError,
    MissionEvent,
    MissionManager,
    MissionState,
)


def advance_to_waiting(manager):
    manager.handle_event(
        MissionEvent.VICTIM_DETECTED,
        victim_id="victim_1",
        victim_position=(3.0, 4.0),
        sim_time=1.0,
    )
    manager.handle_event(MissionEvent.VICTIM_REACHED, sim_time=2.0)
    manager.handle_event(MissionEvent.ANNOUNCEMENT_FINISHED, sim_time=3.0)


def advance_to_escort(manager):
    advance_to_waiting(manager)
    manager.handle_event(MissionEvent.VICTIM_STARTED_MOVING, sim_time=4.0)
    manager.handle_event(
        MissionEvent.PATH_PLANNED,
        exit_id="EXIT2",
        exit_position=(8.0, 9.0),
        path=np.asarray([[1, 2], [2, 2]]),
        sim_time=5.0,
    )


def test_initial_state():
    assert MissionManager().get_state() is MissionState.SEARCH_EXITS


def test_normal_flow_and_context_storage():
    manager = MissionManager()
    advance_to_escort(manager)
    assert manager.current_state is MissionState.ESCORT_VICTIM
    assert manager.selected_victim_id == "victim_1"
    assert manager.selected_victim_position == (3.0, 4.0)
    assert manager.selected_exit_id == "EXIT2"
    assert manager.selected_exit_position == (8.0, 9.0)
    manager.handle_event(MissionEvent.EXIT_REACHED, sim_time=10.0)
    assert manager.current_state is MissionState.EVACUATION_COMPLETE


def test_announcement_wait_and_victim_started_moving():
    manager = MissionManager()
    advance_to_waiting(manager)
    assert manager.current_state is MissionState.WAITING_FOR_VICTIM
    assert manager.waiting_started_at == 3.0
    manager.handle_event(MissionEvent.VICTIM_STARTED_MOVING, sim_time=4.0)
    assert manager.current_state is MissionState.PLAN_EVACUATION
    assert manager.waiting_started_at is None


def test_wait_timeout_reports_immobile_victim():
    manager = MissionManager(victim_wait_timeout_s=2.0)
    advance_to_waiting(manager)
    assert manager.check_wait_timeout(4.9) is None
    transition = manager.check_wait_timeout(5.0)
    assert transition.next_state is MissionState.REPORT_IMMOBILE_VICTIM
    assert manager.immobile_victims[0]["id"] == "victim_1"
    manager.handle_event(MissionEvent.SEARCH_RESUMED, sim_time=5.1)
    assert manager.current_state is MissionState.SEARCH_EXITS


def test_blocked_path_replans_and_returns_to_plan():
    manager = MissionManager()
    advance_to_escort(manager)
    manager.handle_event(MissionEvent.PATH_BLOCKED, sim_time=6.0)
    assert manager.current_state is MissionState.REPLAN
    assert manager.replan_count == 1
    manager.handle_event(MissionEvent.RETRY_REQUESTED, sim_time=6.1)
    assert manager.current_state is MissionState.PLAN_EVACUATION


def test_maximum_replans_prevents_infinite_loop():
    manager = MissionManager(max_replan_count=1)
    advance_to_escort(manager)
    manager.handle_event(MissionEvent.PATH_BLOCKED)
    manager.handle_event(MissionEvent.RETRY_REQUESTED)
    transition = manager.handle_event(MissionEvent.PATH_PLANNING_FAILED)
    assert transition.next_state is MissionState.NO_SAFE_EXIT
    assert manager.replan_count == 1


def test_no_safe_exit_records_failures_and_can_retry():
    manager = MissionManager(max_replan_count=0)
    advance_to_waiting(manager)
    manager.handle_event(MissionEvent.VICTIM_STARTED_MOVING)
    manager.handle_event(
        MissionEvent.PATH_PLANNING_FAILED,
        failed_exits={"EXIT1": "blocked", "EXIT2": "unsafe"},
    )
    assert manager.current_state is MissionState.NO_SAFE_EXIT
    assert manager.failed_exits == {"EXIT1": "blocked", "EXIT2": "unsafe"}
    # A retry is explicitly recognized, but the zero limit keeps it safe.
    transition = manager.handle_event(MissionEvent.RETRY_REQUESTED)
    assert transition.next_state is MissionState.NO_SAFE_EXIT


def test_invalid_transition_is_not_silently_ignored():
    manager = MissionManager()
    assert not manager.can_transition(MissionEvent.PATH_PLANNED)
    with pytest.raises(InvalidTransitionError):
        manager.handle_event(MissionEvent.PATH_PLANNED)
    assert manager.last_error


def test_history_order_and_reset():
    manager = MissionManager()
    advance_to_waiting(manager)
    assert [item.event for item in manager.get_transition_history()] == [
        MissionEvent.VICTIM_DETECTED,
        MissionEvent.VICTIM_REACHED,
        MissionEvent.ANNOUNCEMENT_FINISHED,
    ]
    manager.reset()
    assert manager.current_state is MissionState.SEARCH_EXITS
    assert manager.previous_state is None
    assert manager.get_transition_history() == ()
    assert manager.selected_victim_id is None


def test_to_dict_is_json_serializable_with_numpy_path():
    manager = MissionManager()
    advance_to_escort(manager)
    payload = manager.to_dict()
    json.dumps(payload)
    assert payload["current_state"] == "ESCORT_VICTIM"
    assert payload["current_path"] == [[1, 2], [2, 2]]
    assert payload["transition_history"][0]["event"] == "VICTIM_DETECTED"


def test_remaining_victims_returns_to_search():
    manager = MissionManager()
    advance_to_escort(manager)
    transition = manager.handle_event(
        MissionEvent.EXIT_REACHED, remaining_victims=True
    )
    assert transition.next_state is MissionState.SEARCH_EXITS
    assert manager.selected_victim_id is None


def test_no_remaining_victims_stays_complete():
    manager = MissionManager()
    advance_to_escort(manager)
    manager.handle_event(MissionEvent.EXIT_REACHED, remaining_victims=False)
    assert manager.current_state is MissionState.EVACUATION_COMPLETE
    assert not manager.can_transition(MissionEvent.VICTIM_DETECTED)
