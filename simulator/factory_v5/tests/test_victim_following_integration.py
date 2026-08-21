import numpy as np

from mission.mission_manager import MissionEvent, MissionManager, MissionState
from navigation.victim_following import (
    FollowState, FollowUpdate, VictimFollowingConfig,
    VictimFollowingController,
)
from world import MapMetadata, Victim, VictimStatus, WorldState


def make_world():
    metadata = MapMetadata(0, 5, 0, 5, 1, 6, 6, (0, 0))
    world = WorldState(metadata, np.zeros((6, 6), dtype=bool))
    world.add_victim(Victim("V1", (1, 1)))
    controller = VictimFollowingController(
        metadata, VictimFollowingConfig()
    )
    world.attach_victim_following(controller)
    return world, controller


def make_movable(world):
    for status in (
        VictimStatus.DETECTED, VictimStatus.REACHED,
        VictimStatus.WAITING, VictimStatus.MOVABLE,
    ):
        world.update_victim_status("V1", status, sim_time=1)


def test_world_state_tracks_actual_follow_position_and_legacy_view():
    world, controller = make_world()
    make_movable(world)
    controller.start("V1", (1, 1), (1.5, 1, 0), sim_time=1, costmap_revision=2)
    world.start_victim_following("V1")
    update = FollowUpdate(
        (1.2, 1.0), (1.5, 1.0), 0.3, FollowState.FOLLOWING,
        False, 0.2,
    )
    world.update_victim_following(update, sim_time=2)
    victim = world.get_victim("V1")
    assert victim.position_world == (1.2, 1.0)
    assert victim.current_grid_position == (1, 1)
    assert world.legacy_humans()[0]["x"] == 1.2


def test_world_rejects_immobile_follow_start():
    world, _ = make_world()
    world.update_victim_status("V1", VictimStatus.DETECTED)
    world.update_victim_status("V1", VictimStatus.IMMOBILE)
    try:
        world.start_victim_following("V1")
    except ValueError as exc:
        assert "movable" in str(exc)
    else:
        raise AssertionError("immobile victim incorrectly entered following")


def test_mission_follow_wait_resume_and_failure_transitions():
    manager = MissionManager()
    manager.handle_event(
        MissionEvent.VICTIM_DETECTED,
        victim_id="V1", victim_position=(1, 1),
    )
    manager.handle_event(MissionEvent.VICTIM_REACHED)
    manager.handle_event(MissionEvent.ANNOUNCEMENT_FINISHED)
    manager.handle_event(MissionEvent.VICTIM_STARTED_MOVING)
    manager.handle_event(MissionEvent.PATH_PLANNED, path=[(0, 0), (1, 0)])
    assert manager.current_state is MissionState.ESCORT_VICTIM
    manager.handle_event(MissionEvent.VICTIM_LAGGING)
    assert manager.current_state is MissionState.FOLLOW_WAIT
    manager.handle_event(MissionEvent.VICTIM_CAUGHT_UP)
    assert manager.current_state is MissionState.ESCORT_VICTIM
    manager.handle_event(MissionEvent.VICTIM_LAGGING)
    manager.handle_event(MissionEvent.VICTIM_FOLLOW_FAILED)
    assert manager.current_state is MissionState.FOLLOW_FAILED


def test_completion_records_identity_time_reason_and_clears_active_id():
    world, controller = make_world()
    make_movable(world)
    controller.start("V1", (1, 1), (1.5, 1, 0), sim_time=1, costmap_revision=2)
    world.start_victim_following("V1")
    world.update_victim_status("V1", VictimStatus.EVACUATING)
    world.update_victim_status("V1", VictimStatus.EVACUATED)
    world.complete_victim_following("V1", sim_time=8.5, reason="both at usable exit")
    victim = world.get_victim("V1")
    assert victim.rescued
    assert victim.evacuated_at == 8.5
    assert victim.evacuation_success_reason == "both at usable exit"
    assert world.active_following_victim_id is None
    assert world.legacy_humans() == []
