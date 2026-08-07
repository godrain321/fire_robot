import json
import math

import numpy as np
import pytest

from navigation.victim_following import (
    FollowState, VictimFollowingConfig, VictimFollowingController,
    evacuation_success_ready,
)
from world.fire_maps import MapMetadata


def metadata():
    return MapMetadata(0, 10, 0, 10, 1, 11, 11, (0, 0))


def controller(**overrides):
    config = VictimFollowingConfig(**overrides)
    return VictimFollowingController(metadata(), config)


def empty_maps():
    return np.zeros((11, 11), dtype=bool), np.zeros((11, 11), dtype=bool)


def start(item, victim=(1.0, 1.0), robot=(1.5, 1.0, 0.0)):
    item.start("V1", victim, robot, sim_time=0.0, costmap_revision=0)


def test_config_validation_and_boolean_types():
    with pytest.raises(ValueError):
        VictimFollowingConfig(target_follow_distance_m=0)
    with pytest.raises(ValueError):
        VictimFollowingConfig(max_follow_distance_m=1.0)
    with pytest.raises(ValueError):
        VictimFollowingConfig(resume_follow_distance_m=2.5)
    with pytest.raises(TypeError):
        VictimFollowingConfig.from_mapping({"enabled": "yes"})
    with pytest.raises(ValueError):
        VictimFollowingConfig(require_victim_at_exit_for_success=False)
    with pytest.raises(ValueError):
        VictimFollowingConfig.from_mapping({"teleport_victim": True})


def test_start_preserves_victim_position_and_does_not_teleport():
    item = controller()
    start(item)
    assert item.position_world == (1.0, 1.0)
    assert item.position_world != (1.5, 1.0)
    assert item.state is FollowState.FOLLOWING


def test_actual_pose_history_uses_distance_sampling_and_not_planned_waypoints():
    item = controller(robot_pose_sample_distance_m=0.5)
    start(item)
    assert not item.record_robot_pose((1.7, 1, 0), sim_time=1, costmap_revision=1)
    assert item.record_robot_pose((2.0, 1, 0), sim_time=2, costmap_revision=2)
    assert [(p.x, p.y) for p in item.pose_history] == [(1.5, 1.0), (2.0, 1.0)]


def test_victim_follows_history_with_speed_limit():
    item = controller(target_follow_distance_m=0.5, victim_speed_mps=0.8)
    start(item, victim=(0.5, 1.0), robot=(1.0, 1.0, 0.0))
    item.record_robot_pose((2.0, 1, 0), sim_time=1, costmap_revision=1)
    static, dynamic = empty_maps()
    update = item.update(
        (2.0, 1.0), dt=0.25, sim_time=1,
        static_obstacle_map=static, dynamic_obstacle_map=dynamic,
    )
    assert update.moved_distance_m <= 0.8 * 0.25 + 1e-12
    assert update.position_world[0] > 0.5
    assert update.target_world[0] == pytest.approx(1.5)


def test_victim_does_not_cross_static_or_dynamic_obstacles():
    for layer in ("static", "dynamic"):
        item = controller(target_follow_distance_m=0.2, victim_speed_mps=2.0)
        start(item, victim=(1.0, 1.0), robot=(1.2, 1.0, 0.0))
        item.record_robot_pose((3.0, 1.0, 0), sim_time=1, costmap_revision=1)
        static, dynamic = empty_maps()
        (static if layer == "static" else dynamic)[1, 2] = True
        update = item.update(
            (3.0, 1.0), dt=1.0, sim_time=1,
            static_obstacle_map=static, dynamic_obstacle_map=dynamic,
        )
        assert update.position_world == (1.0, 1.0)


def test_victim_does_not_cut_between_obstacle_corners():
    item = controller(target_follow_distance_m=0.1, victim_speed_mps=2.0)
    start(item, victim=(1, 1), robot=(1, 1, 0))
    item.record_robot_pose((2, 2, 0), sim_time=1, costmap_revision=1)
    static, dynamic = empty_maps()
    static[1, 2] = True
    update = item.update(
        (2, 2), dt=1, sim_time=1,
        static_obstacle_map=static, dynamic_obstacle_map=dynamic,
    )
    assert update.position_world == (1.0, 1.0)


def test_follow_wait_hysteresis_and_resume():
    item = controller(
        target_follow_distance_m=1.0, max_follow_distance_m=2.5,
        resume_follow_distance_m=2.0, victim_speed_mps=0.1,
    )
    start(item, victim=(1, 1), robot=(1, 1, 0))
    static, dynamic = empty_maps()
    lag = item.update(
        (4, 1), dt=0, sim_time=1,
        static_obstacle_map=static, dynamic_obstacle_map=dynamic,
    )
    assert lag.state is FollowState.FOLLOW_WAIT
    assert lag.robot_should_wait and lag.event == "victim_lagging"
    still_waiting = item.update(
        (3.1, 1), dt=0, sim_time=2,
        static_obstacle_map=static, dynamic_obstacle_map=dynamic,
    )
    assert still_waiting.state is FollowState.FOLLOW_WAIT
    resumed = item.update(
        (2.9, 1), dt=0, sim_time=3,
        static_obstacle_map=static, dynamic_obstacle_map=dynamic,
    )
    assert resumed.state is FollowState.FOLLOWING
    assert resumed.event == "victim_caught_up"


def test_timeout_reprompts_then_records_failure():
    item = controller(
        max_follow_distance_m=2.0, resume_follow_distance_m=1.5,
        follow_wait_timeout_s=1.0, max_reprompt_count=1,
    )
    start(item, victim=(1, 1), robot=(1, 1, 0))
    static, dynamic = empty_maps()
    item.update((4, 1), dt=0, sim_time=0, static_obstacle_map=static, dynamic_obstacle_map=dynamic)
    reprompt = item.update((4, 1), dt=0, sim_time=1, static_obstacle_map=static, dynamic_obstacle_map=dynamic)
    assert reprompt.event == "follow_reprompt"
    failed = item.update((4, 1), dt=0, sim_time=2, static_obstacle_map=static, dynamic_obstacle_map=dynamic)
    assert failed.state is FollowState.FOLLOW_FAILED
    assert failed.robot_should_wait and failed.event == "follow_failed"
    assert item.failure_reason


def test_exit_catch_up_targets_robot_without_instant_position_change():
    item = controller(target_follow_distance_m=1.0)
    start(item, victim=(1, 1), robot=(2, 1, 0))
    before = item.position_world
    item.request_exit_catch_up()
    static, dynamic = empty_maps()
    update = item.update((3, 1), dt=0.1, sim_time=1, static_obstacle_map=static, dynamic_obstacle_map=dynamic)
    assert update.target_world == (3.0, 1.0)
    assert update.position_world != (3.0, 1.0)
    assert math.dist(before, update.position_world) <= 0.08 + 1e-12


def test_evacuation_requires_both_agents_usable_exit_and_valid_route():
    common = dict(
        exit_position_world=(5, 5), exit_radius_m=1.0,
        exit_usable=True, route_valid=True, victim_moved_by_following=True,
    )
    assert evacuation_success_ready(
        robot_position_world=(5, 5), victim_position_world=(5.5, 5), **common
    )
    assert not evacuation_success_ready(
        robot_position_world=(5, 5), victim_position_world=(3, 5), **common
    )
    assert not evacuation_success_ready(
        robot_position_world=(3, 5), victim_position_world=(5, 5), **common
    )
    assert not evacuation_success_ready(
        robot_position_world=(5, 5), victim_position_world=(5, 5),
        **{**common, "exit_usable": False},
    )
    assert not evacuation_success_ready(
        robot_position_world=(5, 5), victim_position_world=(5, 5),
        **{**common, "route_valid": False},
    )


def test_result_is_json_serializable_and_clear_removes_history():
    item = controller()
    start(item)
    json.dumps(item.to_dict())
    item.clear()
    assert item.state is FollowState.INACTIVE
    assert item.pose_history == []
