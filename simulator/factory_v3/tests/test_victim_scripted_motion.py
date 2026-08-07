import math

import numpy as np
import pytest

from navigation.victim_scripted_motion import (
    ScriptedVictimMotionConfig, ScriptedVictimMotionController,
)
from world.fire_maps import MapMetadata


def metadata():
    return MapMetadata(0, 4, 0, 4, 0.2, 21, 21, (0, 0))


def config(**overrides):
    values = {
        "enabled": True,
        "speed_mps": 0.8,
        "stop_when_detected": True,
        "waypoints_world": ((1, 1), (1, 2), (2, 2)),
    }
    values.update(overrides)
    return ScriptedVictimMotionConfig.from_mapping(values)


def test_victim_walks_waypoints_with_speed_limit_and_no_teleport():
    controller = ScriptedVictimMotionController(
        metadata(), "V1", (1, 1), config()
    )
    free = np.zeros((21, 21), bool)
    before = controller.position_world
    moved = controller.update(
        dt=0.25, static_obstacle_map=free, dynamic_obstacle_map=free,
    )
    assert moved == pytest.approx(0.2)
    assert math.dist(before, controller.position_world) == pytest.approx(0.2)
    assert controller.position_world == pytest.approx((1, 1.2))


def test_victim_follows_corner_and_reaches_exact_final_coordinate():
    controller = ScriptedVictimMotionController(
        metadata(), "V1", (1, 1), config(speed_mps=1.0)
    )
    free = np.zeros((21, 21), bool)
    for _ in range(4):
        controller.update(
            dt=0.5, static_obstacle_map=free, dynamic_obstacle_map=free,
        )
    assert controller.completed
    assert controller.position_world == (2.0, 2.0)


def test_victim_stops_before_an_occupied_segment():
    controller = ScriptedVictimMotionController(
        metadata(), "V1", (1, 1), config()
    )
    static = np.zeros((21, 21), bool)
    static[6, 5] = True  # world (1.0, 1.2)
    moved = controller.update(
        dt=1.0, static_obstacle_map=static,
        dynamic_obstacle_map=np.zeros_like(static),
    )
    assert moved == 0.0
    assert controller.position_world == (1.0, 1.0)


@pytest.mark.parametrize("values", [
    {"enabled": True, "speed_mps": 0, "waypoints_world": ((0, 0), (1, 1))},
    {"enabled": True, "speed_mps": 1, "waypoints_world": ((0, 0),)},
    {"enabled": "yes", "speed_mps": 1, "waypoints_world": ((0, 0), (1, 1))},
])
def test_invalid_scripted_motion_config(values):
    with pytest.raises((ValueError, TypeError)):
        ScriptedVictimMotionConfig.from_mapping(values)
