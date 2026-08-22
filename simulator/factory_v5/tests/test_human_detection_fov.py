import math

import numpy as np
import pytest

from human_detection_sim import (
    MovingObjectDetectionConfig, MovingObjectDetector, SimpleHumanDetector,
    candidate_confirmation_ready, candidate_standoff_position,
)


def human(x, y):
    return {"id": "V1", "x": x, "y": y}


def test_detects_only_inside_forward_one_hundred_degree_fov():
    detector = SimpleHumanDetector(10.0, 100.0)
    assert detector.detect((0, 0), [human(5, 0)], robot_heading_rad=0.0)
    assert detector.detect(
        (0, 0), [human(5 * math.cos(math.radians(50)),
                           5 * math.sin(math.radians(50)))],
        robot_heading_rad=0.0,
    )
    assert not detector.detect((0, 0), [human(0, 5)], robot_heading_rad=0.0)
    assert not detector.detect((0, 0), [human(-5, 0)], robot_heading_rad=0.0)


def test_fov_rotates_with_robot_heading_and_wraps_at_pi():
    detector = SimpleHumanDetector(10.0, 100.0)
    assert detector.detect(
        (0, 0), [human(0, 5)], robot_heading_rad=math.pi / 2
    )
    target_angle = math.radians(-179)
    assert detector.detect(
        (0, 0), [human(math.cos(target_angle), math.sin(target_angle))],
        robot_heading_rad=math.radians(179),
    )


def test_range_and_line_of_sight_still_apply_inside_fov():
    detector = SimpleHumanDetector(10.0, 100.0)
    assert not detector.detect((0, 0), [human(10.1, 0)], robot_heading_rad=0.0)
    obstacle = np.zeros((20, 20), dtype=bool)
    obstacle[0, 5] = True
    assert not detector.detect(
        (0, 0), [human(1, 0)], robot_heading_rad=0.0,
        obstacle_map=obstacle, map_origin=(0, 0), map_resolution=0.1,
    )


@pytest.mark.parametrize("fov", [0, -1, 361, math.nan])
def test_invalid_fov_is_rejected(fov):
    with pytest.raises(ValueError):
        SimpleHumanDetector(10.0, fov)


def test_moving_object_requires_absolute_position_change_inside_ten_metres():
    detector = MovingObjectDetector(MovingObjectDetectionConfig())
    assert detector.update((0, 0), [human(8, 0)]) == []
    assert detector.update((0, 0), [human(8, 0)]) == []
    candidates = detector.update((0, 0), [human(8.1, 0)])
    assert [item["id"] for item in candidates] == ["V1"]
    assert candidates[0]["position_change_m"] == pytest.approx(0.1)


def test_moving_object_radius_is_omnidirectional_and_excludes_outside_object():
    detector = MovingObjectDetector(MovingObjectDetectionConfig())
    detector.update((0, 0), [human(-9, 0)])
    assert detector.update((0, 0), [human(-9.1, 0)])
    detector.update((0, 0), [human(-11, 0)])
    assert not detector.update((0, 0), [human(-11.1, 0)])


def test_moving_object_is_not_detected_through_slam_wall():
    detector = MovingObjectDetector(MovingObjectDetectionConfig())
    wall = np.zeros((20, 20), dtype=bool)
    wall[0, 5] = True
    kwargs = {
        "obstacle_map": wall,
        "map_origin": (0, 0),
        "map_resolution": 0.1,
    }
    assert detector.update((0, 0), [human(1, 0)], **kwargs) == []
    assert detector.update((0, 0), [human(1.1, 0)], **kwargs) == []


def test_moving_object_config_rejects_invalid_confirmation_geometry():
    with pytest.raises(ValueError):
        MovingObjectDetectionConfig(candidate_stop_distance_m=10.0)


def test_candidate_requires_five_metre_standoff_and_full_three_second_wait():
    config = MovingObjectDetectionConfig()
    assert not candidate_confirmation_ready(5.1, 1.0, 4.0, config)
    assert not candidate_confirmation_ready(5.0, 1.0, 3.9, config)
    assert candidate_confirmation_ready(5.0, 1.0, 4.0, config)


def test_candidate_approach_targets_five_metre_standoff_not_occupied_cell():
    target = candidate_standoff_position((0.0, 0.0), (10.0, 0.0), 5.0)
    assert target == pytest.approx((5.0, 0.0))
    assert candidate_standoff_position(
        (6.0, 0.0), (10.0, 0.0), 5.0
    ) == pytest.approx((6.0, 0.0))
