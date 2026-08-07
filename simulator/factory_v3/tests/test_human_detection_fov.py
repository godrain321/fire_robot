import math

import numpy as np
import pytest

from human_detection_sim import SimpleHumanDetector


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
