import math

import pytest
import yaml

from navigation.external_waypoint_motion import (
    ExternalWaypointFollower, ExternalWaypointMotionConfig,
    adjust_first_waypoint_departure, load_external_waypoints,
)
from robot.path_follower import RobotState


def write_queue(path, *, frame="map", x=5.544808875, y=-10.198562658):
    path.write_text(yaml.safe_dump({
        "version": 1,
        "frame_id": frame,
        "poses": [{
            "header": {"frame_id": frame},
            "pose": {
                "position": {"x": x, "y": y, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }],
    }))


def test_ros_map_waypoint_is_inverse_transformed_to_factory_world(tmp_path):
    queue = tmp_path / "queue.yaml"
    write_queue(queue)
    result = load_external_waypoints(queue, ExternalWaypointMotionConfig())
    assert len(result) == 1
    assert result[0] == pytest.approx((13.0, 16.0), abs=1e-6)


def test_requested_queue_loads_nine_finite_factory_waypoints():
    config = ExternalWaypointMotionConfig(enabled=True, waypoint_file="unused")
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[4]
        / "fire_robot_rpi/maps/waypoint_queue_latest.yaml"
    )
    result = load_external_waypoints(path, config)
    assert len(result) == 9
    assert all(math.isfinite(value) for point in result for value in point)


def test_wrong_frame_and_bad_quaternion_are_rejected(tmp_path):
    queue = tmp_path / "queue.yaml"
    write_queue(queue, frame="odom")
    with pytest.raises(ValueError, match="frame"):
        load_external_waypoints(queue, ExternalWaypointMotionConfig())
    write_queue(queue)
    document = yaml.safe_load(queue.read_text())
    document["poses"][0]["pose"]["orientation"]["w"] = 2.0
    queue.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="normalized"):
        load_external_waypoints(queue, ExternalWaypointMotionConfig())


def test_enabled_mode_requires_existing_file(tmp_path):
    config = ExternalWaypointMotionConfig(enabled=True, waypoint_file="missing.yaml")
    with pytest.raises(ValueError, match="does not exist"):
        config.resolve_waypoint_file(tmp_path)


def test_motion_follower_moves_through_queue_independently_of_costmap():
    follower = ExternalWaypointFollower(
        ((0.4, 0.0), (0.4, 0.4)), speed_mps=0.2,
        angular_speed_rad_s=math.pi, tolerance_m=0.01,
    )
    state = RobotState(0.0, 0.0, 0.0)
    travelled = 0.0
    for _ in range(80):
        moved, _ = follower.update(state, 0.1)
        travelled += moved
    assert follower.waypoint_index == 2
    assert (state.x, state.y) == pytest.approx((0.4, 0.4))
    assert travelled == pytest.approx(0.8)


def test_first_departure_is_y_parallel_and_offset_by_40_cm():
    config = ExternalWaypointMotionConfig(
        align_first_waypoint_x_to_robot_start=True,
        first_waypoint_y_offset_m=0.4,
    )
    original = ((14.257, 15.269), (11.264, 14.930))
    adjusted = adjust_first_waypoint_departure(original, (13.0, 16.0), config)

    assert adjusted[0] == pytest.approx((13.0, 15.669))
    assert adjusted[1] == original[1]
    assert original[0] == (14.257, 15.269)
