from pathlib import Path

import pytest
import yaml

from navigation.slam_reference_waypoints import (
    RosFactoryTransform, SlamReferenceWaypointConfig,
    load_slam_reference_waypoints,
)
from world import MapMetadata


FACTORY_DIR = Path(__file__).resolve().parents[1]


def factory_metadata():
    return MapMetadata(1.8, 30.0, 5.6, 28.0, 0.2, 142, 113, (1.8, 5.6))


def test_all_measured_slam_points_transform_and_round_trip():
    points, transform = load_slam_reference_waypoints(
        FACTORY_DIR / "config/slam_reference_waypoints.yaml",
        FACTORY_DIR / "config/map_metadata.yaml",
        factory_metadata(),
    )
    assert len(points) == 159
    assert points[0].waypoint_id == "w1"
    for point in points:
        restored = transform.factory_to_ros(*point.factory_world)
        assert restored == pytest.approx(point.ros_map_world, abs=1e-10)


def test_transform_matches_existing_semantic_exit_coordinates():
    metadata = yaml.safe_load(
        (FACTORY_DIR / "config/map_metadata.yaml").read_text()
    )
    semantics = yaml.safe_load(
        (FACTORY_DIR / "config/semantic_points.yaml").read_text()
    )
    transform = RosFactoryTransform.from_mapping(metadata)
    exit3 = semantics["poses"]["EXIT3"]
    converted = transform.ros_to_factory(exit3["ros"]["x"], exit3["ros"]["y"])
    assert converted == pytest.approx(
        (exit3["fds_v3"]["x"], exit3["fds_v3"]["y"]), abs=1e-10
    )
    assert transform.ros_yaw_to_factory(exit3["ros"]["yaw"]) == pytest.approx(
        exit3["fds_v3"]["yaw"], abs=1e-10
    )


@pytest.mark.parametrize("values", [
    {"enabled": "yes"},
    {"waypoint_file": ""},
    {"unknown": True},
])
def test_invalid_reference_configuration_is_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        SlamReferenceWaypointConfig.from_mapping(values)
