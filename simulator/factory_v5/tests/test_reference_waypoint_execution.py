from types import SimpleNamespace

from navigation.reference_waypoint_execution import (
    ReferenceWaypointExecutionConfig,
    build_reference_execution_path,
    extract_reference_turn_ids,
)


def waypoint(name, grid):
    return SimpleNamespace(
        waypoint_id=name,
        factory_grid=grid,
        factory_world=(float(grid[0]), float(grid[1])),
    )


class Validator:
    def __init__(self, safe=True):
        self.safe = safe

    def validate_path(self, path, **kwargs):
        return SimpleNamespace(
            safe=self.safe,
            rejection_reasons=(() if self.safe else (
                SimpleNamespace(value="static_obstacle"),
            )),
        )


def test_reference_turn_extraction_skips_collinear_yaml_points():
    points = tuple(waypoint(name, grid) for name, grid in (
        ("w1", (1, 0)), ("w2", (2, 0)), ("w3", (3, 0)),
        ("w4", (3, 1)), ("w5", (3, 2)),
    ))
    assert extract_reference_turn_ids(points, 12.0) == ("w1", "w3", "w5")


def test_execution_path_uses_yaml_turns_as_sequential_targets():
    points = tuple(waypoint(name, grid) for name, grid in (
        ("w1", (1, 0)), ("w2", (2, 0)), ("w3", (3, 0)),
        ("w4", (3, 1)), ("w5", (3, 2)),
    ))
    lookup = {item.waypoint_id: item for item in points}
    original = ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2))
    simplified = SimpleNamespace(
        simplified_path_grid=((0, 0), (3, 0), (4, 2)),
        waypoints_world=((0.1, 0.1), (3.0, 0.0), (4.1, 2.1)),
    )
    result = build_reference_execution_path(
        original_path_grid=original, simplified_result=simplified,
        reference_waypoint_ids=tuple(lookup), waypoint_by_id=lookup,
        config=ReferenceWaypointExecutionConfig(),
        path_simplifier=Validator(), costmap=None,
        static_obstacle_map=None, dynamic_obstacle_map=None,
        estimated_fire_map=None,
    )
    assert result.used_reference_targets
    assert result.reference_waypoint_ids == ("w1", "w3", "w5")
    assert result.grid_path == ((0, 0), (1, 0), (3, 0), (3, 2), (4, 2))
    assert result.world_path[1:4] == ((1.0, 0.0), (3.0, 0.0), (3.0, 2.0))


def test_unsafe_reference_execution_falls_back_to_verified_simplified_path():
    point = waypoint("w1", (1, 0))
    simplified = SimpleNamespace(
        simplified_path_grid=((0, 0), (2, 0)),
        waypoints_world=((0.0, 0.0), (2.0, 0.0)),
    )
    result = build_reference_execution_path(
        original_path_grid=((0, 0), (1, 0), (2, 0)),
        simplified_result=simplified, reference_waypoint_ids=("w1",),
        waypoint_by_id={"w1": point},
        config=ReferenceWaypointExecutionConfig(),
        path_simplifier=Validator(False), costmap=None,
        static_obstacle_map=None, dynamic_obstacle_map=None,
        estimated_fire_map=None,
    )
    assert not result.used_reference_targets
    assert result.grid_path == simplified.simplified_path_grid
    assert result.fallback_reason == "reference_execution_unsafe:static_obstacle"
