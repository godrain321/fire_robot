from types import SimpleNamespace

import numpy as np

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
        self.seen_paths = []

    def validate_path(self, path, **kwargs):
        return SimpleNamespace(
            safe=self.safe,
            rejection_reasons=(() if self.safe else (
                SimpleNamespace(value="static_obstacle"),
            )),
        )

    def simplify(self, path, *, start_world, goal_world, **kwargs):
        self.seen_paths.append(tuple(path))
        endpoints = (tuple(path[0]), tuple(path[-1]))
        return SimpleNamespace(
            success=True,
            simplified_path_grid=endpoints,
            waypoints_world=(tuple(start_world), tuple(goal_world)),
            failure_reason=None,
        )


def planning_maps(width=8, height=4):
    shape = (height, width)
    return {
        "costmap": np.ones(shape),
        "static_obstacle_map": np.zeros(shape, dtype=bool),
        "dynamic_obstacle_map": np.zeros(shape, dtype=bool),
        "estimated_fire_map": SimpleNamespace(
            blocked_mask=np.zeros(shape, dtype=bool)
        ),
    }


def test_reference_turn_extraction_skips_collinear_yaml_points():
    points = tuple(waypoint(name, grid) for name, grid in (
        ("w1", (1, 0)), ("w2", (2, 0)), ("w3", (3, 0)),
        ("w4", (3, 1)), ("w5", (3, 2)),
    ))
    assert extract_reference_turn_ids(points, 12.0) == ("w1", "w3", "w5")


def test_waypoint_turn_threshold_ignores_45_degree_zigzag():
    points = tuple(waypoint(name, grid) for name, grid in (
        ("w1", (0, 0)), ("w2", (1, 0)), ("w3", (2, 1)),
    ))
    assert extract_reference_turn_ids(points, 12.0) == ("w1", "w2", "w3")
    assert extract_reference_turn_ids(points, 50.0) == ("w1", "w3")
    config = ReferenceWaypointExecutionConfig()
    assert config.direction_change_threshold_deg == 12.0
    assert config.waypoint_turn_threshold_deg == 50.0


def test_execution_path_uses_yaml_turns_as_sequential_targets():
    points = tuple(waypoint(name, grid) for name, grid in (
        ("w1", (1, 0)), ("w2", (2, 0)), ("w3", (3, 0)),
        ("w4", (3, 1)), ("w5", (3, 2)),
    ))
    lookup = {item.waypoint_id: item for item in points}
    original = ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2))
    simplified = SimpleNamespace(
        simplified_path_grid=((0, 0), (3, 0), (4, 2)),
        waypoints_world=((-0.2, 0.1), (3.0, 0.0), (4.1, 2.1)),
    )
    result = build_reference_execution_path(
        original_path_grid=original, simplified_result=simplified,
        reference_waypoint_ids=tuple(lookup), waypoint_by_id=lookup,
        config=ReferenceWaypointExecutionConfig(),
        path_simplifier=Validator(), **planning_maps(),
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
        path_simplifier=Validator(False), **planning_maps(),
    )
    assert not result.used_reference_targets
    assert result.grid_path == simplified.simplified_path_grid
    assert result.fallback_reason == "reference_execution_unsafe:static_obstacle"


def test_execution_preserves_reference_targets_supplied_by_planner():
    points = tuple(waypoint(name, grid) for name, grid in (
        ("w1", (1, 0)), ("w2", (3, 0)), ("w3", (5, 0)),
    ))
    lookup = {item.waypoint_id: item for item in points}
    simplified = SimpleNamespace(
        simplified_path_grid=((2, 0), (5, 0)),
        waypoints_world=((2.0, 0.0), (5.0, 0.0)),
    )
    result = build_reference_execution_path(
        original_path_grid=((2, 0), (1, 0), (3, 0), (4, 0), (5, 0)),
        simplified_result=simplified,
        reference_waypoint_ids=("w1", "w2", "w3"),
        waypoint_by_id=lookup, config=ReferenceWaypointExecutionConfig(),
        path_simplifier=Validator(), **planning_maps(),
    )
    assert result.used_reference_targets
    assert result.reference_waypoint_ids == ("w1", "w3")
    assert result.grid_path == ((2, 0), (1, 0), (5, 0))
    assert result.world_path == ((2.0, 0.0), (1.0, 0.0), (5.0, 0.0))


def test_execution_keeps_first_reference_waypoint_from_planner():
    points = tuple(waypoint(name, grid) for name, grid in (
        ("w1", (2, 0)), ("w2", (4, 0)), ("w3", (6, 0)),
    ))
    lookup = {item.waypoint_id: item for item in points}
    simplified = SimpleNamespace(
        simplified_path_grid=((0, 0), (6, 0)),
        waypoints_world=((0.0, 0.0), (6.0, 0.0)),
    )
    result = build_reference_execution_path(
        original_path_grid=tuple((x, 0) for x in range(7)),
        simplified_result=simplified,
        reference_waypoint_ids=("w1", "w2", "w3"),
        waypoint_by_id=lookup, config=ReferenceWaypointExecutionConfig(),
        path_simplifier=Validator(), **planning_maps(),
    )
    assert result.grid_path[1] == (2, 0)
    assert result.reference_waypoint_ids[0] == "w1"


def test_waypoint_segments_use_unweighted_astar_despite_high_finite_cost():
    points = (waypoint("start", (0, 1)), waypoint("goal", (4, 1)))
    lookup = {item.waypoint_id: item for item in points}
    maps = planning_maps(width=5, height=3)
    maps["costmap"][1, 2] = 1000.0
    validator = Validator()
    simplified = SimpleNamespace(
        simplified_path_grid=((0, 1), (4, 1)),
        waypoints_world=((0.0, 1.0), (4.0, 1.0)),
    )

    result = build_reference_execution_path(
        original_path_grid=((0, 1), (1, 1), (2, 1), (3, 1), (4, 1)),
        simplified_result=simplified,
        reference_waypoint_ids=("start", "goal"),
        waypoint_by_id=lookup, config=ReferenceWaypointExecutionConfig(),
        path_simplifier=validator, **maps,
    )

    assert result.used_reference_targets
    assert validator.seen_paths[0] == (
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)
    )
