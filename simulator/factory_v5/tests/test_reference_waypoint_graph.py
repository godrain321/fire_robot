from types import SimpleNamespace

import numpy as np

from planner.reference_waypoint_graph import (
    ReferenceWaypointGraphConfig, ReferenceWaypointGraphPlanner,
)
from world import MapMetadata


def waypoint(name, col, row):
    return SimpleNamespace(
        waypoint_id=name,
        factory_grid=(col, row),
        factory_world=(float(col), float(row)),
    )


def test_global_route_uses_reference_nodes_between_local_connectors():
    metadata = MapMetadata(0, 6, 0, 2, 1, 7, 3, (0, 0))
    points = tuple(waypoint(f"w{i}", i, 1) for i in range(1, 6))
    planner = ReferenceWaypointGraphPlanner(
        metadata, points,
        ReferenceWaypointGraphConfig(
            neighbor_radius_m=1.1, connector_search_radius_m=1.5,
            connector_candidate_count=2,
        ),
    )
    result = planner.plan(np.ones((3, 7)), (0, 1), (6, 1))
    assert result.path[0] == (0, 1)
    assert result.path[-1] == (6, 1)
    assert result.used_reference_graph
    assert result.reference_waypoint_ids == ("w2", "w3", "w4", "w5")


def test_off_graph_start_connects_directly_to_second_reference_waypoint():
    metadata = MapMetadata(0, 4, 0, 3, 1, 5, 4, (0, 0))
    points = (
        waypoint("nearest", 1, 1),
        waypoint("second", 2, 2),
        waypoint("third", 3, 2),
    )
    planner = ReferenceWaypointGraphPlanner(
        metadata, points,
        ReferenceWaypointGraphConfig(
            neighbor_radius_m=1.5, connector_search_radius_m=1.5,
            connector_candidate_count=2,
        ),
    )
    result = planner.plan(np.ones((4, 5)), (1, 0), (4, 2))
    assert result.used_reference_graph
    assert result.reference_waypoint_ids == ("second", "third")
    assert result.path[0] == (1, 0)
    assert (2, 2) in result.path
    assert result.path.index((2, 2)) < result.path.index((3, 2))


def test_single_anchor_short_route_uses_direct_cell_astar():
    metadata = MapMetadata(0, 3, 0, 2, 1, 4, 3, (0, 0))
    points = (
        waypoint("near", 1, 1), waypoint("far", 3, 1),
    )
    planner = ReferenceWaypointGraphPlanner(
        metadata, points,
        ReferenceWaypointGraphConfig(
            neighbor_radius_m=2.1, connector_search_radius_m=1.5,
            connector_candidate_count=1, fallback_to_cell_astar=True,
        ),
    )
    result = planner.plan(np.ones((3, 4)), (0, 1), (1, 2))
    assert result.path[0] == (0, 1)
    assert result.path[-1] == (1, 2)
    assert not result.used_reference_graph
    assert result.reference_waypoint_ids == ()
    assert result.reason == "single reference anchor; cell A* fallback: path found"


def test_blocked_reference_edge_falls_back_to_cell_astar():
    metadata = MapMetadata(0, 6, 0, 2, 1, 7, 3, (0, 0))
    points = tuple(waypoint(f"w{i}", i, 1) for i in range(1, 6))
    planner = ReferenceWaypointGraphPlanner(
        metadata, points,
        ReferenceWaypointGraphConfig(
            neighbor_radius_m=1.1, connector_search_radius_m=1.5,
            connector_candidate_count=1, fallback_to_cell_astar=True,
        ),
    )
    costs = np.ones((3, 7))
    costs[1, 3] = np.inf
    result = planner.plan(costs, (0, 1), (6, 1))
    assert result.path
    assert not result.used_reference_graph
    assert result.reason == "no safe reference graph route; cell A* fallback: path found"


def test_start_equal_goal_returns_already_at_goal_before_graph_search():
    metadata = MapMetadata(0, 2, 0, 2, 1, 3, 3, (0, 0))
    planner = ReferenceWaypointGraphPlanner(
        metadata, (waypoint("w1", 0, 0), waypoint("w2", 2, 2)),
    )

    result = planner.plan(np.ones((3, 3)), (1, 1), (1, 1))

    assert result.path == [(1, 1)]
    assert result.total_cost == 0.0
    assert result.reason == "ALREADY_AT_GOAL"
    assert not result.used_reference_graph


def test_true_no_path_requires_graph_and_cell_astar_failure():
    metadata = MapMetadata(0, 4, 0, 2, 1, 5, 3, (0, 0))
    points = (waypoint("left", 1, 1), waypoint("right", 3, 1))
    planner = ReferenceWaypointGraphPlanner(
        metadata, points,
        ReferenceWaypointGraphConfig(
            neighbor_radius_m=2.1, connector_search_radius_m=1.1,
            fallback_to_cell_astar=True,
        ),
    )
    costs = np.ones((3, 5))
    costs[:, 2] = np.inf

    result = planner.plan(costs, (0, 1), (4, 1))

    assert not result.path
    assert not result.used_reference_graph
    assert "cell A* fallback: no traversable path" in result.reason


def test_graph_selects_lower_cost_reference_branch():
    metadata = MapMetadata(0, 4, 0, 4, 1, 5, 5, (0, 0))
    points = (
        waypoint("start", 1, 2), waypoint("upper1", 2, 1),
        waypoint("upper2", 3, 2), waypoint("lower1", 2, 3),
    )
    planner = ReferenceWaypointGraphPlanner(
        metadata, points,
        ReferenceWaypointGraphConfig(
            neighbor_radius_m=1.5, connector_search_radius_m=1.1,
            connector_candidate_count=1,
        ),
    )
    costs = np.ones((5, 5))
    costs[1, 2] = 20.0
    result = planner.plan(costs, (0, 2), (4, 2))
    assert result.used_reference_graph
    assert "lower1" in result.reference_waypoint_ids
    assert "upper1" not in result.reference_waypoint_ids


def test_graph_configuration_rejects_unsafe_values():
    for values in (
        {"neighbor_radius_m": 0}, {"connector_search_radius_m": -1},
        {"connector_candidate_count": 0}, {"fallback_to_cell_astar": "no"},
        {"unknown": True},
    ):
        try:
            ReferenceWaypointGraphConfig.from_mapping(values)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"configuration should fail: {values}")
