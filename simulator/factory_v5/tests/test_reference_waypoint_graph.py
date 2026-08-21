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
    assert result.reference_waypoint_ids == ("w1", "w2", "w3", "w4", "w5")


def test_blocked_reference_edge_is_not_used_or_bypassed_by_cell_astar():
    metadata = MapMetadata(0, 6, 0, 2, 1, 7, 3, (0, 0))
    points = tuple(waypoint(f"w{i}", i, 1) for i in range(1, 6))
    planner = ReferenceWaypointGraphPlanner(
        metadata, points,
        ReferenceWaypointGraphConfig(
            neighbor_radius_m=1.1, connector_search_radius_m=1.5,
            connector_candidate_count=1, fallback_to_cell_astar=False,
        ),
    )
    costs = np.ones((3, 7))
    costs[1, 3] = np.inf
    result = planner.plan(costs, (0, 1), (6, 1))
    assert not result.path
    assert not result.used_reference_graph
    assert result.reason == "no safe reference graph route"


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
