import json

import numpy as np
import pytest

from navigation.path_simplifier import (
    PathRiskConfig, PathSimplificationConfig, SafePathSimplifier,
    SegmentRejectionReason,
)
from world import EstimatedFireMap, MapMetadata


def setup_layers(*, config=None, size=7):
    metadata = MapMetadata(0, size - 1, 0, size - 1, 1, size, size, (0, 0))
    estimated = EstimatedFireMap(
        metadata, temperature_blocked_c=60, co_blocked_ppm=1600
    )
    estimated.replace_layers(
        np.full((size, size), 20.0), np.zeros((size, size)),
        np.ones((size, size), dtype=bool), np.zeros((size, size)),
    )
    return (
        SafePathSimplifier(metadata, config), metadata,
        np.ones((size, size)), np.zeros((size, size), dtype=bool),
        np.zeros((size, size), dtype=bool), estimated,
    )


def evaluate(simplifier, costs, static, dynamic, estimated, start=(0, 0), end=(3, 0)):
    return simplifier.evaluate_segment(
        start, end, costmap=costs, static_obstacle_map=static,
        dynamic_obstacle_map=dynamic, estimated_fire_map=estimated,
    )


def test_safe_straight_path_reduces_to_two_waypoints():
    simplifier, _, costs, static, dynamic, estimated = setup_layers()
    path = [(0, 0), (1, 0), (2, 0), (3, 0)]
    result = simplifier.simplify(
        path, costmap=costs, static_obstacle_map=static,
        dynamic_obstacle_map=dynamic, estimated_fire_map=estimated,
        costmap_revision=4,
    )
    assert result.success
    assert result.simplified_path_grid == ((0, 0), (3, 0))
    assert result.used_costmap_revision == 4
    assert path == [(0, 0), (1, 0), (2, 0), (3, 0)]
    json.dumps(result.to_dict())


def test_greedy_selects_farthest_safe_corner():
    simplifier, _, costs, static, dynamic, estimated = setup_layers()
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2)]
    result = simplifier.simplify(
        path, costmap=costs, static_obstacle_map=static,
        dynamic_obstacle_map=dynamic, estimated_fire_map=estimated,
    )
    assert result.simplified_path_grid == ((0, 0), (3, 2))


@pytest.mark.parametrize("layer,value,reason", [
    ("static", True, SegmentRejectionReason.STATIC_OBSTACLE),
    ("dynamic", True, SegmentRejectionReason.DYNAMIC_OBSTACLE),
    ("temperature", 60.0, SegmentRejectionReason.TEMPERATURE_LIMIT_EXCEEDED),
    ("co", 1600.0, SegmentRejectionReason.CO_LIMIT_EXCEEDED),
    ("cost", np.nan, SegmentRejectionReason.INVALID_COST),
    ("cost", np.inf, SegmentRejectionReason.INVALID_COST),
    ("cost", -1.0, SegmentRejectionReason.NEGATIVE_COST),
])
def test_hard_constraints_reject_segment(layer, value, reason):
    simplifier, _, costs, static, dynamic, estimated = setup_layers()
    if layer == "static":
        static[0, 1] = value
    elif layer == "dynamic":
        dynamic[0, 1] = value
    elif layer == "temperature":
        estimated.temperature_c[0, 1] = value
    elif layer == "co":
        estimated.co_ppm[0, 1] = value
    else:
        costs[0, 1] = value
    result = evaluate(simplifier, costs, static, dynamic, estimated)
    assert not result.safe
    assert reason in result.rejection_reasons
    assert result.first_rejected_cell == (1, 0)


def test_corner_cutting_is_explicitly_rejected():
    simplifier, _, costs, static, dynamic, estimated = setup_layers()
    static[0, 1] = True
    result = evaluate(
        simplifier, costs, static, dynamic, estimated,
        start=(0, 0), end=(1, 1),
    )
    assert SegmentRejectionReason.CORNER_CUTTING in result.rejection_reasons


def test_out_of_map_segment_is_rejected():
    simplifier, _, costs, static, dynamic, estimated = setup_layers()
    result = evaluate(
        simplifier, costs, static, dynamic, estimated,
        start=(0, 0), end=(8, 0),
    )
    assert SegmentRejectionReason.OUT_OF_MAP in result.rejection_reasons


def test_unknown_ratio_policy_is_applied():
    config = PathSimplificationConfig(
        risk=PathRiskConfig(max_unknown_ratio=0.0)
    )
    simplifier, _, costs, static, dynamic, estimated = setup_layers(config=config)
    estimated.observed_mask[0, 1] = False
    result = evaluate(simplifier, costs, static, dynamic, estimated)
    assert SegmentRejectionReason.UNKNOWN_RATIO_EXCEEDED in result.rejection_reasons


def test_riskier_shortcut_is_rejected_in_favour_of_corner_path():
    config = PathSimplificationConfig(
        risk=PathRiskConfig(max_risk_increase_ratio=1.0)
    )
    simplifier, _, costs, static, dynamic, estimated = setup_layers(config=config)
    costs[1, 1] = 20.0
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    result = simplifier.simplify(
        path, costmap=costs, static_obstacle_map=static,
        dynamic_obstacle_map=dynamic, estimated_fire_map=estimated,
    )
    assert result.success
    assert (2, 0) in result.simplified_path_grid
    assert any(
        SegmentRejectionReason.RISK_INCREASE_EXCEEDED in item.rejection_reasons
        for item in result.rejected_shortcuts
    )


def test_original_unsafe_path_fails_instead_of_moving():
    simplifier, _, costs, static, dynamic, estimated = setup_layers()
    static[0, 1] = True
    result = simplifier.simplify(
        [(0, 0), (1, 0), (2, 0)], costmap=costs,
        static_obstacle_map=static, dynamic_obstacle_map=dynamic,
        estimated_fire_map=estimated,
    )
    assert not result.success
    assert result.simplified_path_grid == ()


def test_disabled_simplifier_safely_falls_back_to_original():
    config = PathSimplificationConfig(enabled=False)
    simplifier, _, costs, static, dynamic, estimated = setup_layers(config=config)
    path = [(0, 0), (1, 0), (1, 1)]
    result = simplifier.simplify(
        path, costmap=costs, static_obstacle_map=static,
        dynamic_obstacle_map=dynamic, estimated_fire_map=estimated,
    )
    assert result.success and result.fallback_used
    assert result.simplified_path_grid == tuple(path)


def test_exact_start_and_goal_world_positions_are_preserved():
    simplifier, _, costs, static, dynamic, estimated = setup_layers()
    result = simplifier.simplify(
        [(0, 0), (1, 0), (2, 0)], costmap=costs,
        static_obstacle_map=static, dynamic_obstacle_map=dynamic,
        estimated_fire_map=estimated, start_world=(0.1, 0.2),
        goal_world=(2.2, 0.3),
    )
    assert result.waypoints_world[0] == (0.1, 0.2)
    assert result.waypoints_world[-1] == (2.2, 0.3)


def test_existing_lattice_grid_conversion_is_used_without_half_cell_offset():
    simplifier, metadata, costs, static, dynamic, estimated = setup_layers()
    result = simplifier.simplify(
        [(0, 0), (1, 0), (1, 1)], costmap=costs,
        static_obstacle_map=static, dynamic_obstacle_map=dynamic,
        estimated_fire_map=estimated,
    )
    assert result.waypoints_world[0] == metadata.grid_to_world(0, 0)
    assert result.waypoints_world[-1] == metadata.grid_to_world(1, 1)


def test_clearance_rejects_nearby_obstacle():
    config = PathSimplificationConfig(robot_clearance_m=1.0)
    simplifier, _, costs, static, dynamic, estimated = setup_layers(config=config)
    static[1, 1] = True
    result = evaluate(simplifier, costs, static, dynamic, estimated)
    assert SegmentRejectionReason.CLEARANCE_VIOLATION in result.rejection_reasons


@pytest.mark.parametrize("values", [
    {"line_traversal": "bresenham"},
    {"prevent_corner_cutting": False},
    {"robot_clearance_m": -0.1},
    {"risk": {"max_unknown_ratio": 1.1}},
    {"risk": {"max_risk_increase_ratio": -1}},
    {"validation": {"reject_nan": "yes"}},
    {"validation": {"validate_co": False}},
    {"use_inflated_costmap": False},
    {"invented": True},
])
def test_invalid_configuration_is_rejected(values):
    with pytest.raises((ValueError, TypeError)):
        PathSimplificationConfig.from_mapping(values)
