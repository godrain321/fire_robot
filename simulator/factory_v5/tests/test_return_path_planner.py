import json

import numpy as np
import pytest

from navigation.return_path_planner import ReturnFailureReason, ReturnPathPlanner
from navigation.travel_history import TravelHistory
from world.fire_maps import EstimatedFireMap, MapMetadata


def setup_map():
    metadata = MapMetadata(0, 5, 0, 5, 1, 6, 6, (0, 0))
    fire = EstimatedFireMap(metadata, temperature_blocked_c=60, co_blocked_ppm=1600)
    shape = (6, 6)
    fire.replace_layers(
        np.full(shape, 20.0), np.zeros(shape), np.ones(shape, dtype=bool),
        np.ones(shape),
    )
    return metadata, fire, np.ones(shape), np.zeros(shape, dtype=bool)


def make_history(metadata):
    history = TravelHistory(metadata)
    history.reset((0, 0))
    for point in ((1, 0), (2, 0), (2, 1), (2, 2)):
        history.record_position(point, recorded_at=len(history.get_points()))
    return history


def plan(planner, history, fire, cost, static, dynamic=None, current=(2, 2)):
    dynamic = np.zeros_like(static) if dynamic is None else dynamic
    return planner.create_plan(
        history, current, cost_map=cost, static_obstacle_map=static,
        dynamic_obstacle_map=dynamic, estimated_fire_map=fire, created_at=10,
    )


def test_reverse_simplifies_lines_preserves_corner_and_source():
    metadata, fire, cost, static = setup_map()
    history = make_history(metadata)
    original = history.get_points()
    result = plan(ReturnPathPlanner(metadata), history, fire, cost, static)
    assert result.success
    assert result.path_grid == ((2, 2), (2, 0), (0, 0))
    assert history.get_points() == original
    json.dumps(result.to_dict())


@pytest.mark.parametrize("layer,reason", [
    ("static", ReturnFailureReason.BLOCKED_BY_STATIC_OBSTACLE),
    ("dynamic", ReturnFailureReason.BLOCKED_BY_DYNAMIC_OBSTACLE),
    ("temperature", ReturnFailureReason.BLOCKED_BY_FIRE_RISK),
    ("co", ReturnFailureReason.BLOCKED_BY_FIRE_RISK),
    ("unobserved", ReturnFailureReason.UNOBSERVED_OR_STALE),
    ("invalid", ReturnFailureReason.INVALID_COST),
])
def test_blocking_reasons(layer, reason):
    metadata, fire, cost, static = setup_map()
    dynamic = np.zeros_like(static)
    cell = (0, 1)  # [row,col] for grid (1,0)
    if layer == "static": static[cell] = True
    elif layer == "dynamic": dynamic[cell] = True
    elif layer == "temperature": fire.temperature_c[cell] = 60
    elif layer == "co": fire.co_ppm[cell] = 1600
    elif layer == "unobserved": fire.observed_mask[cell] = False
    elif layer == "invalid": cost[cell] = np.nan
    result = plan(ReturnPathPlanner(metadata), make_history(metadata), fire, cost, static, dynamic)
    assert not result.success
    assert result.failure_reason is reason
    assert result.blocked_segment_index is not None


def test_empty_single_point_and_already_at_start():
    metadata, fire, cost, static = setup_map()
    planner = ReturnPathPlanner(metadata)
    empty = TravelHistory(metadata)
    assert plan(planner, empty, fire, cost, static).failure_reason is ReturnFailureReason.EMPTY_HISTORY
    single = TravelHistory(metadata)
    single.reset((0, 0))
    assert plan(planner, single, fire, cost, static, current=(1, 0)).failure_reason is ReturnFailureReason.INSUFFICIENT_HISTORY
    at_start = plan(planner, single, fire, cost, static, current=(0, 0))
    assert at_start.success and at_start.path_grid == ((0, 0),)


def test_during_return_revalidation_detects_new_risk_and_deviation():
    metadata, fire, cost, static = setup_map()
    planner = ReturnPathPlanner(metadata)
    result = plan(planner, make_history(metadata), fire, cost, static)
    fire.temperature_c[1, 2] = 80
    invalid = planner.validate_plan(
        result, cost_map=cost, static_obstacle_map=static,
        dynamic_obstacle_map=np.zeros_like(static), estimated_fire_map=fire,
        current_time=11, current_position_world=(2, 2),
    )
    assert invalid.failure_reason is ReturnFailureReason.BLOCKED_BY_FIRE_RISK
    fire.temperature_c[1, 2] = 20
    deviated = planner.validate_plan(
        result, cost_map=cost, static_obstacle_map=static,
        dynamic_obstacle_map=np.zeros_like(static), estimated_fire_map=fire,
        current_time=11, current_position_world=(5, 5),
    )
    assert deviated.failure_reason is ReturnFailureReason.PATH_DEVIATION
