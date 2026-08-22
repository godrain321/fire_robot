import json
import math

import numpy as np
import pytest

from planner.exit_evaluator import (
    ExitEvaluationConfig, ExitEvaluator, ExitRejectionReason,
    within_usable_confirmation_distance,
)
from world.entities import Exit, ExitStatus
from world.fire_maps import EstimatedFireMap, MapMetadata


def environment(size=7):
    metadata = MapMetadata(0, size - 1, 0, size - 1, 1, size, size, (0, 0))
    fire = EstimatedFireMap(metadata, temperature_blocked_c=60, co_blocked_ppm=1600)
    shape = (size, size)
    fire.replace_layers(
        np.full(shape, 20.0), np.zeros(shape), np.ones(shape, dtype=bool),
        np.zeros(shape),
    )
    return metadata, fire, np.ones(shape), np.zeros(shape, dtype=bool)


def evaluator(metadata, **settings):
    return ExitEvaluator(
        metadata, ExitEvaluationConfig(**settings),
        temperature_blocked_c=60, co_blocked_ppm=1600, base_cost=1,
    )


def test_usable_confirmation_distance_is_three_metres_without_coverage_gate():
    config = ExitEvaluationConfig(usable_confirmation_distance_m=3.0)
    assert within_usable_confirmation_distance((0, 0), (3, 0), config)
    assert not within_usable_confirmation_distance((0, 0), (3.01, 0), config)


def evaluate(item, start=(0, 0), **changes):
    metadata, fire, cost, static = environment()
    dynamic = np.zeros_like(static)
    for key, value in changes.items():
        if key == "temperature": fire.temperature_c[value[0]] = value[1]
        elif key == "co": fire.co_ppm[value[0]] = value[1]
        elif key == "observed": fire.observed_mask[value[0]] = value[1]
        elif key == "static": static[value] = True
        elif key == "dynamic": dynamic[value] = True
        elif key == "cost": cost[value[0]] = value[1]
    result = evaluator(metadata).evaluate(
        item, start, cost_map=cost, static_obstacle_map=static,
        dynamic_obstacle_map=dynamic, estimated_fire_map=fire, evaluated_at=3,
    )
    return result, metadata, fire, cost, static


def test_single_safe_exit_metrics_and_json():
    result, *_ = evaluate(Exit("E1", (4, 0), (4, 0), ExitStatus.USABLE))
    assert result.reachable and result.accepted
    assert result.path_grid[0] == (0, 0) and result.path_grid[-1] == (4, 0)
    assert result.path_length_m == 4
    assert result.accumulated_risk_cost == 0
    assert result.max_path_temperature_c == 20
    assert result.max_path_co_ppm == 0
    assert result.unknown_ratio == 0
    json.dumps(result.to_dict())


@pytest.mark.parametrize("status,reason", [
    (ExitStatus.BLOCKED, ExitRejectionReason.EXIT_BLOCKED),
    (ExitStatus.DANGEROUS, ExitRejectionReason.EXIT_DANGEROUS),
    (ExitStatus.DANGER_EXPECTED, ExitRejectionReason.EXIT_DANGER_EXPECTED),
])
def test_exit_status_rejected_before_path(status, reason):
    result, *_ = evaluate(Exit("E", (4, 0), (4, 0), status))
    assert not result.accepted and result.rejection_reasons == (reason,)


def test_out_of_map_and_registered_approach_obstacle_reasons():
    result, *_ = evaluate(Exit("E", (9, 0), (4, 0)))
    assert ExitRejectionReason.OUT_OF_MAP in result.rejection_reasons
    result, *_ = evaluate(Exit("E", (4, 0), (4, 0)), static=(0, 4))
    assert result.rejection_reasons == (ExitRejectionReason.STATIC_OBSTACLE,)
    result, *_ = evaluate(Exit("E", (4, 0), (4, 0)), dynamic=(0, 4))
    assert result.rejection_reasons == (ExitRejectionReason.DYNAMIC_OBSTACLE,)


def test_wall_center_without_approach_searches_neighbor_and_no_candidate_fails():
    metadata, fire, cost, static = environment()
    static[5, 5] = True
    item = Exit("E", (5, 5), None)
    result = evaluator(metadata).evaluate(
        item, (0, 0), cost_map=cost, static_obstacle_map=static,
        dynamic_obstacle_map=np.zeros_like(static), estimated_fire_map=fire,
        evaluated_at=1,
    )
    assert result.accepted and result.approach_position_grid != (5, 5)
    static[4:7, 4:7] = True
    result = evaluator(metadata).evaluate(
        item, (0, 0), cost_map=cost, static_obstacle_map=static,
        dynamic_obstacle_map=np.zeros_like(static), estimated_fire_map=fire,
        evaluated_at=2,
    )
    assert result.rejection_reasons == (ExitRejectionReason.NO_APPROACH_CELL,)


def test_no_path_and_start_equals_goal():
    metadata, fire, cost, static = environment()
    static[:, 3] = True
    item = Exit("E", (5, 0), (5, 0))
    result = evaluator(metadata).evaluate(
        item, (0, 0), cost_map=cost, static_obstacle_map=static,
        dynamic_obstacle_map=np.zeros_like(static), estimated_fire_map=fire,
        evaluated_at=1,
    )
    assert not result.reachable and result.rejection_reasons == (ExitRejectionReason.NO_PATH,)
    same, *_ = evaluate(Exit("S", (0, 0), (0, 0)))
    assert same.accepted and same.path_length_m == 0 and same.path_grid == ((0, 0),)


def test_diagonal_length_and_risk_cost_are_separate():
    diagonal, *_ = evaluate(Exit("D", (1, 1), (1, 1)))
    assert diagonal.path_length_m == pytest.approx(math.sqrt(2))
    metadata, fire, cost, static = environment()
    cost[:] = 2.0
    result = evaluator(metadata).evaluate(
        Exit("E", (2, 0), (2, 0)), (0, 0), cost_map=cost,
        static_obstacle_map=static,
        dynamic_obstacle_map=np.zeros_like(static),
        estimated_fire_map=fire, evaluated_at=1,
    )
    assert result.path_length_m == 2
    assert result.accumulated_risk_cost == pytest.approx(2.0)


@pytest.mark.parametrize("setting,value", [
    ("dangerous_accumulated_risk_cost", 3.0),
    ("dangerous_average_risk_cost", 1.5),
    ("dangerous_max_cell_risk_cost", 1.5),
])
def test_high_finite_path_cost_is_rejected_as_dangerous(setting, value):
    metadata, fire, cost, static = environment()
    cost[:] = 3.0
    result = evaluator(metadata, **{setting: value}).evaluate(
        Exit("E", (2, 0), (2, 0)), (0, 0), cost_map=cost,
        static_obstacle_map=static,
        dynamic_obstacle_map=np.zeros_like(static),
        estimated_fire_map=fire, evaluated_at=1,
    )
    assert result.reachable
    assert not result.accepted
    assert ExitRejectionReason.PATH_RISK_COST_EXCEEDED in result.rejection_reasons


@pytest.mark.parametrize("kind,value,reason", [
    ("temperature", 59.9, None),
    ("temperature", 60.0, ExitRejectionReason.TEMPERATURE_LIMIT_EXCEEDED),
    ("co", 1599.9, None),
    ("co", 1600.0, ExitRejectionReason.CO_LIMIT_EXCEEDED),
])
def test_path_fire_thresholds(kind, value, reason):
    kwargs = {kind: ((0, 1), value)}
    result, *_ = evaluate(Exit("E", (2, 0), (2, 0)), **kwargs)
    assert result.accepted is (reason is None)
    if reason:
        assert reason in result.rejection_reasons


def test_exit_neighborhood_risk_and_unknown_ratio_is_informational_only():
    item = Exit("E", (4, 0), (4, 0))
    hot, *_ = evaluate(item, temperature=((1, 4), 70))
    assert ExitRejectionReason.TEMPERATURE_LIMIT_EXCEEDED in hot.rejection_reasons
    gas, *_ = evaluate(item, co=((1, 4), 1700))
    assert ExitRejectionReason.CO_LIMIT_EXCEEDED in gas.rejection_reasons

    metadata, fire, cost, static = environment()
    fire.observed_mask[0, 1:4] = False
    result = evaluator(metadata).evaluate(
        item, (0, 0), cost_map=cost, static_obstacle_map=static,
        dynamic_obstacle_map=np.zeros_like(static), estimated_fire_map=fire,
        evaluated_at=1,
    )
    assert result.unknown_ratio == pytest.approx(3 / 5)
    assert result.accepted
    assert result.rejection_reasons == tuple()


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_invalid_cost_at_approach_rejected(invalid):
    result, *_ = evaluate(
        Exit("E", (2, 0), (2, 0)), cost=((0, 2), invalid)
    )
    assert result.rejection_reasons == (ExitRejectionReason.INVALID_COST,)


def test_config_validation():
    for kwargs in (
        {"exit_neighborhood_radius_m": -1},
        {"approach_search_radius_m": 0},
        {"usable_confirmation_distance_m": 0.0},
        {"dangerous_average_risk_cost": 0},
        {"dangerous_max_cell_risk_cost": np.inf},
    ):
        with pytest.raises(ValueError):
            ExitEvaluationConfig(**kwargs)
    with pytest.raises(ValueError):
        ExitEvaluationConfig.from_mapping({"temperature_block_threshold_c": 60})
    with pytest.raises(TypeError):
        ExitEvaluationConfig.from_mapping({"reject_blocked_exit": "yes"})
