import json

import numpy as np
import pytest

from navigation.event_replanning import (
    EventReplanningConfig,
    EventReplanningPolicy,
    ReplanPriority,
    ReplanReason,
)
from world.entities import ExitStatus


def policy(**changes):
    return EventReplanningPolicy(EventReplanningConfig(**changes))


def evaluate(item, **changes):
    values = dict(
        current_path=((0, 0), (1, 0), (2, 0)),
        current_costmap=np.ones((4, 4)), costmap_revision=1,
        robot_pose=(0.0, 0.0), elapsed_time=1.0,
    )
    values.update(changes)
    return item.evaluate(**values)


def test_no_change_and_unrelated_obstacle_do_not_replan():
    item = policy(periodic_enabled=False)
    assert not evaluate(item).required
    obstacles = np.zeros((4, 4), dtype=bool)
    obstacles[3, 3] = True
    assert not evaluate(item, dynamic_obstacle_map=obstacles).required


def test_dynamic_obstacle_on_remaining_path_has_highest_priority():
    item = policy(periodic_enabled=False)
    obstacles = np.zeros((4, 4), dtype=bool)
    obstacles[0, 1] = True
    decision = evaluate(
        item, dynamic_obstacle_map=obstacles,
        exit_statuses={"E": ExitStatus.BLOCKED}, current_exit_id="E",
    )
    assert decision.reason is ReplanReason.DYNAMIC_OBSTACLE_ON_PATH
    assert decision.immediate_stop and decision.invalidate_current_path
    assert decision.affected_cell_grid == (1, 0)


def test_already_traversed_obstacle_is_excluded_by_remaining_path_input():
    item = policy(periodic_enabled=False)
    obstacles = np.zeros((4, 4), dtype=bool)
    obstacles[0, 0] = True
    decision = evaluate(
        item, current_path=((1, 0), (2, 0)), dynamic_obstacle_map=obstacles
    )
    assert not decision.required


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_invalid_path_cost_requires_replan(value):
    item = policy(periodic_enabled=False)
    costs = np.ones((4, 4))
    costs[0, 2] = value
    assert evaluate(item, current_costmap=costs).reason is ReplanReason.PATH_CELL_BLOCKED


def test_nan_sensor_observation_is_ignored():
    item = policy(periodic_enabled=False)
    temp = np.full((4, 4), np.nan)
    seen = np.ones((4, 4), dtype=bool)
    assert not evaluate(
        item, temperature_map=temp, temperature_observed_mask=seen
    ).required


@pytest.mark.parametrize(
    "layer,mask,reason",
    [
        ("temperature_map", "temperature_observed_mask", ReplanReason.PATH_TEMPERATURE_BLOCKED),
        ("co_map", "co_observed_mask", ReplanReason.PATH_CO_BLOCKED),
    ],
)
def test_observed_hard_hazard_blocks_immediately(layer, mask, reason):
    item = policy(periodic_enabled=False)
    values = np.zeros((4, 4))
    values[0, 1] = 60.0 if layer.startswith("temperature") else 1600.0
    seen = np.zeros((4, 4), dtype=bool)
    seen[0, 1] = True
    assert evaluate(item, **{layer: values, mask: seen}).reason is reason


def test_unobserved_high_value_is_not_used_as_sensor_evidence():
    item = policy(periodic_enabled=False)
    temp = np.full((4, 4), 100.0)
    seen = np.zeros((4, 4), dtype=bool)
    assert not evaluate(
        item, temperature_map=temp, temperature_observed_mask=seen
    ).required


def test_temperature_hysteresis_releases_only_after_confirmations():
    item = policy(periodic_enabled=False, release_confirmation_observations=3)
    temp = np.zeros((4, 4))
    seen = np.zeros((4, 4), dtype=bool)
    seen[0, 1] = True
    temp[0, 1] = 60
    assert evaluate(
        item, temperature_map=temp, temperature_observed_mask=seen,
        costmap_revision=1,
    ).required
    temp[0, 1] = 54
    assert evaluate(
        item, temperature_map=temp, temperature_observed_mask=seen,
        costmap_revision=2,
    ).required
    assert evaluate(
        item, temperature_map=temp, temperature_observed_mask=seen,
        costmap_revision=3,
    ).required
    assert not evaluate(
        item, temperature_map=temp, temperature_observed_mask=seen,
        costmap_revision=4,
    ).required


def test_temperature_between_release_and_block_keeps_latch():
    item = policy(periodic_enabled=False)
    temp = np.zeros((4, 4))
    seen = np.zeros((4, 4), dtype=bool)
    seen[0, 1] = True
    temp[0, 1] = 60
    evaluate(item, temperature_map=temp, temperature_observed_mask=seen)
    temp[0, 1] = 59
    assert evaluate(
        item, temperature_map=temp, temperature_observed_mask=seen,
        costmap_revision=2,
    ).reason is ReplanReason.PATH_TEMPERATURE_BLOCKED


@pytest.mark.parametrize(
    "status,reason",
    [
        (ExitStatus.BLOCKED, ReplanReason.EXIT_BLOCKED),
        (ExitStatus.DANGEROUS, ReplanReason.EXIT_UNSAFE_FIRE),
    ],
)
def test_invalid_current_exit_is_rejected(status, reason):
    item = policy(periodic_enabled=False)
    result = evaluate(
        item, exit_statuses={"E": status}, current_exit_id="E"
    )
    assert result.reason is reason
    assert result.priority is ReplanPriority.EXIT_INVALID


def test_safer_exit_requires_minimum_improvement_and_uses_stable_tie_break():
    item = policy(periodic_enabled=False, minimum_cost_improvement_ratio=0.15)
    statuses = {name: ExitStatus.USABLE for name in ("A", "B", "C")}
    small = evaluate(
        item, current_exit_id="A", current_exit_cost=100,
        alternative_exit_costs={"B": 90}, exit_statuses=statuses,
    )
    assert not small.required
    better = evaluate(
        item, current_exit_id="A", current_exit_cost=100,
        alternative_exit_costs={"C": 70, "B": 70}, exit_statuses=statuses,
    )
    assert better.reason is ReplanReason.SAFER_EXIT_AVAILABLE
    assert better.alternative_exit_id == "B"


def test_blocked_alternative_is_never_selected():
    item = policy(periodic_enabled=False)
    result = evaluate(
        item, current_exit_id="A", current_exit_cost=100,
        alternative_exit_costs={"B": 1},
        exit_statuses={"A": ExitStatus.USABLE, "B": ExitStatus.BLOCKED},
    )
    assert not result.required


def test_periodic_and_distance_reevaluation_are_low_priority():
    item = policy(periodic_interval_s=5, periodic_travel_distance_m=2)
    evaluate(item, elapsed_time=0)
    assert evaluate(item, elapsed_time=5).reason is ReplanReason.PERIODIC_REEVALUATION
    item.mark_reevaluation_complete(
        elapsed_time=5, robot_pose=(0, 0), costmap_revision=1
    )
    assert evaluate(
        item, elapsed_time=5.3, robot_pose=(2.0, 0)
    ).reason is ReplanReason.DISTANCE_REEVALUATION


def test_periodic_can_be_disabled():
    item = policy(periodic_enabled=False)
    assert not evaluate(item, elapsed_time=999, robot_pose=(99, 0)).required


def test_follow_distance_requests_follow_wait_not_path_invalidation():
    item = policy(periodic_enabled=False, maximum_follow_distance_m=2.0)
    decision = evaluate(
        item, victim_follow_active=True, victim_follow_distance_m=2.1
    )
    assert decision.reason is ReplanReason.VICTIM_FOLLOW_FAILURE
    assert decision.immediate_stop and not decision.invalidate_current_path
    assert decision.detail == "follow_wait_before_replan"


def test_follow_path_block_requests_invalidation():
    item = policy(periodic_enabled=False)
    decision = evaluate(
        item, victim_follow_active=True, victim_path_blocked=True
    )
    assert decision.invalidate_current_path


def test_same_reason_and_revision_is_suppressed_after_processing():
    item = policy(periodic_enabled=False)
    costs = np.ones((4, 4))
    costs[0, 1] = np.inf
    first = evaluate(item, current_costmap=costs, costmap_revision=7)
    item.mark_processed(
        first, costmap_revision=7, elapsed_time=1, robot_pose=(0, 0),
        selected_exit_id="E",
    )
    assert not evaluate(
        item, current_costmap=costs, costmap_revision=7, elapsed_time=1.1
    ).required
    assert evaluate(
        item, current_costmap=costs, costmap_revision=8, elapsed_time=1.2
    ).required


def test_decision_is_json_serializable():
    item = policy(periodic_enabled=False)
    costs = np.ones((4, 4))
    costs[0, 1] = np.inf
    json.dumps(evaluate(item, current_costmap=costs).to_dict())


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature_release_c": 60.0},
        {"co_release_ppm": 1600.0},
        {"minimum_cost_improvement_ratio": -0.1},
        {"periodic_interval_s": 0.0},
        {"maximum_follow_distance_m": 0.0},
        {"release_confirmation_observations": 0},
    ],
)
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        EventReplanningConfig(**changes)


def test_nested_yaml_mapping_is_strictly_validated():
    config = EventReplanningConfig.from_mapping({
        "enabled": True,
        "max_replan_attempts": 5,
        "cooldown_seconds": 0.5,
        "periodic_reevaluation": {
            "enabled": True, "interval_s": 5.0, "travel_distance_m": 2.0,
        },
        "hysteresis": {
            "temperature_block_c": 60, "temperature_release_c": 55,
            "co_block_ppm": 1600, "co_release_ppm": 1400,
            "block_confirmation_observations": 2,
            "release_confirmation_observations": 3,
        },
    })
    assert config.temperature_release_c == 55
    with pytest.raises(ValueError):
        EventReplanningConfig.from_mapping({"periodic_reevaluation": {"wat": 1}})
