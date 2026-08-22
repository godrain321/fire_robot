import json
import math

import numpy as np
import pytest

from navigation.exit_switching import (
    DelayedCostSwitch, ExitSwitchingConfig, RouteCostTrendMonitor,
    RouteTemperatureTrendMonitor,
    current_direction_world,
    evaluate_path_cost, is_opposite_direction,
)
from world import Exit, MapMetadata, WorldState


def test_single_cost_rise_does_not_switch_but_sustained_rise_does():
    config = ExitSwitchingConfig(
        evaluation_window=5, minimum_consecutive_increases=3,
        minimum_increase_ratio=0.10, minimum_absolute_increase=0.25,
    )
    monitor = RouteCostTrendMonitor(config)
    path = ((0, 0), (2, 0))
    decisions = []
    for revision, value in enumerate((1.0, 1.1, 1.2, 1.4), start=1):
        costs = np.full((3, 3), value)
        decisions.append(monitor.record(
            path, costs, revision=revision, evaluated_at=revision,
        ))
    assert not decisions[1].switch_required
    assert decisions[-1].switch_required
    assert decisions[-1].consecutive_increases == 3


def test_five_strictly_increasing_route_temperatures_trigger_switch():
    monitor = RouteTemperatureTrendMonitor(5)
    path = ((0, 0), (2, 0))
    decisions = [
        monitor.record(
            path, np.full((3, 3), value),
            revision=revision, evaluated_at=revision,
        )
        for revision, value in enumerate((20, 21, 22, 23, 24), start=1)
    ]
    assert not any(item.switch_required for item in decisions[:-1])
    assert decisions[-1].switch_required
    assert decisions[-1].consecutive_increases == 4


def test_temperature_plateau_breaks_five_sample_increase():
    monitor = RouteTemperatureTrendMonitor(5)
    path = ((0, 0), (2, 0))
    decision = None
    for revision, value in enumerate((20, 21, 21, 22, 23), start=1):
        decision = monitor.record(
            path, np.full((3, 3), value),
            revision=revision, evaluated_at=revision,
        )
    assert decision is not None and not decision.switch_required


def test_same_revision_is_not_counted_twice_and_reset_clears_trend():
    monitor = RouteCostTrendMonitor(ExitSwitchingConfig())
    path = ((0, 0), (1, 0))
    costs = np.ones((2, 2))
    monitor.record(path, costs, revision=1, evaluated_at=1)
    monitor.record(path, costs * 2, revision=1, evaluated_at=2)
    assert len(monitor.samples) == 1
    monitor.reset(2.0)
    assert monitor.samples == tuple()


def test_path_cost_checks_all_touched_cells_and_invalid_cost():
    costs = np.ones((3, 3))
    costs[0, 1] = 4
    accumulated, average, maximum = evaluate_path_cost(
        ((0, 0), (2, 0)), costs
    )
    assert (accumulated, average, maximum) == (6.0, 2.0, 4.0)
    costs[0, 1] = np.inf
    assert evaluate_path_cost(((0, 0), (2, 0)), costs) is None


def test_direction_priority_and_opposite_world_half_plane():
    direction = current_direction_world(
        (0, 0), (1, 0), ((0, 0), (0, 1)), math.pi,
    )
    assert direction == pytest.approx((1, 0))
    assert is_opposite_direction(
        direction, (0, 0), (-1, 0), minimum_difference_deg=90
    )
    assert not is_opposite_direction(
        direction, (0, 0), (1, 0), minimum_difference_deg=90
    )


def test_switch_config_validation():
    invalid = (
        {"evaluation_window": 0},
        {"minimum_consecutive_increases": 0},
        {"evaluation_window": 3, "minimum_consecutive_increases": 3},
        {"minimum_increase_ratio": -0.1},
        {"minimum_direction_difference_deg": 181},
        {"switch_cooldown_sec": -1},
        {"additional_travel_before_switch_m": -0.1},
        {"enabled": "yes"},
    )
    for values in invalid:
        with pytest.raises((TypeError, ValueError)):
            ExitSwitchingConfig(**values)


def test_world_records_selection_cost_and_switch_serializably():
    metadata = MapMetadata(0, 2, 0, 2, 1, 3, 3, (0, 0))
    world = WorldState(metadata, np.zeros((3, 3), dtype=bool))
    world.add_exit(Exit("A", (0, 0), (0, 0)))
    world.add_exit(Exit("B", (2, 0), (2, 0)))
    world.record_exit_selection(
        "A", reason="shortest", costmap_revision=2,
        path_lengths_m={"A": 1, "B": 3},
    )
    monitor = RouteCostTrendMonitor(ExitSwitchingConfig())
    monitor.record(((0, 0), (1, 0)), np.ones((3, 3)), revision=2, evaluated_at=1)
    world.record_route_cost(
        monitor.samples[-1], baseline=1.0, consecutive=0
    )
    world.record_exit_switch(
        previous_exit_id="A", new_exit_id="B", reason="cost",
        sim_time=5, cooldown_seconds=10, validation_result="validated",
    )
    assert world.exit_switch_is_cooling_down(14.9)
    assert not world.exit_switch_is_cooling_down(15)
    json.dumps(world.to_dict())


def test_soft_cost_switch_waits_for_one_metre_of_actual_travel():
    delay = DelayedCostSwitch(1.0)
    delay.arm("EXIT2", "sustained_route_cost_increase", 3.0)

    assert not delay.ready(3.6)
    assert delay.travelled_distance(3.6) == pytest.approx(0.6)
    assert delay.ready(4.0)
    assert delay.exit_id == "EXIT2"
    delay.clear()
    assert not delay.active
