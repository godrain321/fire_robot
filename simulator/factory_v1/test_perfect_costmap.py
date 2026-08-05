"""Focused verification tests for static weighted costmap planning."""

import math

import numpy as np

from planner.a_star import weighted_a_star, weighted_a_star_with_escape


def test_low_risk_map_uses_geometrically_short_path():
    costs = np.ones((7, 7), dtype=float)
    result = weighted_a_star(costs, (0, 0), (6, 6))

    assert result.path == [(i, i) for i in range(7)]
    assert math.isclose(result.total_cost, 6.0 * math.sqrt(2.0))


def test_high_finite_risk_causes_safer_detour():
    costs = np.ones((7, 7), dtype=float)
    for index in range(1, 6):
        costs[index, index] = 100.0

    result = weighted_a_star(costs, (0, 0), (6, 6))

    assert result.path
    assert all(costs[gy, gx] == 1.0 for gx, gy in result.path)
    assert len(result.path) > 7


def test_temperature_blocked_cells_are_not_crossed():
    temperature = np.full((5, 7), 20.0)
    temperature[:, 3] = 60.0
    temperature[4, 3] = 20.0
    costs = np.ones_like(temperature)
    costs[temperature >= 60.0] = np.inf

    result = weighted_a_star(costs, (0, 2), (6, 2))

    assert result.path
    assert all(temperature[gy, gx] < 60.0 for gx, gy in result.path)


def test_co_blocked_cells_are_not_crossed():
    co_ppm = np.zeros((5, 7))
    co_ppm[:, 3] = 1600.0
    co_ppm[0, 3] = 0.0
    costs = np.ones_like(co_ppm)
    costs[co_ppm >= 1600.0] = np.inf

    result = weighted_a_star(costs, (0, 2), (6, 2))

    assert result.path
    assert all(co_ppm[gy, gx] < 1600.0 for gx, gy in result.path)


def test_fully_blocked_exit_returns_no_path():
    costs = np.ones((5, 5), dtype=float)
    goal = (2, 2)
    for node in ((1, 1), (2, 1), (3, 1), (1, 2), (3, 2),
                 (1, 3), (2, 3), (3, 3)):
        costs[node[1], node[0]] = np.inf

    result = weighted_a_star(costs, (0, 0), goal)

    assert result.path == []
    assert math.isinf(result.total_cost)
    assert "no traversable path" in result.reason


def test_diagonal_corner_cutting_is_rejected():
    costs = np.ones((2, 2), dtype=float)
    costs[0, 1] = np.inf
    costs[1, 0] = np.inf

    result = weighted_a_star(costs, (0, 0), (1, 1))

    assert result.path == []
    assert "no traversable path" in result.reason


def test_infinite_cost_start_escapes_then_replans():
    costs = np.ones((5, 7), dtype=float)
    costs[2, 1:3] = np.inf
    static = np.zeros_like(costs, dtype=bool)

    result = weighted_a_star_with_escape(costs, (1, 2), (6, 2), static)

    assert result.path[0] == (1, 2)
    assert result.path[-1] == (6, 2)
    assert result.escape_path
    assert result.replan_start is not None
    assert np.isfinite(costs[result.replan_start[1], result.replan_start[0]])
    assert "after escaping" in result.reason


def test_escape_never_crosses_static_obstacle():
    costs = np.full((3, 3), np.inf)
    costs[1, 2] = 1.0
    static = np.zeros_like(costs, dtype=bool)
    static[:, 1] = True

    result = weighted_a_star_with_escape(costs, (0, 1), (2, 1), static)

    assert result.path == []
    assert "no finite-cost cell reachable" in result.reason
