import numpy as np

from planner.a_star import unweighted_a_star, weighted_a_star


def test_unweighted_astar_ignores_finite_risk_costs_and_uses_shortest_path():
    costs = np.ones((3, 5), dtype=float)
    costs[1, 2] = 1000.0

    unweighted = unweighted_a_star(costs, (0, 1), (4, 1))
    weighted = weighted_a_star(costs, (0, 1), (4, 1))

    assert unweighted.path == [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
    assert (2, 1) not in weighted.path


def test_unweighted_astar_still_rejects_impassable_cells():
    costs = np.ones((3, 5), dtype=float)
    costs[:, 2] = np.inf

    result = unweighted_a_star(costs, (0, 1), (4, 1))

    assert not result.path
    assert result.reason == "no traversable unweighted A* path"


def test_unweighted_astar_succeeds_immediately_at_goal():
    result = unweighted_a_star(np.ones((2, 2)), (1, 1), (1, 1))

    assert result.path == [(1, 1)]
    assert result.total_cost == 0.0
