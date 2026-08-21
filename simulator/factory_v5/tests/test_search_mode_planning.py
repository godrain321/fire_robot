import math

import numpy as np
import pytest

from mapping.partial_costmap import PartialCostmapConfig, PartialFireCostmap
from navigation.search_mode import (
    SearchPlanningConfig, frontier_mask, plan_nearest_frontier,
    representative_exit_temperature_cost,
)


class TinyGrid:
    resolution = 1.0
    width = 7
    height = 5


def make_belief():
    return PartialFireCostmap(
        TinyGrid(), np.zeros((5, 7), dtype=bool), PartialCostmapConfig(
            grid_resolution=1.0, temperature_weight=8.0, co_weight=8.0,
        )
    )


def test_search_view_keeps_soft_co_cost_but_does_not_block_on_co():
    belief = make_belief()
    belief.co_observed_mask[2, 3] = True
    belief.co_belief_map[2, 3] = 1800.0
    belief.recalculate()
    assert math.isinf(belief.final_cost_map[2, 3])
    search = belief.planning_cost_map(
        temperature_blocked_c=80.0, block_on_co=False
    )
    assert np.isfinite(search[2, 3])
    assert search[2, 3] > belief.config.base_cost


def test_search_temperature_blocks_at_80_not_60_and_belief_is_unchanged():
    belief = make_belief()
    belief.temperature_observed_mask[2, 2:4] = True
    belief.temperature_belief_map[2, 2] = 70.0
    belief.temperature_belief_map[2, 3] = 80.0
    belief.recalculate()
    original = belief.final_cost_map.copy()
    search = belief.planning_cost_map(
        temperature_blocked_c=80.0, block_on_co=False
    )
    assert np.isfinite(search[2, 2])
    assert math.isinf(search[2, 3])
    np.testing.assert_array_equal(belief.final_cost_map, original)


def test_static_and_dynamic_obstacles_never_open_in_search_view():
    belief = make_belief()
    belief.static_obstacle_map[1, 1] = True
    belief.dynamic_inflated_obstacle_map[1, 2] = True
    belief.recalculate()
    search = belief.planning_cost_map(
        temperature_blocked_c=80.0, block_on_co=False
    )
    assert math.isinf(search[1, 1])
    assert math.isinf(search[1, 2])


def test_frontier_is_observed_free_boundary_not_unknown_cell():
    observed = np.zeros((5, 7), dtype=bool)
    observed[2, 1:4] = True
    costs = np.ones((5, 7), dtype=float)
    mask = frontier_mask(observed, costs)
    assert np.all(~mask[~observed])
    assert mask[2, 1] and mask[2, 2] and mask[2, 3]


def test_nearest_reachable_frontier_uses_weighted_astar_and_is_deterministic():
    observed = np.ones((5, 7), dtype=bool)
    observed[:, 6] = False
    observed[0, 0] = False
    costs = np.ones((5, 7), dtype=float)
    costs[1:4, 4] = np.inf
    first = plan_nearest_frontier(costs, observed, (1, 2))
    second = plan_nearest_frontier(costs, observed, (1, 2))
    assert first.success
    assert first == second
    assert observed[first.target_grid[1], first.target_grid[0]]
    assert np.isfinite(costs[first.target_grid[1], first.target_grid[0]])


def test_exit_temperature_uses_observed_neighborhood_mean():
    costs = np.zeros((5, 5), dtype=float)
    observed = np.zeros((5, 5), dtype=bool)
    costs[2, 2] = 8.0
    costs[2, 3] = 4.0
    observed[2, 2:4] = True
    assert representative_exit_temperature_cost(
        costs, observed, (2, 2), radius_cells=1
    ) == pytest.approx(6.0)


@pytest.mark.parametrize("values", [
    {"temperature_block_c": 0.0},
    {"block_on_co": "false"},
    {"frontier_connectivity": 6},
    {"exit_temperature_radius_m": -1.0},
])
def test_invalid_search_configuration_is_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        SearchPlanningConfig.from_mapping(values)
