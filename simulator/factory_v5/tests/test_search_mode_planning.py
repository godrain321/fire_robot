import math

import numpy as np
import pytest

from mapping.partial_costmap import PartialCostmapConfig, PartialFireCostmap
from navigation.search_mode import (
    SearchPlanningConfig, confirmed_usable_exits, failed_recheck_exit_ids,
    frontier_mask,
    initialize_exit_recheck, plan_nearest_frontier,
    representative_exit_temperature_cost, usable_exit_at_pose,
)
from world.entities import Exit, ExitStatus


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


def test_recheck_initialization_skips_blocked_and_dangerous_exits():
    exits = (
        Exit("U", (0, 0), (0, 0), ExitStatus.USABLE),
        Exit("B", (0, 0), (0, 0), ExitStatus.BLOCKED,
             blocked_reason="fixture"),
        Exit("D", (0, 0), (0, 0), ExitStatus.DANGEROUS,
             danger_reason="fixture"),
        Exit("E", (0, 0), (0, 0), ExitStatus.DANGER_EXPECTED,
             danger_reason="fixture"),
    )
    pending, skipped = initialize_exit_recheck(exits)
    assert pending == {"U"}
    assert set(skipped) == {"B", "D", "E"}


def test_confirmed_usable_exits_excludes_unknown_and_is_deterministic():
    exits = (
        Exit("Z", (0, 0), (0, 0), ExitStatus.USABLE),
        Exit("U", (0, 0), (0, 0), ExitStatus.UNKNOWN),
        Exit("A", (0, 0), (0, 0), ExitStatus.USABLE),
    )
    assert tuple(item.exit_id for item in confirmed_usable_exits(exits)) == (
        "A", "Z",
    )


def test_failed_recheck_exits_are_deterministic_and_exclude_accepted():
    class Evaluation:
        def __init__(self, exit_id, accepted):
            self.exit_id = exit_id
            self.accepted = accepted

    assert failed_recheck_exit_ids((
        Evaluation("C", False), Evaluation("A", True),
        Evaluation("B", False),
    )) == ("B", "C")


def test_current_usable_exit_is_selected_without_requiring_another_path():
    exits = (
        Exit("FAR", (4, 0), (4, 0), ExitStatus.USABLE),
        Exit("HERE", (1, 1), (1, 1), ExitStatus.USABLE),
        Exit("BLOCKED", (1, 1), (1, 1), ExitStatus.BLOCKED,
             blocked_reason="fixture"),
    )
    selected = usable_exit_at_pose(
        exits, (1.2, 1.0), maximum_distance_m=0.5
    )
    assert selected is not None and selected.exit_id == "HERE"


def test_current_usable_exit_requires_completion_radius():
    exits = (Exit("FAR", (2, 0), (2, 0), ExitStatus.USABLE),)
    assert usable_exit_at_pose(
        exits, (0, 0), maximum_distance_m=1.0
    ) is None


@pytest.mark.parametrize("values", [
    {"temperature_block_c": 0.0},
    {"block_on_co": "false"},
    {"frontier_connectivity": 6},
    {"exit_temperature_radius_m": -1.0},
])
def test_invalid_search_configuration_is_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        SearchPlanningConfig.from_mapping(values)
