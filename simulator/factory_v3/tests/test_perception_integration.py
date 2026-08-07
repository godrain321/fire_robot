import inspect
from pathlib import Path

import numpy as np
import yaml

from mapping.dynamic_obstacle_mapping import DynamicObstacleMappingConfig
from navigation.exit_blockage import ExitBlockageConfig
from planner.evacuation_planner import EvacuationPlanner, ExitSelectionConfig
from planner.exit_evaluator import ExitEvaluation
from visualization.partial_costmap_viewers import (
    MapOverlayConfig, PerceptionMapDisplayConfig, PygameSimulationViewer,
)
from world.entities import Exit, ExitStatus


class FakeEvaluator:
    def __init__(self, values):
        self.values = values

    def evaluate(self, exit_item, *args, **kwargs):
        return self.values[exit_item.exit_id]


def evaluation(exit_id, length, risk, status="usable"):
    return ExitEvaluation(
        exit_id, status, (0, 0), (0, 0), (0, 0), True, True,
        ((0, 0),), ((0, 0),), length, risk, 25, 10, 25, 10, 0,
        tuple(), 1,
    )


def test_blocked_replan_can_select_lowest_risk_while_normal_policy_stays_compatible():
    values = {
        "near": evaluation("near", 2, 10),
        "safe": evaluation("safe", 5, 2),
    }
    planner = EvacuationPlanner(FakeEvaluator(values), ExitSelectionConfig())
    exits = (Exit("near", (0, 0), (0, 0), ExitStatus.USABLE),
             Exit("safe", (0, 0), (0, 0), ExitStatus.USABLE))
    common = dict(
        cost_map=np.ones((1, 1)), static_obstacle_map=np.zeros((1, 1), bool),
        dynamic_obstacle_map=np.zeros((1, 1), bool),
        estimated_fire_map=None, created_at=1,
    )
    assert planner.plan(exits, (0, 0), **common).selected_exit_id == "near"
    decision = planner.plan(exits, (0, 0), risk_first=True, **common)
    assert decision.selected_exit_id == "safe"
    assert "lowest accumulated path risk" in decision.selection_reason


def test_factory_perception_configuration_loads_and_matches_planner_inflation():
    base = Path(__file__).resolve().parents[1]
    scenario = yaml.safe_load((base / "config/evacuation.yaml").read_text())
    dynamic = DynamicObstacleMappingConfig.from_mapping(
        scenario["dynamic_obstacle_mapping"]
    )
    ExitBlockageConfig.from_mapping(scenario["exit_blockage"])
    PerceptionMapDisplayConfig.from_mapping(scenario["perception_map_display"])
    MapOverlayConfig.from_mapping(scenario["map_overlays"])
    assert dynamic.obstacle_inflation_radius_m == scenario["planner"]["inflation_radius_m"]


def test_main_view_uses_robot_belief_and_mapper_has_no_ground_truth_input():
    import mapping.dynamic_obstacle_mapping as mapping_module

    draw_source = inspect.getsource(PygameSimulationViewer._draw_belief_cells)
    mapper_source = inspect.getsource(mapping_module)
    assert "belief.final_cost_map" in draw_source
    assert "GroundTruth" not in mapper_source
    assert "fds_result" not in mapper_source
