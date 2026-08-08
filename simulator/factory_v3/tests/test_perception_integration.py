import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from mapping.dynamic_obstacle_mapping import DynamicObstacleMappingConfig
from mapping.grid_map import GridMap
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

    draw_source = inspect.getsource(PygameSimulationViewer._belief_rgb_array)
    mapper_source = inspect.getsource(mapping_module)
    assert "belief.temperature_belief_map" in draw_source
    assert "belief.co_belief_map" in draw_source
    assert "finite.max" not in draw_source
    assert "GroundTruth" not in mapper_source
    assert "fds_result" not in mapper_source


def test_dynamic_overlay_draws_only_directly_observed_obstacle_cells():
    draw_source = inspect.getsource(PygameSimulationViewer._belief_rgb_array)
    assert "belief.dynamic_obstacle_map" in draw_source
    assert "belief.dynamic_inflated_obstacle_map" not in draw_source


def test_slam_and_costmap_surfaces_are_cached_by_revision(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    grid = GridMap((0, 1, 0, 1, 0, 1), [], [], 0.5, 0)
    config = SimpleNamespace(base_cost=1.0, simulation_dt=0.1)
    viewer = PygameSimulationViewer(grid, config)
    shape = (grid.height, grid.width)
    belief = SimpleNamespace(
        revision=0,
        observed_mask=np.zeros(shape, dtype=bool),
        temperature_observed_mask=np.zeros(shape, dtype=bool),
        co_observed_mask=np.zeros(shape, dtype=bool),
        temperature_belief_map=np.full(shape, np.nan),
        co_belief_map=np.full(shape, np.nan),
        static_obstacle_map=np.zeros(shape, dtype=bool),
        dynamic_obstacle_map=np.zeros(shape, dtype=bool),
    )
    static_surface = viewer._static_slam_surface
    viewer._draw_belief_cells(belief)
    first_cost_surface = viewer._costmap_surface
    assert viewer._costmap_surface_build_count == 1
    viewer._draw_belief_cells(belief)
    assert viewer._costmap_surface is first_cost_surface
    assert viewer._static_slam_surface is static_surface
    assert viewer._costmap_surface_build_count == 1
    belief.revision = 1
    viewer._draw_belief_cells(belief)
    assert viewer._costmap_surface is not first_cost_surface
    assert viewer._costmap_surface_build_count == 2
    assert viewer._cached_text("same", viewer.small_font, (1, 2, 3)) is viewer._cached_text(
        "same", viewer.small_font, (1, 2, 3)
    )
    first_points = viewer._cached_screen_points("route", ((0, 0), (1, 1)))
    assert first_points is viewer._cached_screen_points(
        "route", ((0, 0), (1, 1))
    )
    viewer.close()


def test_render_interpolates_frames_without_extra_simulation_updates(
    monkeypatch,
):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    grid = GridMap((0, 1, 0, 1, 0, 1), [], [], 0.5, 0)
    viewer = PygameSimulationViewer(
        grid, SimpleNamespace(
            base_cost=1.0, simulation_dt=0.1, render_fps=30,
        )
    )
    rendered = []
    monkeypatch.setattr(
        viewer, "_draw_once",
        lambda *args, **kwargs: rendered.append(args[1]),
    )
    monkeypatch.setattr(viewer.pygame.display, "flip", lambda: None)
    viewer._last_render_pose = (0.0, 0.0, 0.0)
    viewer.draw(None, SimpleNamespace(x=0.9, y=0.0, theta=0.0))
    assert len(rendered) == 3
    assert [item.x for item in rendered] == pytest.approx((0.3, 0.6, 0.9))
    assert viewer._last_render_pose == (0.9, 0.0, 0.0)
    viewer.close()
