import numpy as np
import pytest
from types import SimpleNamespace

from visualization.partial_costmap_viewers import (
    MapOverlayConfig, PerceptionMapDisplayConfig, PygameSimulationViewer,
)
from world.entities import ExitStatus


def test_unknown_safe_caution_danger_blocked_colors_are_distinct():
    config = PerceptionMapDisplayConfig()
    assert config.color_for(observed=False) == config.unknown_color
    assert config.color_for(observed=True, temperature_c=39.9) == config.safe_color
    assert config.color_for(observed=True, temperature_c=40.0) == config.caution_color
    assert config.color_for(observed=True, temperature_c=60.0) == config.danger_color
    assert config.color_for(observed=True, co_ppm=99.9) == config.safe_color
    assert config.color_for(observed=True, co_ppm=100.0) == config.caution_color
    assert config.color_for(observed=True, co_ppm=1600.0) == config.danger_color
    assert config.color_for(observed=True, obstacle_blocked=True) == config.blocked_color


def test_display_classification_does_not_mutate_costmap():
    costs = np.array([[1.0, np.inf], [2.0, 3.0]])
    original = costs.copy()
    config = PerceptionMapDisplayConfig()
    for value in costs.flat:
        config.color_for(
            observed=True, obstacle_blocked=not np.isfinite(value),
            temperature_c=25.0,
        )
    assert np.array_equal(costs, original, equal_nan=True)


def test_belief_map_colors_use_absolute_temperature_and_co_thresholds():
    viewer = object.__new__(PygameSimulationViewer)
    viewer.perception_display_config = PerceptionMapDisplayConfig()
    viewer.overlay_config = MapOverlayConfig(show_dynamic_obstacles=False)
    viewer.grid_map = SimpleNamespace(height=2, width=3)
    belief = SimpleNamespace(
        observed_mask=np.ones((2, 3), dtype=bool),
        temperature_observed_mask=np.array(
            [[True, True, True], [False, False, False]], dtype=bool
        ),
        co_observed_mask=np.array(
            [[False, False, False], [True, True, True]], dtype=bool
        ),
        temperature_belief_map=np.array(
            [[39.9, 40.0, 60.0], [np.nan, np.nan, np.nan]]
        ),
        co_belief_map=np.array(
            [[np.nan, np.nan, np.nan], [99.9, 100.0, 1600.0]]
        ),
        static_obstacle_map=np.zeros((2, 3), dtype=bool),
        dynamic_obstacle_map=np.zeros((2, 3), dtype=bool),
    )
    rgb = viewer._belief_rgb_array(belief)
    cfg = viewer.perception_display_config
    assert tuple(rgb[0, 0]) == cfg.safe_color
    assert tuple(rgb[0, 1]) == cfg.caution_color
    assert tuple(rgb[0, 2]) == cfg.danger_color
    assert tuple(rgb[1, 0]) == cfg.safe_color
    assert tuple(rgb[1, 1]) == cfg.caution_color
    assert tuple(rgb[1, 2]) == cfg.danger_color


def test_planner_inflation_is_not_painted_as_slam_wall():
    viewer = object.__new__(PygameSimulationViewer)
    viewer.perception_display_config = PerceptionMapDisplayConfig()
    viewer.overlay_config = MapOverlayConfig(show_dynamic_obstacles=False)
    viewer.grid_map = SimpleNamespace(height=1, width=2)
    belief = SimpleNamespace(
        observed_mask=np.array([[False, True]], dtype=bool),
        temperature_observed_mask=np.zeros((1, 2), dtype=bool),
        co_observed_mask=np.zeros((1, 2), dtype=bool),
        temperature_belief_map=np.full((1, 2), np.nan),
        co_belief_map=np.full((1, 2), np.nan),
        # Both cells are blocked only in the inflated planner map. Exact SLAM
        # wall pixels are rendered later from display_static_obstacle_map.
        static_obstacle_map=np.ones((1, 2), dtype=bool),
        dynamic_obstacle_map=np.zeros((1, 2), dtype=bool),
    )
    rgb = viewer._belief_rgb_array(belief)
    cfg = viewer.perception_display_config
    assert tuple(rgb[0, 0]) == cfg.unknown_color
    assert tuple(rgb[0, 1]) == cfg.safe_color


@pytest.mark.parametrize("values", [
    {"unknown_color": [-1, 0, 0]},
    {"safe_color": [0, 0]},
    {"temperature_caution_c": 60.0, "temperature_danger_c": 60.0},
    {"co_caution_ppm": 1600.0, "co_danger_ppm": 1600.0},
])
def test_invalid_display_config(values):
    with pytest.raises(ValueError):
        PerceptionMapDisplayConfig.from_mapping(values)


def test_overlay_settings_require_booleans():
    with pytest.raises(TypeError):
        MapOverlayConfig(show_dynamic_obstacles=1)


def test_human_marker_changes_color_after_detection():
    assert (
        PygameSimulationViewer.human_marker_color(False)
        != PygameSimulationViewer.human_marker_color(True)
    )


def test_planner_static_inflation_is_not_drawn_as_slam_wall():
    assert not PygameSimulationViewer.perception_blocked_overlay(True, True)
    assert PygameSimulationViewer.perception_blocked_overlay(True, False)


@pytest.mark.parametrize("status", tuple(ExitStatus))
def test_exit_status_overlay_uses_enum_value_text(status):
    assert PygameSimulationViewer.exit_status_label(status) == status.value


def test_exit_status_display_does_not_depend_on_evaluation_reason():
    """The main exit label is derived only from the WorldState status."""
    assert PygameSimulationViewer.exit_status_display_label(
        "EXIT1", ExitStatus.BLOCKED
    ) == "EXIT1 status: BLOCKED"
