import numpy as np
import pytest

from visualization.partial_costmap_viewers import (
    MapOverlayConfig, PerceptionMapDisplayConfig, PygameSimulationViewer,
)
from world.entities import ExitStatus


def test_unknown_safe_caution_danger_blocked_colors_are_distinct():
    config = PerceptionMapDisplayConfig()
    assert config.color_for(observed=False, blocked=False, normalized_cost=0) == config.unknown_color
    assert config.color_for(observed=True, blocked=False, normalized_cost=0.1) == config.safe_color
    assert config.color_for(observed=True, blocked=False, normalized_cost=0.5) == config.caution_color
    assert config.color_for(observed=True, blocked=False, normalized_cost=0.8) == config.danger_color
    assert config.color_for(observed=True, blocked=True, normalized_cost=0) == config.blocked_color


def test_display_classification_does_not_mutate_costmap():
    costs = np.array([[1.0, np.inf], [2.0, 3.0]])
    original = costs.copy()
    config = PerceptionMapDisplayConfig()
    for value in costs.flat:
        config.color_for(
            observed=True, blocked=not np.isfinite(value),
            normalized_cost=0 if np.isfinite(value) else 1,
        )
    assert np.array_equal(costs, original, equal_nan=True)


@pytest.mark.parametrize("values", [
    {"unknown_color": [-1, 0, 0]},
    {"safe_color": [0, 0]},
    {"safe_cost_max": 0.7, "caution_cost_max": 0.6},
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
