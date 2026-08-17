import numpy as np

from navigation.evacuation_strategy_selector import (
    HazardKnowledgeState, HazardKnowledgeTracker,
)
from world import EstimatedFireMap, MapMetadata


def fire_map():
    metadata = MapMetadata(0, 2, 0, 2, 1, 3, 3, (0, 0))
    result = EstimatedFireMap(
        metadata, temperature_blocked_c=60, co_blocked_ppm=1600
    )
    result.replace_layers(
        np.full((3, 3), np.nan), np.full((3, 3), np.nan),
        np.zeros((3, 3), dtype=bool), np.full((3, 3), np.nan),
    )
    return result


def test_no_observation_means_no_fire_information():
    tracker = HazardKnowledgeTracker(
        temperature_elevated_c=35, co_elevated_ppm=100
    )
    decision = tracker.evaluate(fire_map(), evaluated_at=1)
    assert decision.state is HazardKnowledgeState.NO_FIRE_INFORMATION


def test_sensor_hazard_is_sticky_when_current_reading_later_falls():
    estimated = fire_map()
    tracker = HazardKnowledgeTracker(
        temperature_elevated_c=35, co_elevated_ppm=100
    )
    estimated.update_cell(1, 1, temperature_c=45, sim_time=1)
    first = tracker.evaluate(estimated, evaluated_at=1)
    estimated.update_cell(1, 1, temperature_c=20, sim_time=2)
    second = tracker.evaluate(estimated, evaluated_at=2)
    assert first.state is HazardKnowledgeState.FIRE_INFORMATION_AVAILABLE
    assert second.state is HazardKnowledgeState.FIRE_INFORMATION_AVAILABLE


def test_tracker_api_has_no_ground_truth_input():
    import inspect
    parameters = inspect.signature(HazardKnowledgeTracker.evaluate).parameters
    assert "ground_truth" not in parameters
