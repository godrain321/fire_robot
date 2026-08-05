import json

import numpy as np
import pytest

from navigation.travel_history import (
    TravelHistory, TravelHistoryConfig, TravelRecordReason,
)
from world.fire_maps import MapMetadata


def metadata():
    return MapMetadata(0, 5, 0, 5, 1, 6, 6, (0, 0))


def test_initial_duplicate_new_cell_and_axis_order():
    history = TravelHistory(metadata())
    initial = history.reset((0, 0), recorded_at=0)
    assert initial.reason is TravelRecordReason.INITIAL_POSITION
    assert initial.position_world == (0.0, 0.0)
    assert initial.position_grid == (0, 0)
    assert history.record_position((0, 0), recorded_at=1) is None
    point = history.record_position((0, 1), recorded_at=2)
    assert point.position_grid == (0, 1)  # (col,row), not NumPy [row,col]


def test_distance_and_direction_conditions_without_new_cell():
    distance_history = TravelHistory(metadata(), TravelHistoryConfig(
        record_on_new_grid_cell=False, min_record_distance_m=0.4,
    ))
    distance_history.reset((0, 0))
    assert distance_history.record_position((0.2, 0), recorded_at=1) is None
    point = distance_history.record_position((0.4, 0), recorded_at=2)
    assert point.reason is TravelRecordReason.DISTANCE_THRESHOLD

    direction_history = TravelHistory(metadata(), TravelHistoryConfig(
        record_on_new_grid_cell=False, min_record_distance_m=10,
        direction_change_threshold_deg=35,
    ))
    direction_history.reset((0, 0))
    assert direction_history.record_position((0.1, 0), recorded_at=1) is None
    point = direction_history.record_position((0.1, 0.1), recorded_at=2)
    assert point.reason is TravelRecordReason.DIRECTION_CHANGE


def test_total_distance_immutable_queries_return_mode_and_reset():
    history = TravelHistory(metadata())
    history.reset((0, 0))
    history.record_position((1, 0), recorded_at=1)
    points = history.get_points()
    assert isinstance(points, tuple)
    assert history.total_distance_m == pytest.approx(1.0)
    assert history.record_position((0, 0), recorded_at=2, is_returning=True) is None
    assert history.total_distance_m == pytest.approx(1.0)
    assert history.get_points() == points
    json.dumps(history.to_dict())
    history.clear()
    assert history.get_points() == ()
    assert history.total_distance_m == 0


def test_out_of_bounds_and_config_validation():
    history = TravelHistory(metadata())
    with pytest.raises(ValueError):
        history.reset((-1, 0))
    for kwargs in (
        {"min_record_distance_m": -1}, {"max_points": 0},
        {"direction_change_threshold_deg": 0},
        {"direction_change_threshold_deg": 181},
    ):
        with pytest.raises(ValueError):
            TravelHistoryConfig(**kwargs)


def test_max_points_preserves_start_and_reverse_does_not_mutate():
    history = TravelHistory(metadata(), TravelHistoryConfig(max_points=3))
    history.reset((0, 0))
    for col in range(1, 5):
        history.record_position((col, 0), recorded_at=col)
    before = history.get_points()
    reverse = history.build_reverse_path()
    assert len(before) == 3
    assert before[0].position_world == (0, 0)
    assert reverse == tuple(reversed(before))
    assert history.get_points() == before
