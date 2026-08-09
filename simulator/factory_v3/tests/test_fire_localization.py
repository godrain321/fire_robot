import json
import math

import numpy as np
import pytest

from mapping.fire_localization import (
    FireEstimateState, FireLocalizationConfig, FireLocalizer,
)
from sensors.thermal_camera import ThermalRayObservation, ThermalRaySample
from world.fire_maps import MapMetadata


def metadata():
    return MapMetadata(0, 6, 0, 6, 1, 7, 7, (0, 0))


def config(**changes):
    values = dict(
        prior_probability=0.05,
        possible_probability_threshold=0.20,
        likely_probability_threshold=0.40,
        confirm_probability_threshold=0.60,
        release_probability_threshold=0.50,
        candidate_probability_threshold=0.15,
        minimum_confirmation_observations=3,
        minimum_distinct_observation_poses=2,
        maximum_confirmed_region_cells=20,
        minimum_consecutive_co_rises=2,
        evidence_decay_per_second=0.0,
    )
    values.update(changes)
    return FireLocalizationConfig(**values)


def ray(cells, temperature=80.0, row=0, col=0):
    samples = tuple(
        ThermalRaySample((x, y, 0), (float(x), float(y), 0.3), i + 1, temperature)
        for i, (x, y) in enumerate(cells)
    )
    return ThermalRayObservation(
        row, col, temperature, samples, samples[-1].world_position,
        samples[-1].grid_position, samples[-1].distance, True, False,
    )


def test_heading_rotates_thermal_ray_and_obstacle_trace_stops_evidence():
    localizer = FireLocalizer(metadata(), np.zeros((7, 7), bool), config())
    localizer.add_thermal_observation(
        "t1", 1.0, (1, 1, 0), (ray([(2, 1), (3, 1)]),)
    )
    assert localizer.thermal_fire_evidence[1, 2] > 0
    assert localizer.thermal_fire_evidence[1, 3] > 0
    assert localizer.thermal_fire_evidence[1, 0] == 0
    # The camera trace ends at the obstacle; a cell behind it is absent and
    # therefore receives no direct thermal evidence.
    assert localizer.thermal_fire_evidence[1, 4] == 0
    assert localizer.latest_result.state is not FireEstimateState.CONFIRMED_FIRE_REGION


def test_crossing_repeated_rays_strengthen_common_region_and_can_confirm():
    localizer = FireLocalizer(metadata(), np.zeros((7, 7), bool), config())
    observations = (
        ((0, 3, 0.0), [(1, 3), (2, 3), (3, 3)]),
        ((3, 0, math.pi / 2), [(3, 1), (3, 2), (3, 3)]),
        ((6, 3, math.pi), [(5, 3), (4, 3), (3, 3)]),
        ((3, 6, -math.pi / 2), [(3, 5), (3, 4), (3, 3)]),
        ((0, 3, 0.0), [(1, 3), (2, 3), (3, 3)]),
        ((3, 0, math.pi / 2), [(3, 1), (3, 2), (3, 3)]),
    )
    for index, (pose, cells) in enumerate(observations):
        localizer.add_thermal_observation(
            f"t{index}", float(index + 1), pose, (ray(cells),)
        )
    result = localizer.latest_result
    assert result.highest_probability_grid == (3, 3)
    assert result.distinct_pose_count >= 2
    assert result.valid_observation_count >= 3
    assert result.confirmed
    assert result.state is FireEstimateState.CONFIRMED_FIRE_REGION
    json.dumps(localizer.to_dict(), allow_nan=True)


def test_duplicate_id_reverse_time_and_invalid_sensor_values_are_rejected():
    localizer = FireLocalizer(metadata(), np.zeros((7, 7), bool), config())
    localizer.add_thermal_observation("same", 2, (1, 1, 0), (ray([(2, 1)]),))
    with pytest.raises(ValueError, match="duplicate"):
        localizer.add_thermal_observation("same", 2, (1, 1, 0), ())
    with pytest.raises(ValueError, match="backwards"):
        localizer.add_thermal_observation("old", 1, (1, 1, 0), ())
    other = FireLocalizer(metadata(), np.zeros((7, 7), bool), config())
    with pytest.raises(ValueError, match="finite"):
        other.add_co_observation("co", 1, (1, 1, 0), np.nan, 25)


def test_co_requires_motion_repeated_rise_and_thermal_support():
    localizer = FireLocalizer(metadata(), np.zeros((7, 7), bool), config())
    localizer.add_co_observation("c0", 0, (1, 3, 0), 10, 25)
    localizer.add_co_observation("still", 0.5, (1, 3, 0), 30, 30, thermal_direction_supported=True)
    assert not np.any(localizer.co_gradient_evidence)
    localizer.add_co_observation("c1", 1, (2, 3, 0), 40, 33, thermal_direction_supported=True)
    assert not np.any(localizer.co_gradient_evidence)  # first qualifying rise
    localizer.add_co_observation("c2", 2, (3, 3, 0), 55, 36, thermal_direction_supported=True)
    assert localizer.co_gradient_evidence[3, 4] > 0
    assert localizer.co_gradient_evidence[3, 2] == 0


def test_co_direction_does_not_cross_wall():
    static = np.zeros((7, 7), bool)
    static[3, 4] = True
    localizer = FireLocalizer(metadata(), static, config(minimum_consecutive_co_rises=1))
    localizer.add_co_observation("c0", 0, (1, 3, 0), 10, 25)
    localizer.add_co_observation("c1", 1, (2, 3, 0), 30, 30, thermal_direction_supported=True)
    assert localizer.co_gradient_evidence[3, 3] > 0
    assert localizer.co_gradient_evidence[3, 5] == 0


def test_co_only_trend_is_weaker_than_thermally_supported_trend():
    weak = FireLocalizer(metadata(), np.zeros((7, 7), bool), config(minimum_consecutive_co_rises=1))
    strong = FireLocalizer(metadata(), np.zeros((7, 7), bool), config(minimum_consecutive_co_rises=1))
    for item in (weak, strong):
        item.add_co_observation("base", 0, (1, 3, 0), 10, 25)
    weak.add_co_observation("rise", 1, (2, 3, 0), 30, 25)
    strong.add_co_observation(
        "rise", 1, (2, 3, 0), 30, 28, thermal_direction_supported=True
    )
    assert 0 < weak.co_gradient_evidence[3, 3] < strong.co_gradient_evidence[3, 3]


def test_decay_uses_elapsed_time_and_never_drops_unobserved_below_prior():
    cfg = config(evidence_decay_per_second=0.1)
    localizer = FireLocalizer(metadata(), np.zeros((7, 7), bool), cfg)
    localizer.add_thermal_observation("t0", 0, (1, 1, 0), (ray([(2, 1)]),))
    before = localizer.thermal_fire_evidence[1, 2]
    localizer.add_thermal_observation("empty", 10, (1, 1, 0), ())
    assert localizer.thermal_fire_evidence[1, 2] == pytest.approx(before * math.exp(-1))
    assert localizer.fire_probability[6, 6] == pytest.approx(cfg.prior_probability)


@pytest.mark.parametrize("changes", [
    {"prior_probability": -0.1},
    {"release_probability_threshold": 0.9},
    {"minimum_robot_motion_m": 0},
    {"evidence_decay_per_second": -1},
    {"minimum_confirmation_observations": 0},
])
def test_invalid_config_rejected(changes):
    with pytest.raises(ValueError):
        config(**changes)
