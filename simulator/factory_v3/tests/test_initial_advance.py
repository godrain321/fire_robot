import math

import pytest

from navigation.initial_advance import InitialAdvanceConfig


def test_configured_distance_moves_along_initial_yaw():
    config = InitialAdvanceConfig.from_robot_motion({
        "initial_forward_distance_m": 0.4,
    })
    assert config.target_world((1.0, 2.0), 0.0) == pytest.approx((1.4, 2.0))
    assert config.target_world((1.0, 2.0), math.pi / 2) == pytest.approx(
        (1.0, 2.4)
    )


def test_factory_initial_pose_advances_40_cm_without_changing_yaw():
    config = InitialAdvanceConfig(0.4)
    yaw = math.radians(-85.70)
    target = config.target_world((13.0, 16.0), yaw)
    assert math.dist((13.0, 16.0), target) == pytest.approx(0.4)
    assert target == pytest.approx((13.029991, 15.601126), abs=1e-6)


@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf])
def test_invalid_distance_is_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        InitialAdvanceConfig(value)


def test_default_preserves_legacy_no_advance_behavior():
    config = InitialAdvanceConfig.from_robot_motion({})
    assert config.distance_m == 0.0
    assert config.target_world((3.0, 4.0), 1.2) == (3.0, 4.0)
