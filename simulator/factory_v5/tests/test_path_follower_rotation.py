import math

import pytest

from robot.path_follower import shortest_rotation_delta


def test_shortest_rotation_crosses_positive_boundary_counter_clockwise():
    delta = shortest_rotation_delta(math.radians(170), math.radians(-170))
    assert math.degrees(delta) == pytest.approx(20.0)


def test_shortest_rotation_crosses_negative_boundary_clockwise():
    delta = shortest_rotation_delta(math.radians(-170), math.radians(170))
    assert math.degrees(delta) == pytest.approx(-20.0)


def test_shortest_rotation_selects_clockwise_when_it_is_smaller():
    delta = shortest_rotation_delta(math.radians(20), math.radians(-100))
    assert math.degrees(delta) == pytest.approx(-120.0)


def test_shortest_rotation_selects_counter_clockwise_when_it_is_smaller():
    delta = shortest_rotation_delta(math.radians(-20), math.radians(100))
    assert math.degrees(delta) == pytest.approx(120.0)


def test_exact_half_turn_uses_deterministic_counter_clockwise_tie_break():
    delta = shortest_rotation_delta(0.0, math.pi)
    assert delta == pytest.approx(math.pi)
