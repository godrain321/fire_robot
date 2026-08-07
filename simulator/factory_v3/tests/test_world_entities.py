import pytest

from world.entities import (
    DynamicObstacle, DynamicObstacleShape, DynamicObstacleStatus,
    Exit, ExitStatus, Victim, VictimStatus,
)


def test_exit_creation_and_status_history():
    item = Exit("EXIT1", (0, 1), (1, 1))
    assert item.status is ExitStatus.UNKNOWN
    item.update_status(ExitStatus.DANGEROUS, sim_time=2.0, reason="hot", temperature_c=70)
    assert item.danger_reason == "hot"
    assert item.status_history[-1].previous is ExitStatus.UNKNOWN


def test_exit_rejects_invalid_status_and_missing_reason():
    with pytest.raises(TypeError):
        Exit("EXIT1", (0, 1), (1, 1), status="usable")
    with pytest.raises(ValueError):
        Exit("EXIT1", (0, 1), (1, 1)).update_status(ExitStatus.BLOCKED)


def test_victim_boolean_properties_stay_consistent():
    victim = Victim("v1", (1, 2))
    assert not victim.detected and victim.movable is None and not victim.rescued
    victim.update_status(VictimStatus.DETECTED, sim_time=1)
    victim.update_status(VictimStatus.REACHED, sim_time=2)
    victim.update_status(VictimStatus.MOVABLE, sim_time=3)
    assert victim.detected and victim.movable is True
    victim.update_status(VictimStatus.EVACUATING)
    victim.update_status(VictimStatus.RESCUED)
    assert victim.rescued and victim.movable is True


def test_victim_invalid_transition_rejected_and_unknown_movable_preserved():
    victim = Victim("v1", (1, 2))
    assert victim.movable is None
    with pytest.raises(ValueError):
        victim.update_status(VictimStatus.RESCUED)


def test_dynamic_obstacle_creation_update_and_clear():
    item = DynamicObstacle(
        "cart", (2, 2), DynamicObstacleShape.CIRCLE, (1, 1),
        DynamicObstacleStatus.ACTIVE, confidence=0.8,
    )
    item.update(position_world=(2.5, 2), status=DynamicObstacleStatus.MOVING, sim_time=2)
    item.update(status=DynamicObstacleStatus.CLEARED, sim_time=3)
    assert item.status is DynamicObstacleStatus.CLEARED
    with pytest.raises(ValueError):
        item.update(status=DynamicObstacleStatus.ACTIVE)
