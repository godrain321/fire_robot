#!/usr/bin/env python3
"""Run normal and exceptional mission transitions without FDS or Pygame."""

from pathlib import Path
import sys

FACTORY_DIR = Path(__file__).resolve().parents[1]
if str(FACTORY_DIR) not in sys.path:
    sys.path.insert(0, str(FACTORY_DIR))

from mission.mission_manager import MissionEvent, MissionManager


def emit(manager, event, **context):
    transition = manager.handle_event(event, **context)
    print(
        f"{transition.previous_state.name} --{transition.event.name}--> "
        f"{transition.next_state.name}: {transition.reason}"
    )


def normal_demo():
    print("=== Normal flow ===")
    manager = MissionManager()
    emit(manager, MissionEvent.VICTIM_DETECTED, victim_id="victim_1", victim_position=(2.0, 3.0), sim_time=1.0)
    emit(manager, MissionEvent.VICTIM_REACHED, sim_time=5.0)
    emit(manager, MissionEvent.ANNOUNCEMENT_FINISHED, sim_time=6.0)
    emit(manager, MissionEvent.VICTIM_STARTED_MOVING, sim_time=7.0)
    emit(manager, MissionEvent.PATH_PLANNED, exit_id="EXIT1", exit_position=(9.0, 4.0), path=[(1, 1), (2, 1)], sim_time=8.0)
    emit(manager, MissionEvent.EXIT_REACHED, sim_time=20.0, remaining_victims=False)


def exceptional_demo():
    print("\n=== Exceptional flow ===")
    manager = MissionManager()
    emit(manager, MissionEvent.VICTIM_DETECTED, victim_id="victim_2", victim_position=(5.0, 6.0), sim_time=1.0)
    emit(manager, MissionEvent.VICTIM_REACHED, sim_time=2.0)
    emit(manager, MissionEvent.ANNOUNCEMENT_FINISHED, sim_time=3.0)
    emit(manager, MissionEvent.VICTIM_STARTED_MOVING, sim_time=3.5)
    emit(manager, MissionEvent.PATH_PLANNED, exit_id="EXIT2", exit_position=(10.0, 7.0), path=[(2, 2)], sim_time=4.0)
    emit(manager, MissionEvent.PATH_BLOCKED, reason="thermal risk blocked route", sim_time=5.0)
    emit(manager, MissionEvent.RETRY_REQUESTED, sim_time=5.1)
    emit(manager, MissionEvent.PATH_PLANNING_FAILED, failed_exits={"EXIT1": "blocked", "EXIT2": "unsafe"}, sim_time=5.2)
    emit(manager, MissionEvent.PATH_PLANNING_FAILED, sim_time=5.3)


if __name__ == "__main__":
    normal_demo()
    exceptional_demo()
