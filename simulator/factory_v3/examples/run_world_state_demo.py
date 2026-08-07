#!/usr/bin/env python3
"""Exercise WorldState without loading FDS results or Pygame."""

import json
from pathlib import Path
import sys

import numpy as np

FACTORY_DIR = Path(__file__).resolve().parents[1]
if str(FACTORY_DIR) not in sys.path:
    sys.path.insert(0, str(FACTORY_DIR))

from world import (
    DynamicObstacle, DynamicObstacleShape, DynamicObstacleStatus,
    Exit, ExitStatus, MapMetadata, Victim, VictimStatus, WorldState,
)


metadata = MapMetadata(0.0, 4.0, 0.0, 3.0, 1.0, 5, 4, (0.0, 0.0))
world = WorldState(metadata, np.zeros((4, 5), dtype=bool))
world.add_exit(Exit("EXIT_A", (0.0, 1.0), (1.0, 1.0)))
world.add_exit(Exit("EXIT_B", (4.0, 2.0), (3.0, 2.0)))
world.add_victim(Victim("victim_1", (2.0, 1.0)))
world.add_victim(Victim("victim_2", (2.0, 2.0)))
world.add_dynamic_obstacle(DynamicObstacle(
    "cart_1", (3.0, 1.0), DynamicObstacleShape.RECTANGLE,
    (1.0, 1.0), DynamicObstacleStatus.ACTIVE, source="demo", confidence=0.9,
))
world.update_victim_status("victim_1", VictimStatus.DETECTED, sim_time=1.0)
world.update_exit_status("EXIT_A", ExitStatus.USABLE, sim_time=1.0)
world.update_exit_status("EXIT_B", ExitStatus.DANGEROUS, sim_time=2.0, reason="high CO")
world.clear_dynamic_obstacle("cart_1", sim_time=3.0)
world.estimated_fire_map.update_cell(2, 1, temperature_c=65.0, co_ppm=200.0, sim_time=3.0)
world.ground_truth_fire_map.update_cell(2, 1, temperature_c=120.0, co_ppm=800.0, sim_time=3.0)

assert world.ground_truth_fire_map.temperature_c is not world.estimated_fire_map.temperature_c
print("Usable exits:", [item.exit_id for item in world.get_usable_exits()])
print("Unrescued victims:", [item.victim_id for item in world.get_unrescued_victims()])
print("Independent maps: yes")
print("JSON bytes:", len(json.dumps(world.to_dict())))
