#!/usr/bin/env python3
"""Travel-history and return-path demo without FDS or Pygame."""

import json
from pathlib import Path
import sys

import numpy as np

FACTORY_DIR = Path(__file__).resolve().parents[1]
if str(FACTORY_DIR) not in sys.path:
    sys.path.insert(0, str(FACTORY_DIR))

from navigation import ReturnPathPlanner, TravelHistory, TravelHistoryConfig
from world import EstimatedFireMap, MapMetadata


metadata = MapMetadata(0, 4, 0, 3, 1, 5, 4, (0, 0))
history = TravelHistory(metadata, TravelHistoryConfig(min_record_distance_m=0.4))
history.reset((0, 0), recorded_at=0)
history.record_position((0, 0), recorded_at=0.1)       # stationary: ignored
history.record_position((0.1, 0), recorded_at=0.2)     # below threshold: ignored
history.record_position((1.0, 0), recorded_at=1.0)     # new cell
history.record_position((2.0, 0), recorded_at=2.0)
history.record_position((2.0, 1.0), recorded_at=3.0)   # corner
original = history.get_points()

estimated = EstimatedFireMap(metadata, temperature_blocked_c=60, co_blocked_ppm=1600)
safe = np.ones((4, 5), dtype=bool)
estimated.replace_layers(np.full((4, 5), 20.0), np.zeros((4, 5)), safe, np.zeros((4, 5)))
cost = np.ones((4, 5))
static = np.zeros((4, 5), dtype=bool)
dynamic = np.zeros((4, 5), dtype=bool)
planner = ReturnPathPlanner(metadata)
plan = planner.create_plan(
    history, (2, 1), cost_map=cost, static_obstacle_map=static,
    dynamic_obstacle_map=dynamic, estimated_fire_map=estimated, created_at=4,
)
print("History:", json.dumps(history.to_dict(), indent=2))
print("Safe return:", plan.success, plan.path_grid)

dynamic[0, 1] = True
blocked = planner.create_plan(
    history, (2, 1), cost_map=cost, static_obstacle_map=static,
    dynamic_obstacle_map=dynamic, estimated_fire_map=estimated, created_at=5,
)
print("Blocked return:", blocked.success, blocked.failure_reason.value)
assert history.get_points() == original
json.dumps(plan.to_dict())
