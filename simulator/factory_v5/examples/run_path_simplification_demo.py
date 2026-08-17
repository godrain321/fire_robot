#!/usr/bin/env python3
"""Demonstrate safe A* post-processing without FDS or Pygame."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

FACTORY = Path(__file__).resolve().parents[1]
if str(FACTORY) not in sys.path:
    sys.path.insert(0, str(FACTORY))

from navigation.path_simplifier import (  # noqa: E402
    PathRiskConfig, PathSimplificationConfig, SafePathSimplifier,
    cells_touched_by_segment, extract_direction_change_points,
)
from world import EstimatedFireMap, MapMetadata  # noqa: E402


def environment(config=None):
    metadata = MapMetadata(0, 6, 0, 6, 1, 7, 7, (0, 0))
    estimated = EstimatedFireMap(
        metadata, temperature_blocked_c=60, co_blocked_ppm=1600
    )
    estimated.replace_layers(
        np.full((7, 7), 20.0), np.zeros((7, 7)),
        np.ones((7, 7), dtype=bool), np.zeros((7, 7)),
    )
    return (
        SafePathSimplifier(metadata, config), np.ones((7, 7)),
        np.zeros((7, 7), dtype=bool),
        np.zeros((7, 7), dtype=bool), estimated,
    )


def simplify(label, path, simplifier, costs, static, dynamic, estimated):
    before = tuple(path)
    result = simplifier.simplify(
        path, costmap=costs, static_obstacle_map=static,
        dynamic_obstacle_map=dynamic, estimated_fire_map=estimated,
        costmap_revision=12, start_world=(path[0][0] + 0.1, path[0][1]),
        goal_world=(path[-1][0] + 0.1, path[-1][1]),
    )
    print(f"\n{label}")
    print("Original path points:", result.original_point_count)
    print("Corner points:", result.corner_point_count)
    print("Simplified waypoints:", result.simplified_point_count)
    print(f"Reduction ratio: {result.reduction_ratio * 100:.1f}%")
    print(f"Length: {result.original_length_m:.2f} -> {result.simplified_length_m:.2f} m")
    print(f"Risk: {result.original_risk_cost:.2f} -> {result.simplified_risk_cost:.2f}")
    print("Fallback used:", result.fallback_used)
    print("Validation:", "SAFE" if result.success else result.failure_reason)
    print("Input unchanged:", tuple(path) == before)
    json.dumps(result.to_dict())
    return result


def main():
    simplifier, costs, static, dynamic, estimated = environment()
    simplify("STRAIGHT", [(0, 0), (1, 0), (2, 0), (3, 0)],
             simplifier, costs, static, dynamic, estimated)
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2)]
    print("Direction changes:", extract_direction_change_points(path))
    safe = simplify("MULTIPLE CORNERS / SAFE SHORTCUT", path,
                    simplifier, costs, static, dynamic, estimated)
    print("Supercover (0,0)->(3,2):", cells_touched_by_segment((0, 0), (3, 2)))

    static[1, 1] = True
    blocked = simplify("STATIC WALL / CORNER CUT REJECTED", path,
                       simplifier, costs, static, dynamic, estimated)
    print("Rejected reasons:", [
        [reason.value for reason in item.rejection_reasons]
        for item in blocked.rejected_shortcuts[:3]
    ])
    static[1, 1] = False

    dynamic[1, 1] = True
    simplify("DYNAMIC OBSTACLE", path, simplifier, costs, static, dynamic, estimated)
    dynamic[1, 1] = False
    estimated.temperature_c[1, 1] = 60
    simplify("TEMPERATURE BLOCK", path, simplifier, costs, static, dynamic, estimated)
    estimated.temperature_c[1, 1] = 20
    estimated.co_ppm[1, 1] = 1600
    simplify("CO BLOCK", path, simplifier, costs, static, dynamic, estimated)
    estimated.co_ppm[1, 1] = 0

    costs[1, 1] = np.nan
    simplify("NAN COST", path, simplifier, costs, static, dynamic, estimated)
    costs[1, 1] = 1

    risk_config = PathSimplificationConfig(
        risk=PathRiskConfig(max_risk_increase_ratio=1.0)
    )
    risk_simplifier, risk_costs, static, dynamic, estimated = environment(risk_config)
    risk_costs[1, 1] = 20
    risk = simplify("RISKIER SHORTCUT REJECTED", path, risk_simplifier,
                    risk_costs, static, dynamic, estimated)
    print("Final grid waypoints:", risk.simplified_path_grid)
    print("World waypoints:", risk.waypoints_world)
    print("JSON serialization: OK", bool(json.dumps(safe.to_dict())))


if __name__ == "__main__":
    main()
