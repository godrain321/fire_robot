#!/usr/bin/env python3
"""Validate the factory_v3 evacuation configuration without running FDS."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import yaml

from mapping.fire_costmap import load_factory_geometry
from mapping.grid_map import GridMap


def main() -> int:
    base = Path(__file__).resolve().parent
    scenario = yaml.safe_load(
        (base / "config" / "evacuation.yaml").read_text(encoding="utf-8")
    )
    fds_path = base / scenario["fds_file"]
    text = fds_path.read_text(encoding="utf-8")
    head = re.search(r"&HEAD\b.*?\bCHID\s*=\s*'([^']+)'", text, re.I | re.S)
    time = re.search(r"&TIME\b.*?\bT_END\s*=\s*([0-9.]+)", text, re.I | re.S)
    if head is None or time is None:
        raise ValueError("HEAD CHID or TIME T_END missing")
    required_output_tokens = (
        "ID='TEMP_3D'", "QUANTITY='TEMPERATURE'", "CELL_CENTERED=T",
        "ID='CO_Z130'", "SPEC_ID='CARBON MONOXIDE'", "PBZ=1.3",
    )
    missing = [token for token in required_output_tokens if token not in text]
    if missing:
        raise ValueError(f"required FDS output definitions missing: {missing}")

    mesh, obstacles, holes = load_factory_geometry(fds_path)
    resolution = float(scenario["planner"]["grid_resolution_m"])
    clearance = float(scenario["planner"]["inflation_radius_m"])
    grid = GridMap(mesh, obstacles, holes, resolution, clearance)
    configured = [
        ("robot_start", scenario["robot_start"]),
        *[(human["id"], human) for human in scenario["humans"]],
        *[(item["id"], item["approach"]) for item in scenario["exits"]],
    ]
    point_report = {}
    for name, point in configured:
        world = float(point["x"]), float(point["y"])
        node = grid.world_to_grid(*world)
        if not grid.in_bounds(node):
            raise ValueError(f"{name} outside map: {world} -> {node}")
        if grid.is_blocked(node):
            raise ValueError(f"{name} blocked: {world} -> {node}")
        roundtrip = grid.grid_to_world(*node)
        if max(abs(roundtrip[i] - world[i]) for i in (0, 1)) > resolution / 2 + 1e-6:
            raise ValueError(f"{name} coordinate roundtrip failed")
        point_report[name] = {"world": world, "grid": node, "roundtrip": roundtrip}

    report = {
        "status": "PASS",
        "input_chid": head.group(1),
        "result_chid": scenario["fds_result_chid"],
        "t_end_s": float(time.group(1)),
        "mesh_xb": mesh,
        "planner_shape_yx": [grid.height, grid.width],
        "planner_resolution_m": resolution,
        "fds_obstacle_count_including_catf": len(obstacles),
        "configured_points": point_report,
        "temperature_npz_exists": (base / scenario["temperature_npz"]).is_file(),
        "result_smv_exists": (base / f"{scenario['fds_result_chid']}.smv").is_file(),
        "array_contracts": {
            "temperature_npz": "[time,z,y,x]",
            "camera_volume": "[z,y,x]",
            "co": "[time,y,x] ppm",
            "costmap": "[y,x]",
            "planner_node": "(x,y)",
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
