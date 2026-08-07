#!/usr/bin/env python3
"""Regenerate only scenario.inc while treating map geometry as read-only."""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FACTORY_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_slam_map import load_yaml, merge_rectangles  # noqa: E402


OBST_RE = re.compile(
    r"&OBST\s+ID='([^']+)'.*?XB=([0-9.,+\-Ee]+).*?SURF_ID='([^']+)'"
)
VENT_RE = re.compile(
    r"&VENT\s+ID='([^']+)'.*?XB=([0-9.,+\-Ee]+).*?SURF_ID='([^']+)'"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_records(path: Path, pattern: re.Pattern[str]) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        xb = [
            float(value) for value in match.group(2).split(",")
            if value.strip()
        ]
        if len(xb) != 6:
            raise ValueError(f"invalid XB in {path}: {line}")
        records.append({"id": match.group(1), "xb": xb, "surface": match.group(3)})
    return records


def rasterize(records: list[dict], shape: tuple[int, int], resolution: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for record in records:
        x0, x1, y0, y1 = record["xb"][:4]
        gx0, gx1 = round(x0 / resolution), round(x1 / resolution)
        gy0, gy1 = round(y0 / resolution), round(y1 / resolution)
        mask[max(gy0, 0):min(gy1, shape[0]), max(gx0, 0):min(gx1, shape[1])] = True
    return mask


def nearest_cell(mask: np.ndarray, x: float, y: float, resolution: float) -> tuple[int, int]:
    candidates = np.argwhere(mask)
    if not candidates.size:
        raise ValueError("no candidate grid cell")
    gy, gx = min(
        candidates,
        key=lambda cell: (
            ((cell[1] + 0.5) * resolution - x) ** 2
            + ((cell[0] + 0.5) * resolution - y) ** 2
        ),
    )
    return int(gx), int(gy)


def shortest_path(
    free: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    queue = deque([start])
    parent = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = current[0] + dx, current[1] + dy
            if not (
                0 <= nxt[0] < free.shape[1]
                and 0 <= nxt[1] < free.shape[0]
                and free[nxt[1], nxt[0]]
                and nxt not in parent
            ):
                continue
            parent[nxt] = current
            queue.append(nxt)
    if goal not in parent:
        raise ValueError(f"no free-floor path from {start} to {goal}")
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


def main() -> int:
    scenario_path = FACTORY_DIR / "config" / "scenario.yaml"
    semantic_path = FACTORY_DIR / "config" / "semantic_points.yaml"
    obstacles_path = FACTORY_DIR / "generated" / "obstacles.inc"
    exits_path = FACTORY_DIR / "generated" / "exits.inc"
    output_path = FACTORY_DIR / "generated" / "scenario.inc"
    report_path = FACTORY_DIR / "generated" / "fire_scenario_report.json"
    scenario = load_yaml(scenario_path)
    semantic = load_yaml(semantic_path)
    protected = scenario["protected_geometry"]
    before_hashes = {
        "obstacles": sha256(obstacles_path),
        "exits": sha256(exits_path),
    }
    expected = {
        "obstacles": str(protected["obstacles_sha256"]),
        "exits": str(protected["exits_sha256"]),
    }
    if before_hashes != expected:
        raise RuntimeError(
            "protected geometry hash mismatch; refusing to generate fire scenario"
        )

    resolution = 0.2
    shape = (114, 136)
    obstacle_records = parse_records(obstacles_path, OBST_RE)
    exit_records = parse_records(exits_path, OBST_RE)
    blocked = rasterize(obstacle_records + exit_records, shape, resolution)
    free = ~blocked

    cardboard = scenario["fire"]["cardboard_boxes"]
    requested_ids = list(dict.fromkeys(map(str, cardboard["wall_obstacle_ids"])))
    records_by_id = {record["id"]: record for record in obstacle_records}
    missing = [record_id for record_id in requested_ids if record_id not in records_by_id]
    if missing:
        raise ValueError(f"wall obstacle IDs not found: {missing}")
    guide_records = [records_by_id[record_id] for record_id in requested_ids]
    no_go_guide = rasterize(
        [record for record in guide_records if record["id"].startswith("NO_GO_ZONE_")],
        shape,
        resolution,
    )
    slam_guide = rasterize(
        [record for record in guide_records if record["id"].startswith("SLAM_OCCUPIED_")],
        shape,
        resolution,
    )

    def adjacent_floor(wall: np.ndarray) -> np.ndarray:
        return (
            cv2.dilate(wall.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ) & free

    def expand_floor(mask: np.ndarray, half_width_m: float) -> np.ndarray:
        dilation = math.ceil(half_width_m / resolution)
        if not dilation:
            return mask.copy()
        kernel = np.ones((dilation * 2 + 1, dilation * 2 + 1), np.uint8)
        return (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & free

    no_go_floor = adjacent_floor(no_go_guide)
    slam_floor = adjacent_floor(slam_guide)
    wall_floor = no_go_floor | slam_floor

    existing_text = output_path.read_text(encoding="utf-8")
    existing_vents = parse_records(output_path, VENT_RE)
    oil_records = [
        record for record in existing_vents
        if "_SPILL_" in record["id"] and "CARDBOARD" not in record["id"]
    ]
    oil_mask = rasterize(oil_records, shape, resolution)
    xyz_match = re.search(
        r"ID='[^']+_SPILL_\d+'.*?XYZ=([0-9.+\-Ee]+),([0-9.+\-Ee]+)",
        existing_text,
    )
    if xyz_match is None:
        raise ValueError("existing oil ignition XYZ was not found")
    ignition_x, ignition_y = map(float, xyz_match.groups())
    ignition_floor = nearest_cell(free, ignition_x, ignition_y, resolution)
    start_wall = nearest_cell(wall_floor, ignition_x, ignition_y, resolution)

    target_name = str(cardboard["target_machinery"])
    target = semantic["landmarks"][target_name]["fds"]
    target_x, target_y = float(target["x"]), float(target["y"])
    target_floor = nearest_cell(free, target_x, target_y, resolution)

    requested_slope = float(cardboard["machinery_connector_slope"])
    connector_run_cells = max(
        1, round(float(cardboard["machinery_connector_x_run_m"]) / resolution)
    )
    connector_rise_cells = max(1, round(requested_slope * connector_run_cells))
    diagonal_start = (
        target_floor[0] + connector_run_cells,
        target_floor[1] + connector_rise_cells,
    )
    if not (
        0 <= diagonal_start[0] < free.shape[1]
        and 0 <= diagonal_start[1] < free.shape[0]
    ):
        raise ValueError("configured machinery connector leaves the FDS grid")

    def raster_line(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        x0, y0 = start
        x1, y1 = goal
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        error = dx - dy
        while True:
            if free[y0, x0]:
                cells.append((x0, y0))
            if (x0, y0) == (x1, y1):
                break
            doubled = 2 * error
            if doubled > -dy:
                error -= dy
                x0 += sx
            if doubled < dx:
                error += dx
                y0 += sy
        return cells

    ignition_connector = np.zeros(shape, dtype=bool)
    for gx, gy in shortest_path(free, ignition_floor, start_wall):
        ignition_connector[gy, gx] = True
    diagonal_connector = np.zeros(shape, dtype=bool)
    for gx, gy in raster_line(diagonal_start, target_floor):
        diagonal_connector[gy, gx] = True

    no_go_fire = expand_floor(
        no_go_floor | ignition_connector,
        float(cardboard["no_go_path_half_width_m"]),
    )
    slam_fire = expand_floor(
        slam_floor,
        float(cardboard["slam_wall_path_half_width_m"]),
    )
    connector_fire = expand_floor(
        diagonal_connector,
        float(cardboard["machinery_connector_half_width_m"]),
    )
    fire_mask = no_go_fire | slam_fire | connector_fire
    fire_mask &= free & ~oil_mask
    rectangles = merge_rectangles(fire_mask)
    if not rectangles:
        raise ValueError("specified wall route produced no fire surface")

    base_lines = [
        line for line in existing_text.splitlines()
        if "CARDBOARD" not in line and "ROOF_VENT" not in line
    ]
    surface_id = f"{scenario['fire']['id']}_CARDBOARD"
    cardboard_ramp_id = f"{surface_id}_RAMP"
    cardboard_material_id = f"{surface_id}_MATL"
    hrrpua = float(cardboard["hrrpua_kw_m2"])
    ignition_ramp_s = float(cardboard["local_ignition_ramp_s"])
    burn_duration_s = float(cardboard["burn_duration_s"])
    density = float(cardboard["material_density_kg_m3"])
    specific_heat = float(cardboard["material_specific_heat_kj_kg_k"])
    conductivity = float(cardboard["material_conductivity_w_m_k"])
    thickness = float(cardboard["material_thickness_m"])
    ignition_temperature = float(cardboard["ignition_temperature_c"])
    stack_height = float(cardboard["stack_height_m"])
    if ignition_ramp_s <= 0.0 or burn_duration_s < 420.0:
        raise ValueError(
            "cardboard ignition ramp must be positive and burn duration at least 420 s"
        )
    if min(density, specific_heat, conductivity, thickness, stack_height) <= 0.0:
        raise ValueError("cardboard material properties and stack height must be positive")
    base_lines.extend((
        f"&MATL ID='{cardboard_material_id}', DENSITY={density:.3f}, "
        f"SPECIFIC_HEAT={specific_heat:.4f}, CONDUCTIVITY={conductivity:.4f} /",
        f"&SURF ID='{surface_id}', MATL_ID='{cardboard_material_id}', "
        f"THICKNESS={thickness:.4f}, HRRPUA={hrrpua:.3f}, "
        f"IGNITION_TEMPERATURE={ignition_temperature:.3f}, "
        f"RAMP_Q='{cardboard_ramp_id}', BURN_DURATION={burn_duration_s:.3f}, "
        "COLOR='BROWN' /",
    ))
    base_lines.extend((
        f"&RAMP ID='{cardboard_ramp_id}', T=0.000, F=0.000000 /",
        f"&RAMP ID='{cardboard_ramp_id}', T={ignition_ramp_s:.3f}, F=1.000000 /",
        f"&RAMP ID='{cardboard_ramp_id}', T={burn_duration_s:.3f}, F=1.000000 /",
    ))
    for index, (gx0, gx1, gy0, gy1) in enumerate(rectangles, start=1):
        base_lines.append(
            f"&OBST ID='{scenario['fire']['id']}_CARDBOARD_STACK_{index:03d}', "
            f"XB={gx0 * resolution:.3f},{gx1 * resolution:.3f},"
            f"{gy0 * resolution:.3f},{gy1 * resolution:.3f},"
            f"0.000,{stack_height:.3f}, SURF_ID='{surface_id}' /"
        )
    roof_vent = scenario["fire"]["roof_vent"]
    if bool(roof_vent["enabled"]):
        vent_xb = [float(value) for value in roof_vent["xb"]]
        if len(vent_xb) != 6 or vent_xb[4:] != [3.0, 3.0]:
            raise ValueError("roof vent must be a six-value XB on ZMAX=3.0 m")
        base_lines.append(
            "&VENT ID='ROOF_VENT_OPEN', "
            f"XB={','.join(f'{value:.3f}' for value in vent_xb)}, "
            "SURF_ID='OPEN' /"
        )
    output_path.write_text("\n".join(base_lines).rstrip() + "\n", encoding="utf-8")

    after_hashes = {
        "obstacles": sha256(obstacles_path),
        "exits": sha256(exits_path),
    }
    if after_hashes != before_hashes:
        raise RuntimeError("protected geometry changed during scenario generation")
    ys, xs = np.where(fire_mask)
    report = {
        "status": "PASS",
        "protected_geometry_unchanged": True,
        "protected_hashes": after_hashes,
        "wall_obstacle_ids": requested_ids,
        "target_machinery": target_name,
        "slam_wall_half_width_m": float(cardboard["slam_wall_path_half_width_m"]),
        "no_go_half_width_m": float(cardboard["no_go_path_half_width_m"]),
        "machinery_connector_requested_slope": requested_slope,
        "machinery_connector_actual_slope": (
            (diagonal_start[1] - target_floor[1])
            / (diagonal_start[0] - target_floor[0])
        ),
        "machinery_connector_start_grid": list(diagonal_start),
        "machinery_connector_end_grid": list(target_floor),
        "no_go_region_cells": int((no_go_fire & ~oil_mask).sum()),
        "slam_wall_region_cells": int((slam_fire & ~oil_mask).sum()),
        "machinery_connector_region_cells": int(
            (connector_fire & ~oil_mask).sum()
        ),
        "surface_cells": int(fire_mask.sum()),
        "surface_area_m2": float(fire_mask.sum() * resolution**2),
        "rectangles": len(rectangles),
        "ignition_mode": "surface_temperature",
        "ignition_temperature_c": ignition_temperature,
        "material_density_kg_m3": density,
        "material_specific_heat_kj_kg_k": specific_heat,
        "material_conductivity_w_m_k": conductivity,
        "material_thickness_m": thickness,
        "stack_height_m": stack_height,
        "hrrpua_kw_m2": hrrpua,
        "local_ignition_ramp_s": ignition_ramp_s,
        "burn_duration_s": burn_duration_s,
        "minimum_requested_burn_duration_s": 420.0,
        "roof_vent_xb": roof_vent["xb"] if bool(roof_vent["enabled"]) else None,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
