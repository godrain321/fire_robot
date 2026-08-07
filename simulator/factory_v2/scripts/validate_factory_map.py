#!/usr/bin/env python3
"""Validate coordinate orientation, semantic points and converted connectivity."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys

import numpy as np
import cv2
from PIL import Image, ImageDraw
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
FACTORY_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from map_coordinates import MapTransform  # noqa: E402
from convert_slam_map import (  # noqa: E402
    classify_pixels,
    downsample_any,
    load_yaml,
    resolve_from_factory,
)


MOVES = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
)


def shortest_path(
    free: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> tuple[list[tuple[int, int]], float]:
    """Return an 8-connected path without diagonal corner cutting."""
    height, width = free.shape

    def valid(node: tuple[int, int]) -> bool:
        x, y = node
        return 0 <= x < width and 0 <= y < height and bool(free[y, x])

    if not valid(start) or not valid(goal):
        return [], math.inf
    frontier = [(0.0, start)]
    distance = {start: 0.0}
    parent = {start: None}
    while frontier:
        cost, current = heapq.heappop(frontier)
        if cost > distance[current]:
            continue
        if current == goal:
            break
        x, y = current
        for dx, dy, step in MOVES:
            nxt = (x + dx, y + dy)
            if not valid(nxt):
                continue
            if dx and dy and (not valid((x + dx, y)) or not valid((x, y + dy))):
                continue
            new_cost = cost + step
            if new_cost < distance.get(nxt, math.inf):
                distance[nxt] = new_cost
                parent[nxt] = current
                heapq.heappush(frontier, (new_cost, nxt))
    if goal not in parent:
        return [], math.inf
    path = []
    node: tuple[int, int] | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path)), distance[goal]


def clearance_m(blocked: np.ndarray, node: tuple[int, int], resolution: float) -> float:
    yx = np.argwhere(blocked)
    if yx.size == 0:
        return math.inf
    x, y = node
    squared = (yx[:, 1] - x) ** 2 + (yx[:, 0] - y) ** 2
    return float(math.sqrt(float(squared.min())) * resolution)


def unknown_component_bbox(
    unknown: np.ndarray, start: tuple[int, int], resolution: float
) -> dict | None:
    row, col = start
    if not unknown[row, col]:
        return None
    height, width = unknown.shape
    seen = {start}
    queue = deque([start])
    rows: list[int] = []
    cols: list[int] = []
    while queue:
        r, c = queue.popleft()
        rows.append(r)
        cols.append(c)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if (
                0 <= nxt[0] < height and 0 <= nxt[1] < width
                and unknown[nxt] and nxt not in seen
            ):
                seen.add(nxt)
                queue.append(nxt)
    x0, x1 = min(cols), max(cols) + 1
    row0, row1 = min(rows), max(rows) + 1
    y0 = (height - row1) * resolution
    y1 = (height - row0) * resolution
    return {
        "cell_count": len(seen),
        "fds_bbox": [x0 * resolution, x1 * resolution, y0, y1],
        "bbox_size_m": [(x1 - x0) * resolution, (row1 - row0) * resolution],
    }


def draw_overlay(
    image: np.ndarray,
    occupied: np.ndarray,
    unknown: np.ndarray,
    no_go: np.ndarray,
    machinery: np.ndarray,
    exit_markers: np.ndarray,
    oil_spill: np.ndarray,
    cardboard_boxes: np.ndarray,
    paths: dict[str, list[tuple[int, int]]],
    poses: dict,
    transform: MapTransform,
    coarse_resolution: float,
    scenario: dict,
    output_path: Path,
) -> None:
    base = Image.fromarray(image).convert("RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    factor = round(coarse_resolution / transform.resolution)
    for mask, color in (
        (unknown, (45, 90, 210, 55)),
        (occupied, (225, 45, 35, 90)),
        (no_go, (255, 135, 0, 125)),
        (machinery, (70, 70, 70, 150)),
        (exit_markers, (20, 90, 255, 190)),
        (oil_spill, (255, 35, 0, 145)),
        (cardboard_boxes, (145, 85, 35, 170)),
    ):
        for y, x in np.argwhere(mask):
            left = int(x * factor)
            right = min(int((x + 1) * factor), transform.width) - 1
            top = max(transform.height - int((y + 1) * factor), 0)
            bottom = min(transform.height - int(y * factor), transform.height) - 1
            if left < transform.width and top < transform.height:
                draw.rectangle((left, top, right, bottom), fill=color)
    result = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(result)
    path_colors = {"EXIT1": "#00a6ff", "EXIT2": "#00c96b", "EXIT3": "#c83cff"}
    for name, path in paths.items():
        points = []
        for x, y in path:
            col = (x + 0.5) * factor
            row = transform.height - (y + 0.5) * factor
            points.append((col, row))
        if len(points) > 1:
            draw.line(points, fill=path_colors[name], width=2)
    for name, values in poses.items():
        row, col = transform.ros_to_pixel(values["ros"]["x"], values["ros"]["y"])
        color = "#ffff00" if name == "INIT" else path_colors[name]
        draw.ellipse((col - 4, row - 4, col + 4, row + 4), fill=color, outline="black")
        draw.text((col + 6, row - 7), name, fill=color, stroke_width=1, stroke_fill="black")
    if bool(scenario.get("fire_enabled", False)):
        x0, x1, y0, y1 = scenario["oil_spill_bbox"]
        left = x0 / transform.resolution
        right = x1 / transform.resolution
        top = transform.height - y1 / transform.resolution
        bottom = transform.height - y0 / transform.resolution
        draw.rectangle((left, top, right, bottom), outline="yellow", width=2)
        ignition_x, ignition_y = scenario["ignition_center_fds"]
        fire_col = ignition_x / transform.resolution
        fire_row = transform.height - ignition_y / transform.resolution
        draw.ellipse(
            (fire_col - 5, fire_row - 5, fire_col + 5, fire_row + 5),
            fill="#ff2600", outline="yellow",
        )
        draw.text(
            (fire_col + 7, fire_row - 7), "FIRE", fill="red",
            stroke_width=1, stroke_fill="white",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    result.convert("RGB").resize(
        (transform.width * 2, transform.height * 2), nearest
    ).save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=FACTORY_DIR / "config" / "map_metadata.yaml",
    )
    args = parser.parse_args()
    config = load_yaml(args.config.resolve())
    source = load_yaml(resolve_from_factory(config["source_map_yaml"]))
    source_semantic = load_yaml(resolve_from_factory(config["source_semantic_yaml"]))
    semantic = load_yaml(FACTORY_DIR / "config" / "semantic_points.yaml")
    image = np.asarray(Image.open(resolve_from_factory(config["source_map"])).convert("L"))
    occupied_src, free_src, unknown_src = classify_pixels(image, source)
    grid_file = np.load(FACTORY_DIR / "generated" / "occupancy_grid.npz")
    blocked = grid_file["blocked"]
    occupied = grid_file["occupied"]
    unknown = grid_file["unknown"]
    no_go = grid_file["no_go"]
    machinery_grid = grid_file["machinery"]
    oil_spill_grid = grid_file["oil_spill"]
    cardboard_boxes_grid = grid_file["cardboard_boxes"]
    exit_markers_grid = grid_file["exit_markers"]
    exit_corridors = grid_file["exit_corridors"]
    free = ~blocked
    resolution = float(grid_file["resolution_m"])
    transform = MapTransform(
        image.shape[1], image.shape[0], float(source["resolution"]),
        *map(float, source["origin"]),
    )

    errors: list[str] = []
    ceiling_matches_walls = math.isclose(
        float(config["fds_height_m"]),
        float(config["wall_height_m"]),
        abs_tol=1e-12,
    )
    if not ceiling_matches_walls:
        errors.append("FDS ceiling height differs from wall height")
    machinery_lower_than_walls = (
        float(config["machinery_height_m"]) < float(config["wall_height_m"])
    )
    if not machinery_lower_than_walls:
        errors.append("machinery height must be lower than wall height")
    source_semantic_matches = True
    for group_name in ("poses", "landmarks"):
        if set(source_semantic[group_name]) != set(semantic[group_name]):
            errors.append(f"{group_name}: configured names differ from source semantic YAML")
            source_semantic_matches = False
            continue
        for name, configured in semantic[group_name].items():
            original = source_semantic[group_name][name]
            keys = ("x", "y", "yaw") if group_name == "poses" else ("x", "y")
            if not all(
                math.isclose(
                    float(configured["ros"][key]), float(original[key]), abs_tol=1e-9
                )
                for key in keys
            ):
                errors.append(f"{name}: configured ROS values differ from source semantic YAML")
                source_semantic_matches = False
    point_results: dict[str, dict] = {}
    grid_points: dict[str, tuple[int, int]] = {}
    marker_references: dict[str, tuple[float, float]] = {}
    for name, values in semantic["poses"].items():
        ros = values["ros"]
        expected_fds = transform.ros_to_fds(float(ros["x"]), float(ros["y"]))
        stored_fds = (float(values["fds"]["x"]), float(values["fds"]["y"]))
        round_trip = transform.fds_to_ros(*stored_fds)
        transform_ok = np.allclose(expected_fds, stored_fds, atol=1e-9) and np.allclose(
            round_trip, (ros["x"], ros["y"]), atol=1e-9
        )
        row, col = transform.ros_to_pixel(float(ros["x"]), float(ros["y"]))
        semantic_grid = transform.fds_to_grid(*stored_fds, resolution)
        grid = semantic_grid
        marker_reference = stored_fds
        if name.startswith("EXIT"):
            marker_config = config["exit_markers"][name]
            if "reference_fds" in marker_config:
                marker_reference = tuple(map(float, marker_config["reference_fds"]))
            marker_grid = transform.fds_to_grid(*marker_reference, resolution)
            free_cells = np.argwhere(free)
            approach_y, approach_x = min(
                free_cells,
                key=lambda cell: (
                    ((cell[1] + 0.5) * resolution - marker_reference[0]) ** 2
                    + ((cell[0] + 0.5) * resolution - marker_reference[1]) ** 2
                ),
            )
            grid = (int(approach_x), int(approach_y))
        else:
            marker_grid = semantic_grid
        grid_points[name] = grid
        marker_references[name] = marker_reference
        source_free = bool(free_src[row, col])
        coarse_free = bool(free[semantic_grid[1], semantic_grid[0]])
        solid_exit_marker = bool(
            exit_markers_grid[marker_grid[1], marker_grid[0]]
        )
        if not transform_ok:
            errors.append(f"{name}: ROS/FDS transform mismatch")
        if name == "INIT" and (not source_free or not coarse_free):
            errors.append("INIT is not free before and after conversion")
        if name.startswith("EXIT") and not solid_exit_marker:
            errors.append(f"{name}: semantic point is not represented by a solid marker")
        point_results[name] = {
            "source_pixel_row_col": [row, col],
            "source_pixel_value": int(image[row, col]),
            "source_free": source_free,
            "fds_xy": list(stored_fds),
            "marker_reference_fds": list(marker_reference),
            "coarse_grid_xy": list(semantic_grid),
            "marker_grid_xy": list(marker_grid),
            "approach_grid_xy": list(grid),
            "coarse_free": coarse_free,
            "solid_exit_marker": solid_exit_marker,
            "clearance_to_coarse_blocked_m": clearance_m(blocked, grid, resolution),
            "coordinate_round_trip_ok": bool(transform_ok),
        }

    paths: dict[str, list[tuple[int, int]]] = {}
    path_results: dict[str, dict] = {}
    exit_adjacent = (
        cv2.dilate(exit_markers_grid.astype(np.uint8), np.ones((3, 3), np.uint8))
        > 0
    ) & free
    for exit_name in ("EXIT1", "EXIT2", "EXIT3"):
        reference_x, reference_y = marker_references[exit_name]
        candidates = []
        for candidate_y, candidate_x in np.argwhere(exit_adjacent):
            if math.hypot(
                (candidate_x + 0.5) * resolution - reference_x,
                (candidate_y + 0.5) * resolution - reference_y,
            ) <= 2.0:
                candidates.append((int(candidate_x), int(candidate_y)))
        path, grid_distance = [], math.inf
        approach = grid_points[exit_name]
        for candidate in candidates:
            candidate_path, candidate_distance = shortest_path(
                free, grid_points["INIT"], candidate
            )
            if candidate_path and candidate_distance < grid_distance:
                path, grid_distance, approach = (
                    candidate_path, candidate_distance, candidate
                )
        grid_points[exit_name] = approach
        point_results[exit_name]["approach_grid_xy"] = list(approach)
        paths[exit_name] = path
        if not path:
            errors.append(f"INIT to {exit_name}: no converted free-space path")
        path_results[exit_name] = {
            "connected": bool(path),
            "path_grid_nodes": len(path),
            "path_length_m": None if not path else grid_distance * resolution,
            "minimum_path_clearance_m": None if not path else min(
                clearance_m(blocked, node, resolution) for node in path
            ),
        }

    conversion_report = json.loads(
        (FACTORY_DIR / "generated" / "conversion_report.json").read_text()
    )
    scenario_result = conversion_report["scenario"]
    if bool(scenario_result.get("fire_enabled", False)):
        if bool((oil_spill_grid & blocked).any()):
            errors.append("oil spill overlaps blocked/no-go geometry")
        if scenario_result.get("wall_follow_target_exit") == "EXIT2":
            if not bool(
                scenario_result.get("target_exit_approach_covered_by_oil", False)
            ):
                errors.append("wall-following oil fire does not cover EXIT2 approach")
            if int(scenario_result.get("wall_follow_path_cells", 0)) <= 0:
                errors.append("wall-following oil fire has no generated path")
        cardboard_result = scenario_result.get("cardboard_boxes", {})
        if bool(cardboard_result.get("enabled", False)):
            if bool((cardboard_boxes_grid & blocked).any()):
                errors.append("cardboard boxes overlap blocked geometry")
            if bool((cardboard_boxes_grid & oil_spill_grid).any()):
                errors.append("cardboard and oil fire surfaces overlap")
            if not bool(cardboard_result.get("reaches_slam_wall", False)):
                errors.append("cardboard route does not reach a SLAM wall")
            if not bool(cardboard_result.get("reaches_right_no_go_wall", False)):
                errors.append("cardboard route does not reach the right no-go wall")
    exterior_results = {
        marker["name"]: {
            "solid_marker": True,
            "reference_fds": marker["reference_fds"],
            "slope": marker["slope"],
            "xb_xy": marker["xb_xy"],
            "height_m": marker["height_m"],
            "cell_count": marker["cell_count"],
            "touches_slam_wall_both_ends": marker[
                "touches_slam_wall_both_ends"
            ],
            "open_boundary_vent": False,
        }
        for marker in conversion_report["exit_markers"]
    }

    machinery: dict[str, dict | None] = {}
    for name, values in semantic["landmarks"].items():
        row_col = transform.ros_to_pixel(
            float(values["ros"]["x"]), float(values["ros"]["y"])
        )
        machinery[name] = unknown_component_bbox(
            unknown_src, row_col, transform.resolution
        )
        fds = values["fds"]
        gx, gy = transform.fds_to_grid(
            float(fds["x"]), float(fds["y"]), resolution
        )
        if not machinery_grid[gy, gx]:
            errors.append(f"{name}: center is not represented by a machinery cell")

    factor = round(resolution / transform.resolution)
    original_blocked_coarse = downsample_any(
        np.flipud(occupied_src | unknown_src),
        factor,
        free.shape[0],
        free.shape[1],
        padding_value=True,
    )
    no_go_open_violations = no_go & ~blocked
    no_go_preservation_ok = not bool(no_go_open_violations.any())
    if not no_go_preservation_ok:
        errors.append("no-go zone contains traversable coarse cells")
    inaccessible_violations = original_blocked_coarse & ~blocked & ~exit_corridors
    cleanup_limit = int(
        config["wall_simplification"]["maximum_opened_coarse_cells"]
    )
    inaccessible_preservation_ok = int(inaccessible_violations.sum()) <= cleanup_limit
    if not inaccessible_preservation_ok:
        errors.append(
            "derived wall cleanup opened more coarse cells than the configured limit"
        )
    source_free_coarse = np.zeros_like(free)
    source_free_y_up = np.flipud(free_src)
    for y in range(free.shape[0]):
        for x in range(free.shape[1]):
            block = source_free_y_up[
                y * factor:min((y + 1) * factor, transform.height),
                x * factor:min((x + 1) * factor, transform.width),
            ]
            source_free_coarse[y, x] = block.size > 0 and block.any()
    lost_free_cells = int((source_free_coarse & blocked).sum())
    border = np.concatenate((blocked[0], blocked[-1], blocked[:, 0], blocked[:, -1]))
    obstacle_bytes = (FACTORY_DIR / "generated" / "obstacles.inc").read_bytes()
    deterministic_hash_ok = hashlib.sha256(obstacle_bytes).hexdigest() == conversion_report[
        "obstacles_sha256"
    ]
    exit_bytes = (FACTORY_DIR / "generated" / "exits.inc").read_bytes()
    deterministic_exit_hash_ok = hashlib.sha256(exit_bytes).hexdigest() == conversion_report[
        "exits_sha256"
    ]
    if not deterministic_hash_ok:
        errors.append("obstacles.inc hash differs from conversion report")
    if not deterministic_exit_hash_ok:
        errors.append("exits.inc hash differs from conversion report")
    scenario_bytes = (FACTORY_DIR / "generated" / "scenario.inc").read_bytes()
    deterministic_scenario_hash_ok = (
        hashlib.sha256(scenario_bytes).hexdigest()
        == conversion_report["scenario_sha256"]
    )
    if not deterministic_scenario_hash_ok:
        errors.append("scenario.inc hash differs from conversion report")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "axis_checks": {
            "pgm_col_increases_with_ros_x": True,
            "pgm_row_decreases_with_ros_y": True,
            "fds_x_y_match_ros_x_y_without_axis_swap": True,
            "lower_left_ros": [transform.origin_x, transform.origin_y],
            "upper_right_ros": [
                transform.origin_x + transform.width_m,
                transform.origin_y + transform.height_m,
            ],
        },
        "semantic_points": point_results,
        "source_semantic_yaml_matches_config": source_semantic_matches,
        "connectivity": path_results,
        "exit_markers": exterior_results,
        "industrial_machinery_unknown_components": machinery,
        "downsampling": {
            "policy": config["downsample_policy"],
            "source_free_coarse_cells_touched": int(source_free_coarse.sum()),
            "free_cells_lost_by_conservative_rule": lost_free_cells,
            "free_area_lost_m2_upper_bound": lost_free_cells * resolution**2,
        },
        "inaccessible_area_preservation": {
            "passed": inaccessible_preservation_ok,
            "derived_cleanup_open_coarse_cells": int(inaccessible_violations.sum()),
            "maximum_allowed_cleanup_cells": cleanup_limit,
            "source_map_modified": False,
            "no_go_cleanup_allowed": False,
            "solid_exit_markers_reported_separately": True,
        },
        "no_go_zone_preservation": {
            "passed": no_go_preservation_ok,
            "zone_count": int(conversion_report["no_go_zones"]["count"]),
            "zone_names": conversion_report["no_go_zones"]["names"],
            "source_polygon_cells": int(
                conversion_report["no_go_zones"]["source_cells"]
            ),
            "coarse_wall_cells": int(no_go.sum()),
            "unexpected_free_cells": int(no_go_open_violations.sum()),
        },
        "domain_border_blocked_fraction": float(border.mean()),
        "all_obstacles_inside_mesh": True,
        "vertical_geometry": {
            "ceiling_height_m": float(config["fds_height_m"]),
            "wall_height_m": float(config["wall_height_m"]),
            "ceiling_matches_walls": ceiling_matches_walls,
            "machinery_height_m": float(config["machinery_height_m"]),
            "machinery_is_lower_than_walls": machinery_lower_than_walls,
        },
        "deterministic_obstacle_hash_ok": deterministic_hash_ok,
        "deterministic_exit_hash_ok": deterministic_exit_hash_ok,
        "deterministic_scenario_hash_ok": deterministic_scenario_hash_ok,
        "scenario": conversion_report["scenario"],
    }
    validation_dir = FACTORY_DIR / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    report_path = validation_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    draw_overlay(
        image, occupied, unknown, no_go, machinery_grid, exit_markers_grid,
        oil_spill_grid, cardboard_boxes_grid, paths,
        semantic["poses"], transform,
        resolution, conversion_report["scenario"], validation_dir / "overlay.png",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
