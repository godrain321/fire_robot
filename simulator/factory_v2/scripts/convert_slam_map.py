#!/usr/bin/env python3
"""Convert the fixed SLAM occupancy grid into merged FDS OBST records."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import yaml
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
FACTORY_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from map_coordinates import MapTransform  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return data


def resolve_from_factory(value: str) -> Path:
    return (FACTORY_DIR / value).resolve()


def classify_pixels(image: np.ndarray, metadata: dict) -> tuple[np.ndarray, ...]:
    occupancy = (255.0 - image.astype(float)) / 255.0
    if int(metadata["negate"]) != 0:
        occupancy = image.astype(float) / 255.0
    occupied = occupancy > float(metadata["occupied_thresh"])
    free = occupancy < float(metadata["free_thresh"])
    unknown = ~(occupied | free)
    return occupied, free, unknown


def rasterize_no_go_zones(
    zones_document: dict, transform: MapTransform
) -> tuple[np.ndarray, list[dict]]:
    """Rasterize ROS-map polygon zones using the map-tools pixel convention."""
    zones = zones_document.get("no_go_zones")
    if not isinstance(zones, list):
        raise ValueError("no_go_zones must be a list")
    image = Image.new("L", (transform.width, transform.height), 0)
    draw = ImageDraw.Draw(image)
    parsed = []
    used_names = set()
    for index, zone in enumerate(zones):
        if not isinstance(zone, dict) or zone.get("type") != "polygon":
            raise ValueError(f"no_go_zones[{index}] must be a polygon mapping")
        name = str(zone.get("name", "")).strip()
        points = zone.get("points")
        if not name or name in used_names:
            raise ValueError(f"invalid or duplicate no-go zone name: {name!r}")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"{name}: polygon needs at least three points")
        used_names.add(name)
        pixels = []
        ros_points = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"{name}: each point must be [x, y]")
            x_ros, y_ros = float(point[0]), float(point[1])
            row, col = transform.ros_to_pixel(x_ros, y_ros)
            pixels.append((col, row))
            ros_points.append([x_ros, y_ros])
        draw.polygon(pixels, fill=255)
        parsed.append({"name": name, "points_ros": ros_points})
    return np.asarray(image, dtype=np.uint8) > 0, parsed


def extract_machinery_mask(
    unknown: np.ndarray,
    occupied: np.ndarray,
    semantic: dict,
    transform: MapTransform,
    outline_dilation_cells: int,
) -> tuple[np.ndarray, list[dict]]:
    """Extract the SLAM unknown component and its immediate occupied outline."""
    if outline_dilation_cells < 0:
        raise ValueError("machinery outline dilation must be non-negative")
    count, labels = cv2.connectedComponents(unknown.astype(np.uint8), connectivity=4)
    del count
    machinery = np.zeros_like(unknown)
    records = []
    for name, values in semantic["landmarks"].items():
        if not name.startswith("INDUSTRIAL_MACHINERY_"):
            continue
        row, col = transform.ros_to_pixel(
            float(values["ros"]["x"]), float(values["ros"]["y"])
        )
        label = int(labels[row, col])
        if label == 0:
            raise ValueError(f"{name} center is not inside an unknown map component")
        component = labels == label
        # Restore the native SLAM footprint instead of replacing it with a
        # rectangular bounding box. One source pixel captures its thin outline.
        kernel_size = outline_dilation_cells * 2 + 1
        expanded = cv2.dilate(
            component.astype(np.uint8),
            np.ones((kernel_size, kernel_size), np.uint8),
        ) > 0
        footprint = expanded & (unknown | occupied)
        machinery |= footprint
        ys, xs = np.where(footprint)
        records.append({
            "name": name,
            "source_cells": int(footprint.sum()),
            "outline_dilation_cells": outline_dilation_cells,
            "source_pixel_bbox": [
                int(xs.min()), int(xs.max()) + 1,
                int(ys.min()), int(ys.max()) + 1,
            ],
        })
    return machinery, records


def downsample_any(
    mask_y_up: np.ndarray,
    factor: int,
    ny: int,
    nx: int,
    *,
    padding_value: bool = False,
) -> np.ndarray:
    padded = np.full((ny * factor, nx * factor), padding_value, dtype=bool)
    height, width = mask_y_up.shape
    padded[:height, :width] = mask_y_up
    return padded.reshape(ny, factor, nx, factor).any(axis=(1, 3))


def simplify_free_space(
    free: np.ndarray,
    resolution: float,
    tolerance_m: float,
    minimum_area_m2: float,
) -> np.ndarray:
    """Approximate mapped free-space contours with clean straight segments."""
    source = (free.astype(np.uint8) * 255)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        source, connectivity=8
    )
    minimum_cells = max(1, math.ceil(minimum_area_m2 / resolution**2))
    retained = np.zeros_like(source)
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_cells:
            retained[labels == label] = 255

    contours, hierarchy = cv2.findContours(
        retained, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return np.zeros_like(free)
    simplified = np.zeros_like(source)
    epsilon = tolerance_m / resolution
    records = []
    for index, contour in enumerate(contours):
        depth = 0
        parent = int(hierarchy[0][index][3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[0][parent][3])
        approximation = cv2.approxPolyDP(contour, epsilon, True)
        records.append((depth, index, approximation))
    for depth, _, approximation in sorted(records):
        color = 255 if depth % 2 == 0 else 0
        cv2.drawContours(simplified, [approximation], -1, color, thickness=cv2.FILLED)
    return simplified.astype(bool)


def ray_boundary_hit(
    x: float, y: float, dx: float, dy: float, width: float, height: float
) -> tuple[float, str, float, float]:
    """Return the first positive intersection of a ray and the FDS boundary."""
    candidates = []
    if dx > 0.0:
        candidates.append(((width - x) / dx, "XMAX"))
    elif dx < 0.0:
        candidates.append(((0.0 - x) / dx, "XMIN"))
    if dy > 0.0:
        candidates.append(((height - y) / dy, "YMAX"))
    elif dy < 0.0:
        candidates.append(((0.0 - y) / dy, "YMIN"))
    positive = [(distance, side) for distance, side in candidates if distance >= 0.0]
    distance, side = min(positive)
    return distance, side, x + distance * dx, y + distance * dy


def route_around_no_go(
    no_go: np.ndarray,
    start: tuple[int, int],
    side: str,
    hit_x: float,
    hit_y: float,
    resolution: float,
) -> list[tuple[int, int]]:
    """Route from an exit to its intended boundary without entering no-go cells."""
    height, width = no_go.shape
    if side in ("XMIN", "XMAX"):
        boundary_x = 0 if side == "XMIN" else width - 1
        preferred = round(hit_y / resolution - 0.5)
        goals = [(boundary_x, y) for y in range(height) if not no_go[y, boundary_x]]
        goals.sort(key=lambda node: abs(node[1] - preferred))
    else:
        boundary_y = 0 if side == "YMIN" else height - 1
        preferred = round(hit_x / resolution - 0.5)
        goals = [(x, boundary_y) for x in range(width) if not no_go[boundary_y, x]]
        goals.sort(key=lambda node: abs(node[0] - preferred))
    if not goals:
        raise ValueError(f"no non-no-go boundary cell is available on {side}")
    goal = goals[0]

    moves = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    )
    frontier = [(0.0, start)]
    distance = {start: 0.0}
    parent = {start: None}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        current_cost = distance[current]
        for dx, dy, step in moves:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if no_go[nxt[1], nxt[0]]:
                continue
            if dx and dy and (
                no_go[current[1], current[0] + dx]
                or no_go[current[1] + dy, current[0]]
            ):
                continue
            new_cost = current_cost + step
            if new_cost < distance.get(nxt, math.inf):
                distance[nxt] = new_cost
                parent[nxt] = current
                heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(frontier, (new_cost + heuristic, nxt))
    if goal not in parent:
        raise ValueError(f"no route from exit grid {start} to boundary {side}")
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


def create_exit_markers(
    slam_wall: np.ndarray,
    semantic: dict,
    resolution: float,
    opening_config: dict,
) -> tuple[list[dict], np.ndarray]:
    """Rasterize configured sloped solid markers without opening the map wall."""
    height_cells, width_cells = slam_wall.shape
    thickness = float(opening_config["thickness_m"])
    if thickness <= 0.0:
        raise ValueError("exit marker thickness must be positive")
    markers = []
    marker_mask = np.zeros_like(slam_wall)

    def grid_cell(x: float, y: float) -> tuple[int, int] | None:
        gx, gy = math.floor(x / resolution), math.floor(y / resolution)
        if 0 <= gx < width_cells and 0 <= gy < height_cells:
            return gx, gy
        return None

    def trace_to_wall(
        center_x: float, center_y: float, dx: float, dy: float, sign: float
    ) -> tuple[list[tuple[int, int]], bool]:
        cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        step = resolution / 10.0
        maximum = math.hypot(width_cells * resolution, height_cells * resolution)
        distance = 0.0
        while distance <= maximum:
            cell = grid_cell(
                center_x + sign * distance * dx,
                center_y + sign * distance * dy,
            )
            if cell is None:
                return cells, False
            gx, gy = cell
            if slam_wall[gy, gx]:
                return cells, True
            if cell not in seen:
                seen.add(cell)
                cells.append(cell)
            distance += step
        return cells, False

    def trace_segment(
        center_x: float, center_y: float, dx: float, dy: float, length: float
    ) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        sample_count = max(2, math.ceil(length / (resolution / 10.0)))
        for along in np.linspace(-0.5 * length, 0.5 * length, sample_count + 1):
            cell = grid_cell(center_x + along * dx, center_y + along * dy)
            if cell is not None and cell not in seen:
                seen.add(cell)
                cells.append(cell)
        return cells

    for name in ("EXIT1", "EXIT2", "EXIT3"):
        marker_config = opening_config[name]
        if marker_config.get("reference") == "semantic_center":
            pose = semantic["poses"][name]["fds"]
            center_x, center_y = float(pose["x"]), float(pose["y"])
        else:
            center_x, center_y = map(float, marker_config["reference_fds"])
        slope = float(marker_config["slope"])
        normalization = math.hypot(1.0, slope)
        dx, dy = 1.0 / normalization, slope / normalization
        touches_slam_wall = None
        if marker_config.get("extent") == "until_slam_wall":
            negative, negative_hit = trace_to_wall(center_x, center_y, dx, dy, -1.0)
            positive, positive_hit = trace_to_wall(center_x, center_y, dx, dy, 1.0)
            if not negative_hit or not positive_hit:
                raise ValueError(f"{name}: sloped marker did not reach SLAM walls")
            cells = list(reversed(negative)) + positive[1:]
            touches_slam_wall = True
            requested_length = None
        else:
            requested_length = float(marker_config["length_m"])
            cells = trace_segment(center_x, center_y, dx, dy, requested_length)
        marker = np.zeros_like(slam_wall)
        for gx, gy in cells:
            marker[gy, gx] = True
        if not marker.any():
            raise ValueError(f"{name}: marker rasterization produced no cells")
        marker_mask |= marker
        ys, xs = np.where(marker)
        rectangles = merge_rectangles(marker)
        markers.append({
            "name": name,
            "reference_fds": [center_x, center_y],
            "slope": slope,
            "xb_xy": [
                int(xs.min()) * resolution,
                (int(xs.max()) + 1) * resolution,
                int(ys.min()) * resolution,
                (int(ys.max()) + 1) * resolution,
            ],
            "requested_length_m": requested_length,
            "thickness_m": thickness,
            "height_m": float(opening_config["height_m"]),
            "cell_count": int(marker.sum()),
            "rectangles": [list(rectangle) for rectangle in rectangles],
            "touches_slam_wall_both_ends": touches_slam_wall,
        })
    return markers, marker_mask


def fds_exit_text(markers: list[dict], resolution: float) -> str:
    lines = [
        "! AUTO-GENERATED solid EXIT markers; do not edit.",
        "! These are configured colored wall markers, not OPEN vents.",
        "",
    ]
    for marker in markers:
        for index, (gx0, gx1, gy0, gy1) in enumerate(
            marker["rectangles"], start=1
        ):
            lines.append(
                f"&OBST ID='{marker['name']}_SOLID_{index:03d}', "
                f"XB={gx0 * resolution:.3f},{gx1 * resolution:.3f},"
                f"{gy0 * resolution:.3f},{gy1 * resolution:.3f},"
                f"0.000,{marker['height_m']:.3f}, SURF_ID='EXIT_SOLID' /"
            )
    lines.append("")
    return "\n".join(lines)


def fds_scenario_text(
    scenario: dict,
    blocked: np.ndarray,
    no_go_mask: np.ndarray,
    slam_wall_mask: np.ndarray,
    resolution: float,
    semantic: dict,
    exit_marker_records: list[dict],
) -> tuple[str, dict, np.ndarray, np.ndarray]:
    """Generate a machine-adjacent fire spreading over connected oil cells."""
    fire = scenario.get("fire", {})
    if not isinstance(fire, dict) or not bool(fire.get("enabled", False)):
        return (
            "! AUTO-GENERATED scenario: fire disabled.\n",
            {"fire_enabled": False},
            np.zeros_like(blocked),
            np.zeros_like(blocked),
        )
    fire_id = str(fire["id"])
    hrrpua = float(fire["hrrpua_kw_m2"])
    if hrrpua <= 0.0:
        raise ValueError("fire HRRPUA must be positive")
    reference_x, reference_y = map(float, fire["selection_reference_fds"])
    machines = semantic["landmarks"]
    machine_name, machine = min(
        machines.items(),
        key=lambda item: math.hypot(
            float(item[1]["fds"]["x"]) - reference_x,
            float(item[1]["fds"]["y"]) - reference_y,
        ),
    )
    machine_x = float(machine["fds"]["x"])
    machine_y = float(machine["fds"]["y"])
    free = ~blocked
    free_cells = np.argwhere(free)
    if free_cells.size == 0:
        raise ValueError("no free cell is available for machinery ignition")
    ignition_gy, ignition_gx = min(
        free_cells,
        key=lambda cell: (
            ((cell[1] + 0.5) * resolution - machine_x) ** 2
            + ((cell[0] + 0.5) * resolution - machine_y) ** 2
        ),
    )
    ignition_gx, ignition_gy = int(ignition_gx), int(ignition_gy)
    ignition_x = (ignition_gx + 0.5) * resolution
    ignition_y = (ignition_gy + 0.5) * resolution
    spill = fire["oil_spill"]
    radius = float(spill["radius_m"])
    spread_rate = float(spill["spread_rate_m_s"])
    if radius <= 0.0 or spread_rate <= 0.0:
        raise ValueError("oil spill radius and spread rate must be positive")

    oil_mask = np.zeros_like(blocked)
    queue = [(ignition_gx, ignition_gy)]
    oil_mask[ignition_gy, ignition_gx] = True
    queue_index = 0
    while queue_index < len(queue):
        gx, gy = queue[queue_index]
        queue_index += 1
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = gx + dx, gy + dy
            if not (0 <= nx < free.shape[1] and 0 <= ny < free.shape[0]):
                continue
            if oil_mask[ny, nx] or not free[ny, nx]:
                continue
            cell_x = (nx + 0.5) * resolution
            cell_y = (ny + 0.5) * resolution
            if math.hypot(cell_x - ignition_x, cell_y - ignition_y) > radius:
                continue
            oil_mask[ny, nx] = True
            queue.append((nx, ny))

    target_exit = str(spill.get("follow_no_go_wall_to_exit", "")).strip()
    wall_path: list[tuple[int, int]] = []
    if target_exit:
        if target_exit not in semantic["poses"]:
            raise ValueError(f"unknown fire spread target exit: {target_exit}")
        target_marker = next(
            marker for marker in exit_marker_records
            if marker["name"] == target_exit
        )
        target_x, target_y = map(float, target_marker["reference_fds"])
        semantic_target = (
            min(max(math.floor(target_x / resolution), 0), free.shape[1] - 1),
            min(max(math.floor(target_y / resolution), 0), free.shape[0] - 1),
        )
        target_candidates = np.argwhere(free)
        target_gy, target_gx = min(
            target_candidates,
            key=lambda cell: (
                ((cell[1] + 0.5) * resolution - target_x) ** 2
                + ((cell[0] + 0.5) * resolution - target_y) ** 2
            ),
        )
        target = (int(target_gx), int(target_gy))
        wall_weight = float(spill.get("wall_follow_weight", 4.0))
        if wall_weight < 0.0:
            raise ValueError("wall_follow_weight must be non-negative")
        # Distance is measured from the configured no-go solid. A weighted
        # Dijkstra route remains on free floor but prefers cells beside its wall.
        wall_distance = cv2.distanceTransform(
            (~no_go_mask).astype(np.uint8), cv2.DIST_L2, 3
        )
        start = (ignition_gx, ignition_gy)
        frontier = [(0.0, start)]
        distance = {start: 0.0}
        parent = {start: None}
        while frontier:
            cost, current = heapq.heappop(frontier)
            if cost > distance[current]:
                continue
            if current == target:
                break
            for dx, dy, step in (
                (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            ):
                nxt = (current[0] + dx, current[1] + dy)
                if not (
                    0 <= nxt[0] < free.shape[1]
                    and 0 <= nxt[1] < free.shape[0]
                    and free[nxt[1], nxt[0]]
                ):
                    continue
                proximity_cost = wall_weight * min(
                    float(wall_distance[nxt[1], nxt[0]]), 10.0
                )
                new_cost = cost + step + proximity_cost
                if new_cost < distance.get(nxt, math.inf):
                    distance[nxt] = new_cost
                    parent[nxt] = current
                    heapq.heappush(frontier, (new_cost, nxt))
        if target not in parent:
            raise ValueError(f"no free-floor fire route reaches {target_exit}")
        node = target
        while node is not None:
            wall_path.append(node)
            node = parent[node]
        wall_path.reverse()
        path_mask = np.zeros_like(blocked)
        for gx, gy in wall_path:
            path_mask[gy, gx] = True
        half_width_cells = max(
            0, math.ceil(float(spill.get("path_half_width_m", 0.0)) / resolution)
        )
        if half_width_cells:
            kernel = np.ones(
                (half_width_cells * 2 + 1, half_width_cells * 2 + 1), np.uint8
            )
            path_mask = cv2.dilate(path_mask.astype(np.uint8), kernel) > 0
        exit_radius = float(spill.get("exit_block_radius_m", 0.0))
        yy, xx = np.indices(blocked.shape)
        exit_mask = (
            ((xx + 0.5) * resolution - target_x) ** 2
            + ((yy + 0.5) * resolution - target_y) ** 2
            <= exit_radius**2
        )
        oil_mask |= (path_mask | exit_mask) & free

    cardboard_config = fire.get("cardboard_boxes", {})
    cardboard_mask = np.zeros_like(blocked)
    cardboard_path: list[tuple[int, int]] = []
    cardboard_target_name = None
    cardboard_wall_goal = None
    right_no_go_goal = None
    cardboard_origin = (ignition_gx, ignition_gy)
    if bool(cardboard_config.get("enabled", False)):
        other_machines = [
            (name, values) for name, values in machines.items()
            if name != machine_name
        ]
        if not other_machines:
            raise ValueError("cardboard spread needs another industrial machinery")
        cardboard_target_name, cardboard_target = min(
            other_machines,
            key=lambda item: math.hypot(
                float(item[1]["fds"]["x"]) - machine_x,
                float(item[1]["fds"]["y"]) - machine_y,
            ),
        )
        target_machine_x = float(cardboard_target["fds"]["x"])
        target_machine_y = float(cardboard_target["fds"]["y"])

        def nearest_cell(mask: np.ndarray, x: float, y: float) -> tuple[int, int]:
            candidates = np.argwhere(mask)
            if not candidates.size:
                raise ValueError("cardboard route has no candidate floor cell")
            gy, gx = min(
                candidates,
                key=lambda cell: (
                    ((cell[1] + 0.5) * resolution - x) ** 2
                    + ((cell[0] + 0.5) * resolution - y) ** 2
                ),
            )
            return int(gx), int(gy)

        def floor_path(
            start: tuple[int, int], goal: tuple[int, int]
        ) -> list[tuple[int, int]]:
            frontier = [start]
            parent = {start: None}
            cursor = 0
            while cursor < len(frontier):
                current = frontier[cursor]
                cursor += 1
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
                    frontier.append(nxt)
            if goal not in parent:
                raise ValueError("cardboard floor route is disconnected")
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))

        def wall_preferred_path(
            start: tuple[int, int],
            goal: tuple[int, int],
            preferred_wall: np.ndarray,
            weight: float,
        ) -> list[tuple[int, int]]:
            wall_distance = cv2.distanceTransform(
                (~preferred_wall).astype(np.uint8), cv2.DIST_L2, 3
            )
            frontier = [(0.0, start)]
            distance = {start: 0.0}
            parent = {start: None}
            while frontier:
                cost, current = heapq.heappop(frontier)
                if cost > distance[current]:
                    continue
                if current == goal:
                    break
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nxt = current[0] + dx, current[1] + dy
                    if not (
                        0 <= nxt[0] < free.shape[1]
                        and 0 <= nxt[1] < free.shape[0]
                        and free[nxt[1], nxt[0]]
                    ):
                        continue
                    new_cost = cost + 1.0 + weight * min(
                        float(wall_distance[nxt[1], nxt[0]]), 10.0
                    )
                    if new_cost < distance.get(nxt, math.inf):
                        distance[nxt] = new_cost
                        parent[nxt] = current
                        heapq.heappush(frontier, (new_cost, nxt))
            if goal not in parent:
                raise ValueError("cardboard no-go wall route is disconnected")
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))

        cardboard_origin = nearest_cell(oil_mask, target_machine_x, target_machine_y)
        target_floor = nearest_cell(free, target_machine_x, target_machine_y)
        wall_adjacent_floor = (
            cv2.dilate(
                slam_wall_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
            ) > 0
        ) & free
        yy, xx = np.indices(free.shape)
        upper_slam_wall_floor = wall_adjacent_floor & (
            (yy + 0.5) * resolution > target_machine_y
        )
        cardboard_wall_goal = nearest_cell(
            upper_slam_wall_floor, target_machine_x, target_machine_y
        )
        no_go_adjacent_floor = (
            cv2.dilate(no_go_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ) & free
        right_no_go_floor = no_go_adjacent_floor & (
            (xx + 0.5) * resolution > target_machine_x
        )
        right_no_go_goal = nearest_cell(
            right_no_go_floor, target_machine_x, target_machine_y
        )
        no_go_weight = float(
            cardboard_config.get("no_go_wall_follow_weight", 6.0)
        )
        if no_go_weight < 0.0:
            raise ValueError("cardboard no-go wall weight must be non-negative")
        cardboard_path = []
        if bool(cardboard_config.get("follow_right_no_go_wall", True)):
            cardboard_path = wall_preferred_path(
                cardboard_origin, right_no_go_goal, no_go_mask, no_go_weight
            )
            cardboard_path += floor_path(right_no_go_goal, target_floor)[1:]
        else:
            cardboard_path = floor_path(cardboard_origin, target_floor)
        if bool(cardboard_config.get("extend_to_upper_slam_wall", True)):
            cardboard_path += floor_path(target_floor, cardboard_wall_goal)[1:]
        for gx, gy in cardboard_path:
            cardboard_mask[gy, gx] = True
        half_width_cells = max(
            0,
            math.ceil(
                float(cardboard_config.get("path_half_width_m", 0.0))
                / resolution
            ),
        )
        if half_width_cells:
            kernel = np.ones(
                (half_width_cells * 2 + 1, half_width_cells * 2 + 1),
                np.uint8,
            )
            cardboard_mask = (
                cv2.dilate(cardboard_mask.astype(np.uint8), kernel) > 0
            )
        cardboard_mask &= free & ~oil_mask

    oil_rectangles = merge_rectangles(oil_mask)
    if not oil_rectangles:
        raise ValueError("oil spill did not produce a burnable surface")
    ramp_id = f"{fire_id}_RAMP"
    surface_id = f"{fire_id}_OIL"
    lines = [
        "! AUTO-GENERATED scenario; edit config/scenario.yaml, not this file.",
        f"&SURF ID='{surface_id}', HRRPUA={hrrpua:.3f}, RAMP_Q='{ramp_id}', COLOR='RED' /",
    ]
    ramp = fire.get("ramp", [])
    if not isinstance(ramp, list) or not ramp:
        raise ValueError("enabled fire requires at least one ramp point")
    for point in ramp:
        lines.append(
            f"&RAMP ID='{ramp_id}', T={float(point['time_s']):.3f}, "
            f"F={float(point['fraction']):.6f} /"
        )
    for index, (gx0, gx1, gy0, gy1) in enumerate(oil_rectangles, start=1):
        x0, x1 = gx0 * resolution, gx1 * resolution
        y0, y1 = gy0 * resolution, gy1 * resolution
        lines.append(
            f"&VENT ID='{fire_id}_SPILL_{index:03d}', "
            f"XB={x0:.3f},{x1:.3f},{y0:.3f},{y1:.3f},0.000,0.000, "
            f"SURF_ID='{surface_id}', XYZ={ignition_x:.3f},{ignition_y:.3f},0.000, "
            f"SPREAD_RATE={spread_rate:.4f} /"
        )

    cardboard_rectangles = merge_rectangles(cardboard_mask)
    cardboard_hrrpua = float(cardboard_config.get("hrrpua_kw_m2", 0.0))
    cardboard_spread_rate = float(cardboard_config.get("spread_rate_m_s", 0.0))
    if bool(cardboard_config.get("enabled", False)):
        if cardboard_hrrpua <= 0.0 or cardboard_spread_rate <= 0.0:
            raise ValueError("cardboard HRRPUA and spread rate must be positive")
        if not cardboard_rectangles:
            raise ValueError("cardboard route produced no burnable floor surface")
        cardboard_surface_id = f"{fire_id}_CARDBOARD"
        cardboard_origin_x = (cardboard_origin[0] + 0.5) * resolution
        cardboard_origin_y = (cardboard_origin[1] + 0.5) * resolution
        lines.append(
            f"&SURF ID='{cardboard_surface_id}', "
            f"HRRPUA={cardboard_hrrpua:.3f}, RAMP_Q='{ramp_id}', COLOR='BROWN' /"
        )
        for index, (gx0, gx1, gy0, gy1) in enumerate(
            cardboard_rectangles, start=1
        ):
            lines.append(
                f"&VENT ID='{fire_id}_CARDBOARD_{index:03d}', "
                f"XB={gx0 * resolution:.3f},{gx1 * resolution:.3f},"
                f"{gy0 * resolution:.3f},{gy1 * resolution:.3f},0.000,0.000, "
                f"SURF_ID='{cardboard_surface_id}', "
                f"XYZ={cardboard_origin_x:.3f},{cardboard_origin_y:.3f},0.000, "
                f"SPREAD_RATE={cardboard_spread_rate:.4f} /"
            )
    lines.append("")
    oil_y, oil_x = np.where(oil_mask)
    oil_center_x = (oil_x + 0.5) * resolution
    oil_center_y = (oil_y + 0.5) * resolution
    maximum_spread_distance = float(np.hypot(
        oil_center_x - ignition_x, oil_center_y - ignition_y
    ).max())
    spill_bbox = [
        int(oil_x.min()) * resolution,
        (int(oil_x.max()) + 1) * resolution,
        int(oil_y.min()) * resolution,
        (int(oil_y.max()) + 1) * resolution,
    ]
    cardboard_report = {"enabled": False}
    if cardboard_rectangles:
        cardboard_y, cardboard_x = np.where(cardboard_mask)
        cardboard_distances = np.hypot(
            (cardboard_x + 0.5) * resolution - cardboard_origin_x,
            (cardboard_y + 0.5) * resolution - cardboard_origin_y,
        )
        cardboard_report = {
            "enabled": True,
            "target_machinery": cardboard_target_name,
            "hrrpua_kw_m2": cardboard_hrrpua,
            "spread_rate_m_s": cardboard_spread_rate,
            "path_cells": len(cardboard_path),
            "surface_cells": int(cardboard_mask.sum()),
            "surface_area_m2": float(cardboard_mask.sum() * resolution**2),
            "rectangles": len(cardboard_rectangles),
            "origin_fds": [cardboard_origin_x, cardboard_origin_y],
            "nearest_wall_goal_grid": list(cardboard_wall_goal),
            "right_no_go_goal_grid": list(right_no_go_goal),
            "follows_right_no_go_wall": bool(
                cardboard_config.get("follow_right_no_go_wall", True)
            ),
            "extends_to_upper_slam_wall": bool(
                cardboard_config.get("extend_to_upper_slam_wall", True)
            ),
            "reaches_slam_wall": bool(
                wall_adjacent_floor[
                    cardboard_wall_goal[1], cardboard_wall_goal[0]
                ]
            ),
            "reaches_right_no_go_wall": bool(
                right_no_go_floor[right_no_go_goal[1], right_no_go_goal[0]]
            ),
            "excludes_blocked_geometry": not bool(
                (cardboard_mask & blocked).any()
            ),
            "overlaps_oil_surface": bool((cardboard_mask & oil_mask).any()),
            "maximum_spread_distance_m": float(cardboard_distances.max()),
            "estimated_full_spread_time_s": float(
                cardboard_distances.max() / cardboard_spread_rate
            ),
        }
    report = {
        "fire_enabled": True,
        "id": fire_id,
        "selection_reference_fds": [reference_x, reference_y],
        "selected_machinery": machine_name,
        "selected_machinery_center_fds": [machine_x, machine_y],
        "distance_reference_to_machinery_m": math.hypot(
            reference_x - machine_x, reference_y - machine_y
        ),
        "ignition_center_fds": [ignition_x, ignition_y],
        "ignition_distance_to_machinery_m": math.hypot(
            ignition_x - machine_x, ignition_y - machine_y
        ),
        "hrrpua_kw_m2": hrrpua,
        "oil_spill_radius_m": radius,
        "oil_spread_rate_m_s": spread_rate,
        "maximum_spread_distance_m": maximum_spread_distance,
        "estimated_full_spread_time_s": maximum_spread_distance / spread_rate,
        "oil_spill_cells": int(oil_mask.sum()),
        "oil_spill_area_m2": float(oil_mask.sum() * resolution**2),
        "oil_spill_rectangles": len(oil_rectangles),
        "oil_spill_bbox": spill_bbox,
        "oil_spill_excludes_blocked_and_no_go": not bool((oil_mask & blocked).any()),
        "wall_follow_target_exit": target_exit or None,
        "wall_follow_path_cells": len(wall_path),
        "target_exit_approach_covered_by_oil": (
            bool(oil_mask[target[1], target[0]]) if target_exit else False
        ),
        "target_exit_semantic_grid": (
            list(semantic_target) if target_exit else None
        ),
        "target_exit_fire_approach_grid": list(target) if target_exit else None,
        "cardboard_boxes": cardboard_report,
    }
    return "\n".join(lines), report, oil_mask, cardboard_mask


def merge_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Merge identical horizontal runs through consecutive rows deterministically."""
    rectangles: list[tuple[int, int, int, int]] = []
    active: dict[tuple[int, int], int] = {}
    height, width = mask.shape
    for y in range(height):
        runs: list[tuple[int, int]] = []
        x = 0
        while x < width:
            if not mask[y, x]:
                x += 1
                continue
            x_start = x
            while x < width and mask[y, x]:
                x += 1
            runs.append((x_start, x))
        run_set = set(runs)
        for run, y_start in sorted(active.items()):
            if run not in run_set:
                rectangles.append((run[0], run[1], y_start, y))
        active = {
            run: active.get(run, y)
            for run in runs
        }
    for run, y_start in sorted(active.items()):
        rectangles.append((run[0], run[1], y_start, height))
    return sorted(rectangles, key=lambda item: (item[2], item[0], item[3], item[1]))


def fds_obstacle_text(
    occupied_rectangles: list[tuple[int, int, int, int]],
    unknown_rectangles: list[tuple[int, int, int, int]],
    no_go_rectangles: list[tuple[int, int, int, int]],
    machinery_rectangles: list[tuple[int, int, int, int]],
    resolution: float,
    wall_height: float,
    machinery_height: float,
) -> str:
    lines = [
        "! AUTO-GENERATED by scripts/convert_slam_map.py; do not edit.",
        "! Coordinates are FDS metres from the SLAM map lower-left corner.",
        "",
    ]
    groups = (
        ("SLAM_OCCUPIED", "SLAM_WALL", wall_height, occupied_rectangles),
        ("SLAM_UNKNOWN", "UNMAPPED_SOLID", wall_height, unknown_rectangles),
        ("NO_GO_ZONE", "NO_GO_WALL", wall_height, no_go_rectangles),
        ("MACHINERY", "MACHINERY", machinery_height, machinery_rectangles),
    )
    for prefix, surface, height, rectangles in groups:
        lines.append(f"! {prefix}: {len(rectangles)} merged rectangles")
        for index, (x0, x1, y0, y1) in enumerate(rectangles, start=1):
            xb = (
                x0 * resolution, x1 * resolution,
                y0 * resolution, y1 * resolution,
            )
            lines.append(
                f"&OBST ID='{prefix}_{index:04d}', "
                f"XB={xb[0]:.3f},{xb[1]:.3f},{xb[2]:.3f},{xb[3]:.3f},"
                f"0.000,{height:.3f}, SURF_ID='{surface}' /"
            )
        lines.append("")
    return "\n".join(lines)


def verify_source_metadata(config: dict, source: dict, image: np.ndarray) -> None:
    checks = {
        "image_width_cells": image.shape[1],
        "image_height_cells": image.shape[0],
        "source_resolution_m": float(source["resolution"]),
        "negate": int(source["negate"]),
        "occupied_thresh": float(source["occupied_thresh"]),
        "free_thresh": float(source["free_thresh"]),
    }
    for key, actual in checks.items():
        expected = config[key]
        if isinstance(actual, float):
            matches = math.isclose(float(expected), actual, abs_tol=1e-12)
        else:
            matches = expected == actual
        if not matches:
            raise ValueError(f"{key}: config={expected!r}, source={actual!r}")
    if not np.allclose(config["ros_origin"], source["origin"], atol=1e-12):
        raise ValueError("config ros_origin does not match source map YAML")
    if config["unknown_policy"] != "blocked":
        raise ValueError("only the safety-preserving unknown_policy=blocked is supported")
    if config["downsample_policy"] != "any_blocked":
        raise ValueError("only downsample_policy=any_blocked is supported")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=FACTORY_DIR / "config" / "map_metadata.yaml",
    )
    args = parser.parse_args()
    config = load_yaml(args.config.resolve())
    source_yaml_path = resolve_from_factory(config["source_map_yaml"])
    source_map_path = resolve_from_factory(config["source_map"])
    source = load_yaml(source_yaml_path)
    semantic = load_yaml(FACTORY_DIR / "config" / "semantic_points.yaml")
    scenario = load_yaml(FACTORY_DIR / "config" / "scenario.yaml")
    image = np.asarray(Image.open(source_map_path).convert("L"))
    verify_source_metadata(config, source, image)

    transform = MapTransform(
        image.shape[1], image.shape[0], float(source["resolution"]),
        float(source["origin"][0]), float(source["origin"][1]),
        float(source["origin"][2]),
    )
    no_go_path = resolve_from_factory(config["source_no_go_zones"])
    no_go_document = load_yaml(no_go_path)
    no_go_source, no_go_zones = rasterize_no_go_zones(no_go_document, transform)

    source_resolution = float(source["resolution"])
    fds_resolution = float(config["fds_resolution_m"])
    factor_float = fds_resolution / source_resolution
    factor = round(factor_float)
    if not math.isclose(factor_float, factor, abs_tol=1e-12):
        raise ValueError("fds_resolution_m must be an integer multiple of source resolution")

    height, width = image.shape
    nx = math.ceil(width / factor)
    ny = math.ceil(height / factor)
    occupied, free, unknown = classify_pixels(image, source)
    machinery_source, machinery_records = extract_machinery_mask(
        unknown, occupied, semantic, transform,
        int(config["machinery_outline_dilation_cells"]),
    )
    simplify_config = config["wall_simplification"]
    if bool(simplify_config["enabled"]):
        simplified_free = simplify_free_space(
            free,
            source_resolution,
            float(simplify_config["contour_tolerance_m"]),
            float(simplify_config["minimum_free_component_area_m2"]),
        )
    else:
        simplified_free = free.copy()
    source_blocked = occupied | unknown
    if bool(simplify_config.get("allow_derived_spike_cleanup", False)):
        # This changes generated geometry only. The source SLAM map remains
        # untouched, while brief contour spikes may be removed within epsilon.
        simplified_blocked = (~simplified_free) | no_go_source
    else:
        simplified_blocked = (~simplified_free) | source_blocked | no_go_source
    # Preserve occupied vs unknown visualization where possible. Any boundary
    # moved by simplification is conservatively classified as unknown solid.
    occupied_simplified = occupied & simplified_blocked
    unknown_simplified = simplified_blocked & ~occupied_simplified
    # Array row zero becomes the lower (south) edge in all generated grids.
    occupied_ds = downsample_any(np.flipud(occupied_simplified), factor, ny, nx)
    unknown_ds = downsample_any(
        np.flipud(unknown_simplified), factor, ny, nx, padding_value=True
    )
    no_go_ds = downsample_any(np.flipud(no_go_source), factor, ny, nx)
    machinery_ds = downsample_any(np.flipud(machinery_source), factor, ny, nx)
    # Restore the original safety precedence: a no-go polygon remains a no-go
    # wall even where it overlaps a SLAM-derived machinery outline.
    machinery_ds &= ~no_go_ds
    occupied_ds &= ~(no_go_ds | machinery_ds)
    unknown_ds &= ~(occupied_ds | no_go_ds | machinery_ds)
    exit_openings, exit_markers = create_exit_markers(
        occupied_ds, semantic, fds_resolution, config["exit_markers"],
    )
    blocked = occupied_ds | unknown_ds | no_go_ds | machinery_ds | exit_markers
    occupied_rectangles = merge_rectangles(occupied_ds)
    unknown_rectangles = merge_rectangles(unknown_ds)
    no_go_rectangles = merge_rectangles(no_go_ds)
    machinery_rectangles = merge_rectangles(machinery_ds)

    generated_dir = FACTORY_DIR / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    obstacle_path = generated_dir / "obstacles.inc"
    obstacle_text = fds_obstacle_text(
        occupied_rectangles, unknown_rectangles, no_go_rectangles,
        machinery_rectangles, fds_resolution, float(config["wall_height_m"]),
        float(config["machinery_height_m"]),
    )
    obstacle_path.write_text(obstacle_text, encoding="utf-8")
    exit_text = fds_exit_text(exit_openings, fds_resolution)
    (generated_dir / "exits.inc").write_text(exit_text, encoding="utf-8")
    scenario_text, scenario_report, oil_mask, cardboard_mask = fds_scenario_text(
        scenario, blocked, no_go_ds, occupied_ds | unknown_ds,
        fds_resolution, semantic, exit_openings
    )
    (generated_dir / "scenario.inc").write_text(scenario_text, encoding="utf-8")
    np.savez_compressed(
        generated_dir / "occupancy_grid.npz",
        blocked=blocked,
        occupied=occupied_ds,
        unknown=unknown_ds,
        no_go=no_go_ds,
        machinery=machinery_ds,
        oil_spill=oil_mask,
        cardboard_boxes=cardboard_mask,
        exit_corridors=np.zeros_like(blocked),
        exit_markers=exit_markers,
        free=~blocked,
        resolution_m=np.array(fds_resolution),
    )

    report = {
        "source_map": str(source_map_path.relative_to(FACTORY_DIR.parent.parent)),
        "source_sha256": hashlib.sha256(source_map_path.read_bytes()).hexdigest(),
        "source_no_go_zones": str(no_go_path.relative_to(FACTORY_DIR.parent.parent)),
        "source_no_go_sha256": hashlib.sha256(no_go_path.read_bytes()).hexdigest(),
        "no_go_zones": {
            "count": len(no_go_zones),
            "names": [zone["name"] for zone in no_go_zones],
            "source_cells": int(no_go_source.sum()),
            "coarse_cells": int(no_go_ds.sum()),
        },
        "machinery": {
            "height_m": float(config["machinery_height_m"]),
            "source_components": machinery_records,
            "coarse_cells": int(machinery_ds.sum()),
            "rectangles": len(machinery_rectangles),
        },
        "source_shape_cells": [height, width],
        "source_extent_m": [transform.width_m, transform.height_m],
        "source_counts": {
            "occupied": int(occupied.sum()),
            "free": int(free.sum()),
            "unknown": int(unknown.sum()),
        },
        "fds_resolution_m": fds_resolution,
        "fds_grid_shape_yx": [ny, nx],
        "fds_extent_m": [nx * fds_resolution, ny * fds_resolution],
        "fds_height_m": float(config["fds_height_m"]),
        "mesh_ijk": [
            nx, ny,
            round(float(config["fds_height_m"]) / fds_resolution),
        ],
        "mesh_cell_count": int(
            nx * ny * round(float(config["fds_height_m"]) / fds_resolution)
        ),
        "coarse_counts": {
            "occupied": int(occupied_ds.sum()),
            "unknown": int(unknown_ds.sum()),
            "no_go": int(no_go_ds.sum()),
            "machinery": int(machinery_ds.sum()),
            "blocked": int(blocked.sum()),
            "free": int((~blocked).sum()),
        },
        "wall_simplification": {
            "enabled": bool(simplify_config["enabled"]),
            "contour_tolerance_m": float(simplify_config["contour_tolerance_m"]),
            "minimum_free_component_area_m2": float(
                simplify_config["minimum_free_component_area_m2"]
            ),
            "source_cells_changed": int(np.count_nonzero(simplified_free != free)),
            "free_cells_changed_to_wall": int(np.count_nonzero(simplified_blocked & free)),
            "inaccessible_cells_changed_to_free": int(
                np.count_nonzero(source_blocked & ~simplified_blocked)
            ),
            "original_inaccessible_cells": int(source_blocked.sum()),
            "allow_derived_spike_cleanup": bool(
                simplify_config.get("allow_derived_spike_cleanup", False)
            ),
        },
        "exit_markers": exit_openings,
        "scenario": scenario_report,
        "obstacle_rectangles": {
            "occupied": len(occupied_rectangles),
            "unknown": len(unknown_rectangles),
            "no_go": len(no_go_rectangles),
            "machinery": len(machinery_rectangles),
            "total": (
                len(occupied_rectangles) + len(unknown_rectangles)
                + len(no_go_rectangles) + len(machinery_rectangles)
            ),
        },
        "obstacles_sha256": hashlib.sha256(obstacle_text.encode()).hexdigest(),
        "exits_sha256": hashlib.sha256(exit_text.encode()).hexdigest(),
        "scenario_sha256": hashlib.sha256(scenario_text.encode()).hexdigest(),
    }
    (generated_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
