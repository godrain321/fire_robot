#!/usr/bin/env python3
"""Create factory_v3 by rigidly rotating the current factory_v2 geometry."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
V3_DIR = SCRIPT_DIR.parent
REPO_DIR = V3_DIR.parents[1]
V2_DIR = REPO_DIR / "simulator" / "factory_v2"
RESOLUTION = 0.2
SOURCE_WIDTH = 27.2
SOURCE_HEIGHT = 22.8
FDS_HEIGHT = 3.0
DOMAIN_X_MIN = 1.8
DOMAIN_X_MAX = 30.0
DOMAIN_Y_MIN = 5.6
DOMAIN_Y_MAX = 28.0

OBST_RE = re.compile(
    r"&OBST\s+ID='([^']+)'.*?XB=([0-9.,+\-Ee]+).*?SURF_ID='([^']+)'"
)
VENT_RE = re.compile(
    r"&VENT\s+ID='([^']+)'.*?XB=([0-9.,+\-Ee]+).*?SURF_ID='([^']+)'"
)


def sha256(path: Path) -> str:
    """Return a stable checksum for source-protection verification."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_records(path: Path, pattern: re.Pattern[str]) -> list[dict]:
    """Parse one-line FDS OBST or VENT records."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        xb = [
            float(value) for value in match.group(2).split(",") if value.strip()
        ]
        if len(xb) != 6:
            raise ValueError(f"invalid XB record: {line}")
        records.append(
            {"id": match.group(1), "xb": xb, "surface": match.group(3), "line": line}
        )
    return records


def source_mask(records: list[dict]) -> np.ndarray:
    """Rasterize axis-aligned source records at the FDS horizontal resolution."""
    shape = (round(SOURCE_HEIGHT / RESOLUTION), round(SOURCE_WIDTH / RESOLUTION))
    mask = np.zeros(shape, dtype=bool)
    for record in records:
        x0, x1, y0, y1 = record["xb"][:4]
        gx0, gx1 = round(x0 / RESOLUTION), round(x1 / RESOLUTION)
        gy0, gy1 = round(y0 / RESOLUTION), round(y1 / RESOLUTION)
        mask[gy0:gy1, gx0:gx1] = True
    return mask


def merge_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Merge true cells into deterministic, non-overlapping rectangles."""
    rectangles: list[tuple[int, int, int, int]] = []
    active: dict[tuple[int, int], tuple[int, int]] = {}
    for gy, row in enumerate(mask):
        runs = []
        gx = 0
        while gx < len(row):
            if not row[gx]:
                gx += 1
                continue
            start = gx
            while gx < len(row) and row[gx]:
                gx += 1
            runs.append((start, gx))
        current = set(runs)
        for run in sorted(set(active) - current):
            start_y, end_y = active.pop(run)
            rectangles.append((run[0], run[1], start_y, end_y))
        for run in runs:
            if run in active:
                active[run] = (active[run][0], gy + 1)
            else:
                active[run] = (gy, gy + 1)
    for run, (start_y, end_y) in sorted(active.items()):
        rectangles.append((run[0], run[1], start_y, end_y))
    return sorted(rectangles, key=lambda rect: (rect[2], rect[0], rect[3], rect[1]))


def clip_to_domain(mask: np.ndarray) -> np.ndarray:
    """Discard geometry outside the configured rectangular v3 domain."""
    clipped = np.zeros_like(mask)
    gx0, gx1 = round(DOMAIN_X_MIN / RESOLUTION), round(DOMAIN_X_MAX / RESOLUTION)
    gy0, gy1 = round(DOMAIN_Y_MIN / RESOLUTION), round(DOMAIN_Y_MAX / RESOLUTION)
    clipped[gy0:gy1, gx0:gx1] = mask[gy0:gy1, gx0:gx1]
    return clipped


def domain_mask(shape: tuple[int, int]) -> np.ndarray:
    """Return all cells in the configured rectangular v3 domain."""
    domain = np.zeros(shape, dtype=bool)
    gx0, gx1 = round(DOMAIN_X_MIN / RESOLUTION), round(DOMAIN_X_MAX / RESOLUTION)
    gy0, gy1 = round(DOMAIN_Y_MIN / RESOLUTION), round(DOMAIN_Y_MAX / RESOLUTION)
    domain[gy0:gy1, gx0:gx1] = True
    return domain


def add_slam_wall_bridges(mask: np.ndarray) -> np.ndarray:
    """Apply explicitly documented one-cell corrections to close the enclosure."""
    corrected = mask.copy()
    bridges = ((12.4, 12.6, 6.4, 6.6),)
    for x0, x1, y0, y1 in bridges:
        gx0, gx1 = round(x0 / RESOLUTION), round(x1 / RESOLUTION)
        gy0, gy1 = round(y0 / RESOLUTION), round(y1 / RESOLUTION)
        corrected[gy0:gy1, gx0:gx1] = True
    return corrected


def exterior_fill(slam_wall: np.ndarray, exits: np.ndarray) -> np.ndarray:
    """Flood from the rectangular boundary up to the closed SLAM/EXIT enclosure."""
    domain = domain_mask(slam_wall.shape)
    passable = domain & ~(slam_wall | exits)
    seeds = np.zeros_like(passable)
    gx0, gx1 = round(DOMAIN_X_MIN / RESOLUTION), round(DOMAIN_X_MAX / RESOLUTION)
    gy0, gy1 = round(DOMAIN_Y_MIN / RESOLUTION), round(DOMAIN_Y_MAX / RESOLUTION)
    seeds[gy0, gx0:gx1] = passable[gy0, gx0:gx1]
    seeds[gy1 - 1, gx0:gx1] = passable[gy1 - 1, gx0:gx1]
    seeds[gy0:gy1, gx0] = passable[gy0:gy1, gx0]
    seeds[gy0:gy1, gx1 - 1] = passable[gy0:gy1, gx1 - 1]
    count, labels = cv2.connectedComponents(passable.astype(np.uint8), connectivity=4)
    exterior = np.zeros_like(passable)
    seed_labels = np.unique(labels[seeds])
    for label in seed_labels:
        if label != 0 and label < count:
            exterior |= labels == label
    return exterior


def reference_angle(exit_records: list[dict]) -> tuple[float, tuple[float, float], float]:
    """Fit the long axis of EXIT1 marker centers and return its alignment rotation."""
    selected = [record for record in exit_records if record["id"].startswith("EXIT1_")]
    if len(selected) < 2:
        raise ValueError("at least two EXIT1 records are required")
    centers = np.array(
        [[(r["xb"][0] + r["xb"][1]) / 2, (r["xb"][2] + r["xb"][3]) / 2]
         for r in selected]
    )
    pivot = centers.mean(axis=0)
    covariance = np.cov((centers - pivot).T)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, np.argmax(values)]
    if direction[0] < 0:
        direction = -direction
    source_angle = math.atan2(direction[1], direction[0])
    slope = direction[1] / direction[0]
    return -source_angle, (float(pivot[0]), float(pivot[1])), float(slope)


class Rotation:
    """Rigid XY rotation followed by a translation into the positive quadrant."""

    def __init__(self, theta: float, pivot: tuple[float, float]):
        self.theta = theta
        self.pivot = np.array(pivot, dtype=float)
        self.matrix = np.array(
            [[math.cos(theta), -math.sin(theta)],
             [math.sin(theta), math.cos(theta)]]
        )
        corners = np.array(
            [[0.0, 0.0], [SOURCE_WIDTH, 0.0], [0.0, SOURCE_HEIGHT],
             [SOURCE_WIDTH, SOURCE_HEIGHT]]
        )
        raw = (corners - self.pivot) @ self.matrix.T
        self.raw_min = raw.min(axis=0)
        raw_size = raw.max(axis=0) - self.raw_min
        self.width_cells = math.ceil(raw_size[0] / RESOLUTION)
        self.height_cells = math.ceil(raw_size[1] / RESOLUTION)
        self.width = self.width_cells * RESOLUTION
        self.height = self.height_cells * RESOLUTION

    def point(self, x: float, y: float) -> tuple[float, float]:
        rotated = self.matrix @ (np.array([x, y]) - self.pivot) - self.raw_min
        return float(rotated[0]), float(rotated[1])

    def mask(self, mask: np.ndarray) -> np.ndarray:
        """Inverse-sample a source mask at every rotated output cell center."""
        ys, xs = np.indices((self.height_cells, self.width_cells), dtype=float)
        output_points = np.stack(
            ((xs + 0.5) * RESOLUTION, (ys + 0.5) * RESOLUTION), axis=-1
        )
        source_points = (
            (output_points + self.raw_min) @ self.matrix + self.pivot
        )
        source_x = np.floor(source_points[..., 0] / RESOLUTION).astype(int)
        source_y = np.floor(source_points[..., 1] / RESOLUTION).astype(int)
        valid = (
            (source_x >= 0) & (source_x < mask.shape[1])
            & (source_y >= 0) & (source_y < mask.shape[0])
        )
        result = np.zeros(valid.shape, dtype=bool)
        result[valid] = mask[source_y[valid], source_x[valid]]
        return result


def group_key(record: dict, kind: str) -> tuple[str, float, float, str]:
    """Keep surfaces, heights, and individual EXIT markers distinct."""
    marker = ""
    if kind == "exit":
        marker = record["id"].split("_", 1)[0]
    return record["surface"], record["xb"][4], record["xb"][5], marker


def write_obstacles(
    path: Path, records: list[dict], rotation: Rotation, kind: str,
    enclosure_records: list[dict] | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Rotate, merge, and write OBST records."""
    groups: dict[tuple[str, float, float, str], list[dict]] = defaultdict(list)
    unmapped_records = [
        record for record in records if record["surface"] == "UNMAPPED_SOLID"
    ]
    for record in records:
        if kind == "obstacle" and record["surface"] == "UNMAPPED_SOLID":
            continue
        groups[group_key(record, kind)].append(record)
    lines = [f"! AUTO-GENERATED rotated {kind} geometry; do not edit.", ""]
    combined = np.zeros((rotation.height_cells, rotation.width_cells), dtype=bool)
    slam_wall_mask = np.zeros_like(combined)
    count = 0
    for surface, z0, z1, marker in sorted(groups):
        rotated = clip_to_domain(
            rotation.mask(source_mask(groups[(surface, z0, z1, marker)]))
        )
        if kind == "obstacle" and surface == "SLAM_WALL" and z0 == 0.0 and z1 == 3.0:
            rotated = add_slam_wall_bridges(rotated)
            slam_wall_mask |= rotated
        combined |= rotated
        prefix = marker or re.sub(r"[^A-Z0-9]+", "_", surface.upper())
        for rectangle in merge_rectangles(rotated):
            count += 1
            gx0, gx1, gy0, gy1 = rectangle
            lines.append(
                f"&OBST ID='V3_{prefix}_{count:04d}', "
                f"XB={gx0 * RESOLUTION:.3f},{gx1 * RESOLUTION:.3f},"
                f"{gy0 * RESOLUTION:.3f},{gy1 * RESOLUTION:.3f},"
                f"{z0:.3f},{z1:.3f}, SURF_ID='{surface}' /"
            )
    others_mask = np.zeros_like(combined)
    if kind == "obstacle":
        if enclosure_records is None:
            raise ValueError("exit enclosure records are required for OTHERS fill")
        exit_mask = clip_to_domain(rotation.mask(source_mask(enclosure_records)))
        source_domain = np.ones(
            (round(SOURCE_HEIGHT / RESOLUTION), round(SOURCE_WIDTH / RESOLUTION)),
            dtype=bool,
        )
        rotated_source_domain = rotation.mask(source_domain)
        others_mask = clip_to_domain(rotation.mask(source_mask(unmapped_records)))
        others_mask |= domain_mask(combined.shape) & ~rotated_source_domain
        others_mask &= ~combined & ~exit_mask
        for index, rectangle in enumerate(merge_rectangles(others_mask), start=1):
            gx0, gx1, gy0, gy1 = rectangle
            lines.append(
                f"&OBST ID='V3_OTHERS_{index:04d}', "
                f"XB={gx0 * RESOLUTION:.3f},{gx1 * RESOLUTION:.3f},"
                f"{gy0 * RESOLUTION:.3f},{gy1 * RESOLUTION:.3f},"
                f"0.000,{FDS_HEIGHT:.3f}, SURF_ID='OTHERS' /"
            )
        combined |= others_mask
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return count, combined, others_mask


def write_scenario(path: Path, rotation: Rotation) -> tuple[int, int]:
    """Rotate current, including manually edited, scenario geometry."""
    source_path = V2_DIR / "generated" / "scenario.inc"
    source_text = source_path.read_text(encoding="utf-8")
    obstacle_records = parse_records(source_path, OBST_RE)
    vent_records = parse_records(source_path, VENT_RE)
    definitions = [
        line for line in source_text.splitlines()
        if not line.lstrip().startswith(("&OBST", "&VENT"))
    ]
    lines = ["! AUTO-GENERATED rotated scenario; edit factory_v2 source scenario first."]
    lines.extend(line for line in definitions if line.strip() and not line.startswith("!"))

    scenario_count = 0
    groups: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for record in obstacle_records:
        groups[(record["surface"], record["xb"][4], record["xb"][5])].append(record)
    for (surface, z0, z1), group in sorted(groups.items()):
        for rectangle in merge_rectangles(
            clip_to_domain(rotation.mask(source_mask(group)))
        ):
            scenario_count += 1
            gx0, gx1, gy0, gy1 = rectangle
            lines.append(
                f"&OBST ID='V3_CARDBOARD_{scenario_count:03d}', "
                f"XB={gx0 * RESOLUTION:.3f},{gx1 * RESOLUTION:.3f},"
                f"{gy0 * RESOLUTION:.3f},{gy1 * RESOLUTION:.3f},"
                f"{z0:.3f},{z1:.3f}, SURF_ID='{surface}' /"
            )

    oil_records = [record for record in vent_records if record["surface"] != "OPEN"]
    oil_count = 0
    if oil_records:
        xyz_match = re.search(
            r"XYZ=([0-9.+\-Ee]+),([0-9.+\-Ee]+),([0-9.+\-Ee]+)",
            oil_records[0]["line"],
        )
        spread_match = re.search(r"SPREAD_RATE=([0-9.+\-Ee]+)", oil_records[0]["line"])
        if xyz_match is None or spread_match is None:
            raise ValueError("oil XYZ or SPREAD_RATE missing")
        ignition_x, ignition_y = rotation.point(
            float(xyz_match.group(1)), float(xyz_match.group(2))
        )
        oil_mask = clip_to_domain(rotation.mask(source_mask(oil_records)))
        for rectangle in merge_rectangles(oil_mask):
            oil_count += 1
            gx0, gx1, gy0, gy1 = rectangle
            lines.append(
                f"&VENT ID='V3_OIL_SPILL_{oil_count:03d}', "
                f"XB={gx0 * RESOLUTION:.3f},{gx1 * RESOLUTION:.3f},"
                f"{gy0 * RESOLUTION:.3f},{gy1 * RESOLUTION:.3f},0.000,0.000, "
                f"SURF_ID='{oil_records[0]['surface']}', "
                f"XYZ={ignition_x:.3f},{ignition_y:.3f},0.000, "
                f"SPREAD_RATE={float(spread_match.group(1)):.4f} /"
            )

    open_records = [record for record in vent_records if record["surface"] == "OPEN"]
    for index, rectangle in enumerate(
        merge_rectangles(clip_to_domain(rotation.mask(source_mask(open_records)))), start=1
    ):
        gx0, gx1, gy0, gy1 = rectangle
        lines.append(
            f"&VENT ID='V3_ROOF_VENT_{index:03d}', "
            f"XB={gx0 * RESOLUTION:.3f},{gx1 * RESOLUTION:.3f},"
            f"{gy0 * RESOLUTION:.3f},{gy1 * RESOLUTION:.3f},"
            f"{FDS_HEIGHT:.3f},{FDS_HEIGHT:.3f}, SURF_ID='OPEN' /"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return scenario_count, oil_count


def write_semantics(rotation: Rotation) -> None:
    """Write semantic points with both original and rotated FDS coordinates."""
    source = yaml.safe_load(
        (V2_DIR / "config" / "semantic_points.yaml").read_text(encoding="utf-8")
    )
    source["coordinate_convention"] = "ros_original_and_fds_v2_and_fds_v3_rotated"
    for section in ("poses", "landmarks"):
        for item in source.get(section, {}).values():
            original = item["fds"]
            x, y = rotation.point(float(original["x"]), float(original["y"]))
            rotated = {"x": x, "y": y}
            if "yaw" in original:
                yaw = float(original["yaw"]) + rotation.theta
                rotated["yaw"] = math.atan2(math.sin(yaw), math.cos(yaw))
            item["fds_v3"] = rotated
    (V3_DIR / "config" / "semantic_points.yaml").write_text(
        yaml.safe_dump(source, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def write_fds(rotation: Rotation) -> None:
    """Write the standalone rotated FDS case."""
    width_cells = round((DOMAIN_X_MAX - DOMAIN_X_MIN) / RESOLUTION)
    height_cells = round((DOMAIN_Y_MAX - DOMAIN_Y_MIN) / RESOLUTION)
    text = f"""&HEAD CHID='factory_v3', TITLE='Rotated factory_v2; EXIT1 aligned to X axis' /

&MESH IJK={width_cells},{height_cells},15,
      XB={DOMAIN_X_MIN:.1f},{DOMAIN_X_MAX:.1f},{DOMAIN_Y_MIN:.1f},{DOMAIN_Y_MAX:.1f},0.0,3.0 /
&TIME T_END=700 /
&REAC FUEL='PROPANE', SOOT_YIELD=0.01, CO_YIELD=0.02,
      HEAT_OF_COMBUSTION=46460. /

&SURF ID='SLAM_WALL', COLOR='GRAY 30' /
&SURF ID='OTHERS', COLOR='GRAY 70' /
&SURF ID='NO_GO_WALL', COLOR='ORANGE' /
&SURF ID='MACHINERY', COLOR='DARK GRAY' /
&SURF ID='EXIT_SOLID', COLOR='BLUE' /

&CATF OTHER_FILES='generated/obstacles.inc','generated/exits.inc',
      'generated/scenario.inc' /

&DUMP DT_SLCF=1.0, DT_SL3D=1.0 /
&SLCF ID='TEMP_3D', XB={DOMAIN_X_MIN:.1f},{DOMAIN_X_MAX:.1f},{DOMAIN_Y_MIN:.1f},{DOMAIN_Y_MAX:.1f},0.0,3.0,
      QUANTITY='TEMPERATURE', CELL_CENTERED=T /
&SLCF ID='CO_Z130', PBZ=1.3, QUANTITY='VOLUME FRACTION',
      SPEC_ID='CARBON MONOXIDE', CELL_CENTERED=T /
&TAIL /
"""
    (V3_DIR / "factory_v3.fds").write_text(text, encoding="utf-8")


def write_overlay(source_mask_all: np.ndarray, rotated_mask: np.ndarray, theta: float) -> None:
    """Save a side-by-side geometry validation image."""
    scale = 4
    panels = []
    labels = (
        "factory_v2 source occupancy",
        f"factory_v3 rotated ({math.degrees(theta):.3f} deg)",
    )
    for mask, label in zip((source_mask_all, rotated_mask), labels):
        image = np.where(np.flipud(mask), 30, 245).astype(np.uint8)
        image = cv2.resize(
            image, (image.shape[1] * scale, image.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.copyMakeBorder(
            image, 45, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        cv2.putText(
            image, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (20, 20, 20), 2, cv2.LINE_AA,
        )
        panels.append(image)
    target_height = max(panel.shape[0] for panel in panels)
    padded = [
        cv2.copyMakeBorder(
            panel, 0, target_height - panel.shape[0], 0, 0,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        for panel in panels
    ]
    cv2.imwrite(
        str(V3_DIR / "validation" / "rotation_overlay.png"),
        cv2.hconcat(padded),
    )


def main() -> int:
    """Generate all v3 artifacts without writing anywhere under factory_v2."""
    source_paths = {
        "factory_v2_fds": V2_DIR / "factory_v2.fds",
        "obstacles": V2_DIR / "generated" / "obstacles.inc",
        "exits": V2_DIR / "generated" / "exits.inc",
        "scenario": V2_DIR / "generated" / "scenario.inc",
    }
    before = {name: sha256(path) for name, path in source_paths.items()}
    obstacle_records = parse_records(source_paths["obstacles"], OBST_RE)
    exit_records = parse_records(source_paths["exits"], OBST_RE)
    theta, pivot, fitted_slope = reference_angle(exit_records)
    rotation = Rotation(theta, pivot)

    obstacle_count, source_rotated, others_mask = write_obstacles(
        V3_DIR / "generated" / "obstacles.inc", obstacle_records, rotation, "obstacle",
        enclosure_records=exit_records,
    )
    exit_count, exit_rotated, _ = write_obstacles(
        V3_DIR / "generated" / "exits.inc", exit_records, rotation, "exit"
    )
    scenario_count, oil_count = write_scenario(
        V3_DIR / "generated" / "scenario.inc", rotation
    )
    write_semantics(rotation)
    write_fds(rotation)
    write_overlay(
        source_mask(obstacle_records + exit_records),
        clip_to_domain(source_rotated | exit_rotated),
        theta,
    )
    after = {name: sha256(path) for name, path in source_paths.items()}
    if after != before:
        raise RuntimeError("factory_v2 changed during factory_v3 generation")

    report = {
        "status": "PASS",
        "factory_v2_unchanged": True,
        "factory_v2_source_hashes": after,
        "exit1_fitted_source_slope": fitted_slope,
        "source_axis_angle_deg": -math.degrees(theta),
        "rotation_angle_deg": math.degrees(theta),
        "rotation_pivot_fds_v2": list(pivot),
        "translation_raw_min": rotation.raw_min.tolist(),
        "unmapped_solid_policy": "reclassified_as_OTHERS",
        "outer_boundary_surface": "OTHERS",
        "others_cells": int(others_mask.sum()),
        "slam_wall_bridge_xb": [12.4, 12.6, 6.4, 6.6, 0.0, FDS_HEIGHT],
        "mesh": {
            "ijk": [
                round((DOMAIN_X_MAX - DOMAIN_X_MIN) / RESOLUTION),
                round((DOMAIN_Y_MAX - DOMAIN_Y_MIN) / RESOLUTION),
                15,
            ],
            "xb": [DOMAIN_X_MIN, DOMAIN_X_MAX, DOMAIN_Y_MIN, DOMAIN_Y_MAX, 0.0, FDS_HEIGHT],
            "cell_count": (
                round((DOMAIN_X_MAX - DOMAIN_X_MIN) / RESOLUTION)
                * round((DOMAIN_Y_MAX - DOMAIN_Y_MIN) / RESOLUTION) * 15
            ),
        },
        "generated_obstacles": obstacle_count,
        "generated_exit_obstacles": exit_count,
        "generated_scenario_obstacles": scenario_count,
        "generated_oil_vents": oil_count,
    }
    (V3_DIR / "validation" / "rotation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
