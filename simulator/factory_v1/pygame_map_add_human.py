# pygame_map_matplotlib_thermal_viewer.py

import argparse
import json
import math
import os
import re
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pygame
import matplotlib.pyplot as plt
import fdsreader as fds

from sensors.mq135_sensor import MQ135Sensor
from sensors.thermal_camera import ThermalCameraMLX90640

try:
    # human_detection_sim.py를 현재 viewer 파일과 같은 폴더에 둔 경우
    from human_detection_sim import SimpleHumanDetector
except ImportError:
    # sensors 폴더 안에 둔 경우를 위한 fallback
    from sensors.human_detection_sim import SimpleHumanDetector


# ============================================================
# SCREEN SETTINGS
# ============================================================

SCREEN_W = 950
SCREEN_H = 900
FPS = 30

MAP_LEFT = 20
MAP_TOP = 20
MAP_W = 700
MAP_H = 850

PANEL_X = 745
PANEL_Y = 30


# ============================================================
# FDS NAMELIST PARSER
# ============================================================

def read_fds_namelists(fds_path):
    with open(fds_path, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned_lines = []
    for line in text.splitlines():
        line = line.split("!")[0]
        if line.strip():
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    blocks = re.findall(r"&([A-Z0-9_]+)\s+(.*?)/", text, flags=re.S | re.I)

    namelists = []
    for name, body in blocks:
        body = " ".join(body.replace("\n", " ").split())
        namelists.append((name.upper(), body))

    return namelists


def parse_xb(body):
    match = re.search(r"\bXB\s*=\s*([^/]+)", body, flags=re.I)
    if not match:
        return None

    raw = match.group(1)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", raw)

    if len(nums) < 6:
        return None

    return list(map(float, nums[:6]))


def parse_id(body, default_name):
    match = re.search(r"\bID\s*=\s*'([^']+)'", body, flags=re.I)
    if match:
        return match.group(1)
    return default_name


def parse_surf_id(body):
    match = re.search(r"\bSURF_ID\s*=\s*'([^']+)'", body, flags=re.I)
    if match:
        return match.group(1)
    return ""


def load_factory_from_fds(fds_path):
    namelists = read_fds_namelists(fds_path)

    mesh_xb = None
    obstacles = []
    holes = []
    burner_vents = []

    obst_count = 0
    hole_count = 0
    vent_count = 0

    for name, body in namelists:
        if name == "MESH":
            xb = parse_xb(body)
            if xb:
                mesh_xb = xb

        elif name == "OBST":
            xb = parse_xb(body)
            if xb:
                obst_count += 1
                obstacles.append({
                    "id": parse_id(body, f"OBST_{obst_count}"),
                    "xb": xb,
                    "surf_id": parse_surf_id(body),
                })

        elif name == "HOLE":
            xb = parse_xb(body)
            if xb:
                hole_count += 1
                holes.append({
                    "id": parse_id(body, f"HOLE_{hole_count}"),
                    "xb": xb,
                })

        elif name == "VENT":
            xb = parse_xb(body)
            surf_id = parse_surf_id(body)

            if xb and surf_id.upper() == "BURNER":
                vent_count += 1
                burner_vents.append({
                    "id": parse_id(body, f"BURNER_{vent_count}"),
                    "xb": xb,
                })

    if mesh_xb is None:
        raise RuntimeError("FDS 파일에서 &MESH XB=... 정보를 찾지 못했습니다.")

    return mesh_xb, obstacles, holes, burner_vents


# ============================================================
# TEMPERATURE NPZ LOAD
# ============================================================

def make_axis_coords(n, min_v, max_v, resolution):
    length = max_v - min_v
    n_center = int(round(length / resolution))
    n_boundary = n_center + 1

    if n == n_center:
        return min_v + resolution / 2.0 + np.arange(n) * resolution

    if n == n_boundary:
        return np.linspace(min_v, max_v, n)

    return np.linspace(min_v, max_v, n)


def load_temperature_npz(npz_path, x_max=20.0, y_max=30.0, z_max=5.0):
    npz_path = Path(npz_path)
    data = np.load(npz_path)

    temp = np.asarray(data["temperature"], dtype=float)
    times = np.asarray(data["times"], dtype=float)
    xy_res = float(data["xy_resolution"])
    z_res = float(data["z_resolution"])

    print("[INFO] Loaded temperature npz")
    print(f"  path: {npz_path}")
    print(f"  raw temperature shape: {temp.shape}")
    print(f"  times shape: {times.shape}")
    print(f"  xy_resolution: {xy_res}")
    print(f"  z_resolution: {z_res}")

    if temp.ndim != 4:
        raise RuntimeError(f"temperature는 4차원이어야 합니다. 현재 shape={temp.shape}")

    if temp.shape[0] != len(times):
        raise RuntimeError(
            "temperature의 첫 번째 축이 time이 아닌 것 같습니다. "
            f"temperature shape={temp.shape}, len(times)={len(times)}"
        )

    nx_expected = int(round(x_max / xy_res))
    ny_expected = int(round(y_max / xy_res))
    nz_expected = int(round(z_max / z_res))

    spatial_shape = temp.shape[1:]

    best_perm = None
    best_score = 1e9

    for perm in permutations(range(3)):
        sx = spatial_shape[perm[0]]
        sy = spatial_shape[perm[1]]
        sz = spatial_shape[perm[2]]

        def score_dim(value, expected):
            return min(abs(value - expected), abs(value - (expected + 1)))

        score = (
            score_dim(sx, nx_expected)
            + score_dim(sy, ny_expected)
            + score_dim(sz, nz_expected)
        )

        if score < best_score:
            best_score = score
            best_perm = perm

    # 최종 shape: (time, x, y, z)
    temp = np.transpose(temp, (0, 1 + best_perm[0], 1 + best_perm[1], 1 + best_perm[2]))

    nx, ny, nz = temp.shape[1:]

    x_coords = make_axis_coords(nx, 0.0, x_max, xy_res)
    y_coords = make_axis_coords(ny, 0.0, y_max, xy_res)
    z_coords = make_axis_coords(nz, 0.0, z_max, z_res)

    print("[INFO] Normalized temperature data")
    print(f"  normalized shape: {temp.shape}  # (time, x, y, z)")
    print(f"  x coords: {len(x_coords)}, {x_coords.min():.2f} ~ {x_coords.max():.2f}")
    print(f"  y coords: {len(y_coords)}, {y_coords.min():.2f} ~ {y_coords.max():.2f}")
    print(f"  z coords: {len(z_coords)}, {z_coords.min():.2f} ~ {z_coords.max():.2f}")

    return temp, times, x_coords, y_coords, z_coords, xy_res, z_res


# ============================================================
# CO SLICE LOAD
# ============================================================

def find_co_slice(sim):
    print("\n[INFO] FDS slices found:")
    for i, slc in enumerate(sim.slices):
        print(f"  [{i}] {slc}")

    for slc in sim.slices:
        text = str(slc).upper()
        if "CO_Z130" in text:
            print("\n[INFO] Selected CO slice by ID: CO_Z130")
            return slc

    for slc in sim.slices:
        text = str(slc).upper()
        if "CARBON MONOXIDE" in text:
            print("\n[INFO] Selected CO slice by quantity: CARBON MONOXIDE")
            return slc

    raise RuntimeError("CO_Z130 또는 CARBON MONOXIDE slice를 찾지 못했습니다.")


def load_co_slice(fds_dir):
    sim = fds.Simulation(str(fds_dir))
    co_slice = find_co_slice(sim)

    result = co_slice.to_global(
        masked=True,
        fill=np.nan,
        return_coordinates=True
    )

    co_data, coords = result
    co_data = np.asarray(co_data, dtype=float)

    x_coords = np.asarray(coords["x"], dtype=float)
    y_coords = np.asarray(coords["y"], dtype=float)
    times = np.asarray(co_slice.times, dtype=float)

    print("\n[INFO] Loaded CO slice")
    print(f"  shape: {co_data.shape}")
    print(f"  time steps: {len(times)}")
    print(f"  x range: {x_coords.min():.2f} ~ {x_coords.max():.2f}")
    print(f"  y range: {y_coords.min():.2f} ~ {y_coords.max():.2f}")

    return co_data, times, x_coords, y_coords


# ============================================================
# SAMPLING
# ============================================================

def nearest_time_index(times, t):
    return int(np.nanargmin(np.abs(times - t)))


def sample_2d_xy(data_2d, x_coords, y_coords, x, y):
    ix = int(np.nanargmin(np.abs(x_coords - x)))
    iy = int(np.nanargmin(np.abs(y_coords - y)))

    if data_2d.shape == (len(x_coords), len(y_coords)):
        return float(data_2d[ix, iy])

    if data_2d.shape == (len(y_coords), len(x_coords)):
        return float(data_2d[iy, ix])

    raise RuntimeError(
        f"2D shape mismatch: data={data_2d.shape}, "
        f"x={len(x_coords)}, y={len(y_coords)}"
    )


# ============================================================
# OBSTACLE VOLUME FOR THERMAL CAMERA
# ============================================================

def build_obstacle_volume_zyx(obstacles, holes, x_coords, y_coords, z_coords):
    nx = len(x_coords)
    ny = len(y_coords)
    nz = len(z_coords)

    obstacle_volume = np.zeros((nz, ny, nx), dtype=bool)

    def fill_xb(xb, value):
        x1, x2, y1, y2, z1, z2 = xb

        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        z_min, z_max = min(z1, z2), max(z1, z2)

        xi = np.where((x_coords >= x_min) & (x_coords <= x_max))[0]
        yi = np.where((y_coords >= y_min) & (y_coords <= y_max))[0]
        zi = np.where((z_coords >= z_min) & (z_coords <= z_max))[0]

        if len(xi) == 0 or len(yi) == 0 or len(zi) == 0:
            return

        obstacle_volume[np.ix_(zi, yi, xi)] = value

    for obs in obstacles:
        fill_xb(obs["xb"], True)

    for hole in holes:
        fill_xb(hole["xb"], False)

    print("\n[INFO] Built obstacle_volume for thermal camera")
    print(f"  shape: {obstacle_volume.shape}  # (z, y, x)")
    print(f"  obstacle cells: {int(obstacle_volume.sum())}")

    return obstacle_volume


# ============================================================
# ROBOT PATH / MOTION
# ============================================================

def load_robot_path(path_file):
    default_data = {
        "motion_config": {
            "linear_speed": 0.45,
            "angular_speed_deg": 45.0
        },
        "start_pose": {
            "x": 1.0,
            "y": 1.0,
            "theta_deg": 90.0
        },
        "path": [
            [1.0, 1.0],
            [1.0, 21.0],
            [10.0, 21.0],
            [10.0, 25.0]
        ]
    }

    if not os.path.exists(path_file):
        print(f"[INFO] {path_file}이 없어서 기본 경로를 사용합니다.")
        return default_data

    with open(path_file, "r", encoding="utf-8") as f:
        return json.load(f)


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def build_motion_segments(path, start_theta, linear_speed, angular_speed):
    segments = []

    if len(path) < 2:
        return segments

    current_theta = start_theta

    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]

        dx = x1 - x0
        dy = y1 - y0
        distance = math.hypot(dx, dy)

        if distance < 1e-6:
            continue

        target_theta = math.atan2(dy, dx)
        dtheta = wrap_angle(target_theta - current_theta)

        if abs(dtheta) > math.radians(1.0):
            turn_duration = abs(dtheta) / angular_speed

            segments.append({
                "type": "turn",
                "duration": turn_duration,
                "x": x0,
                "y": y0,
                "theta0": current_theta,
                "theta1": target_theta,
                "dtheta": dtheta,
            })

        move_duration = distance / linear_speed

        segments.append({
            "type": "move",
            "duration": move_duration,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "theta": target_theta,
        })

        current_theta = target_theta

    return segments


def pose_from_motion_segments(segments, elapsed_time):
    if not segments:
        return 1.0, 1.0, 0.0

    t = elapsed_time

    for seg in segments:
        duration = seg["duration"]

        if t <= duration:
            ratio = 0.0 if duration < 1e-6 else t / duration

            if seg["type"] == "turn":
                x = seg["x"]
                y = seg["y"]
                theta = seg["theta0"] + seg["dtheta"] * ratio
                theta = wrap_angle(theta)
                return x, y, theta

            if seg["type"] == "move":
                x = seg["x0"] + (seg["x1"] - seg["x0"]) * ratio
                y = seg["y0"] + (seg["y1"] - seg["y0"]) * ratio
                theta = seg["theta"]
                return x, y, theta

        t -= duration

    last = segments[-1]

    if last["type"] == "move":
        return last["x1"], last["y1"], last["theta"]

    return last["x"], last["y"], last["theta1"]



# ============================================================
# HUMAN PATH / SIMPLE DETECTION
# ============================================================

def load_human_paths(human_file):
    """
    사람/요구조자 개체의 이동 경로를 JSON에서 읽는다.

    JSON 예시:
    {
      "default_motion_config": {
        "linear_speed": 0.25,
        "angular_speed_deg": 90.0
      },
      "humans": [
        {
          "id": "victim_1",
          "radius": 0.25,
          "start_pose": {"x": 4.0, "y": 4.0, "theta_deg": 0.0},
          "path": [[4.0, 4.0]]
        },
        {
          "id": "victim_2",
          "motion_config": {"linear_speed": 0.35},
          "start_pose": {"x": 14.0, "y": 25.0, "theta_deg": 180.0},
          "path": [[14.0, 25.0], [10.0, 25.0], [10.0, 22.0]]
        }
      ]
    }
    """

    default_data = {
        "default_motion_config": {
            "linear_speed": 0.25,
            "angular_speed_deg": 90.0
        },
        "humans": [
            {
                "id": "victim_1",
                "radius": 0.25,
                "start_pose": {
                    "x": 5.0,
                    "y": 5.0,
                    "theta_deg": 0.0
                },
                "path": [
                    [5.0, 5.0]
                ]
            },
            {
                "id": "victim_2",
                "radius": 0.25,
                "motion_config": {
                    "linear_speed": 0.20,
                    "angular_speed_deg": 90.0
                },
                "start_pose": {
                    "x": 14.0,
                    "y": 24.0,
                    "theta_deg": 180.0
                },
                "path": [
                    [14.0, 24.0],
                    [10.0, 24.0],
                    [10.0, 21.0]
                ]
            }
        ]
    }

    if not os.path.exists(human_file):
        print(f"[INFO] {human_file}이 없어서 기본 사람 개체를 사용합니다.")
        return default_data

    with open(human_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 실수로 최상위에 바로 list를 둔 경우도 허용
    if isinstance(data, list):
        data = {
            "default_motion_config": default_data["default_motion_config"],
            "humans": data
        }

    if "humans" not in data:
        raise RuntimeError("human JSON에는 'humans' 리스트가 필요합니다.")

    return data


def build_human_entities(human_data):
    """
    JSON 데이터를 시뮬레이션에서 바로 사용할 수 있는 사람 개체 리스트로 변환한다.
    """

    default_motion_cfg = human_data.get("default_motion_config", {})
    default_linear_speed = float(default_motion_cfg.get("linear_speed", 0.25))
    default_angular_speed_deg = float(default_motion_cfg.get("angular_speed_deg", 90.0))

    entities = []

    for i, human in enumerate(human_data.get("humans", [])):
        human_id = str(human.get("id", f"victim_{i + 1}"))
        radius = float(human.get("radius", 0.25))

        start_pose = human.get("start_pose", {})
        path = human.get("path", [])

        if not path:
            start_x = float(start_pose.get("x", 1.0))
            start_y = float(start_pose.get("y", 1.0))
            path = [[start_x, start_y]]

        path = [[float(p[0]), float(p[1])] for p in path]

        # start_pose가 없으면 path의 첫 점을 시작점으로 사용
        start_x = float(start_pose.get("x", path[0][0]))
        start_y = float(start_pose.get("y", path[0][1]))

        if "theta_deg" in start_pose:
            start_theta = math.radians(float(start_pose.get("theta_deg", 0.0)))
        elif len(path) >= 2:
            start_theta = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
        else:
            start_theta = 0.0

        motion_cfg = human.get("motion_config", {})
        linear_speed = float(motion_cfg.get("linear_speed", default_linear_speed))
        angular_speed_deg = float(motion_cfg.get("angular_speed_deg", default_angular_speed_deg))
        angular_speed = math.radians(angular_speed_deg)

        segments = build_motion_segments(
            path=path,
            start_theta=start_theta,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
        )

        entities.append({
            "id": human_id,
            "radius": radius,
            "start_x": start_x,
            "start_y": start_y,
            "start_theta": start_theta,
            "path": path,
            "segments": segments,
            "linear_speed": linear_speed,
        })

    print("\n[INFO] Loaded human entities")
    for human in entities:
        print(
            f"  {human['id']}: start=({human['start_x']:.2f}, {human['start_y']:.2f}), "
            f"path points={len(human['path'])}, speed={human['linear_speed']:.2f} m/s"
        )

    return entities


def pose_from_entity(entity, elapsed_time):
    """
    사람 개체의 현재 위치를 반환한다.
    path가 한 점뿐이면 정지 사람으로 처리한다.
    """

    segments = entity.get("segments", [])

    if not segments:
        return entity["start_x"], entity["start_y"], entity["start_theta"]

    return pose_from_motion_segments(segments, elapsed_time)


def humans_at_time(human_entities, elapsed_time):
    """
    현재 시각의 사람 위치 리스트를 만든다.
    SimpleHumanDetector가 읽을 수 있도록 id, x, y를 포함한다.
    """

    humans = []

    for entity in human_entities:
        x, y, theta = pose_from_entity(entity, elapsed_time)

        humans.append({
            "id": entity["id"],
            "x": x,
            "y": y,
            "theta": theta,
            "radius": entity.get("radius", 0.25),
        })

    return humans


# ============================================================
# WORLD TO SCREEN TRANSFORM
# ============================================================

class WorldTransform:
    def __init__(self, mesh_xb, left, top, width, height):
        self.x_min = mesh_xb[0]
        self.x_max = mesh_xb[1]
        self.y_min = mesh_xb[2]
        self.y_max = mesh_xb[3]

        self.left = left
        self.top = top
        self.width = width
        self.height = height

        map_w = self.x_max - self.x_min
        map_h = self.y_max - self.y_min

        scale_x = width / map_w
        scale_y = height / map_h
        self.scale = min(scale_x, scale_y)

        self.draw_w = map_w * self.scale
        self.draw_h = map_h * self.scale

        self.offset_x = left + (width - self.draw_w) / 2.0
        self.offset_y = top + (height - self.draw_h) / 2.0

    def world_to_screen(self, x, y):
        px = self.offset_x + (x - self.x_min) * self.scale
        py = self.offset_y + (self.y_max - y) * self.scale
        return int(px), int(py)

    def rect_from_xb(self, xb, min_px=2):
        x1, x2, y1, y2, z1, z2 = xb

        px1, py1 = self.world_to_screen(x1, y1)
        px2, py2 = self.world_to_screen(x2, y2)

        left = min(px1, px2)
        top = min(py1, py2)
        width = max(abs(px2 - px1), min_px)
        height = max(abs(py2 - py1), min_px)

        return pygame.Rect(left, top, width, height)


# ============================================================
# DRAWING FUNCTIONS
# ============================================================

def draw_text(screen, font, text, x, y, color=(20, 20, 20)):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))


def draw_robot(screen, transform, x, y, theta):
    robot_length = 0.70
    robot_width = 0.45

    cx, cy = transform.world_to_screen(x, y)

    length_px = robot_length * transform.scale
    width_px = robot_width * transform.scale

    local_corners = [
        (+length_px / 2, +width_px / 2),
        (+length_px / 2, -width_px / 2),
        (-length_px / 2, -width_px / 2),
        (-length_px / 2, +width_px / 2),
    ]

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    points = []
    for lx, ly in local_corners:
        sx = cx + lx * cos_t - ly * sin_t
        sy = cy - (lx * sin_t + ly * cos_t)
        points.append((sx, sy))

    pygame.draw.polygon(screen, (30, 120, 255), points)
    pygame.draw.polygon(screen, (0, 0, 0), points, 2)

    arrow_len = length_px * 0.7
    end_x = cx + arrow_len * cos_t
    end_y = cy - arrow_len * sin_t

    pygame.draw.line(screen, (255, 255, 255), (cx, cy), (end_x, end_y), 3)
    pygame.draw.circle(screen, (255, 255, 255), (int(end_x), int(end_y)), 5)



def draw_detection_range(screen, transform, robot_x, robot_y, detection_range):
    """
    사람 인식 가능 거리 원을 표시한다.
    """

    cx, cy = transform.world_to_screen(robot_x, robot_y)
    radius_px = int(detection_range * transform.scale)

    if radius_px > 0:
        pygame.draw.circle(screen, (60, 170, 80), (cx, cy), radius_px, 2)


def draw_human(screen, transform, font, human, detected=False):
    """
    사람/요구조자 개체를 맵에 표시한다.
    """

    x = float(human["x"])
    y = float(human["y"])
    radius_m = float(human.get("radius", 0.25))
    human_id = str(human.get("id", "victim"))

    cx, cy = transform.world_to_screen(x, y)
    radius_px = max(6, int(radius_m * transform.scale))

    if detected:
        fill_color = (255, 40, 80)
        outline_color = (255, 255, 255)
    else:
        fill_color = (150, 80, 200)
        outline_color = (50, 20, 80)

    pygame.draw.circle(screen, fill_color, (cx, cy), radius_px)
    pygame.draw.circle(screen, outline_color, (cx, cy), radius_px, 2)

    # 사람 표시용 작은 십자
    pygame.draw.line(screen, outline_color, (cx - radius_px, cy), (cx + radius_px, cy), 1)
    pygame.draw.line(screen, outline_color, (cx, cy - radius_px), (cx, cy + radius_px), 1)

    label = font.render(human_id, True, (20, 20, 20))
    screen.blit(label, (cx + radius_px + 3, cy - radius_px - 3))


def draw_human_detection_lines(screen, transform, robot_x, robot_y, humans, detected_ids):
    """
    인식된 사람과 로봇 사이를 선으로 연결한다.
    """

    robot_p = transform.world_to_screen(robot_x, robot_y)

    for human in humans:
        human_id = str(human.get("id", "victim"))

        if human_id not in detected_ids:
            continue

        human_p = transform.world_to_screen(float(human["x"]), float(human["y"]))
        pygame.draw.line(screen, (0, 180, 80), robot_p, human_p, 3)


def draw_map(screen, transform, mesh_xb, obstacles, holes, burner_vents):
    x0, y0 = transform.world_to_screen(mesh_xb[0], mesh_xb[2])
    x1, y1 = transform.world_to_screen(mesh_xb[1], mesh_xb[3])

    map_rect = pygame.Rect(
        min(x0, x1),
        min(y0, y1),
        abs(x1 - x0),
        abs(y1 - y0)
    )

    pygame.draw.rect(screen, (250, 250, 250), map_rect)
    pygame.draw.rect(screen, (0, 0, 0), map_rect, 2)

    for x in range(int(mesh_xb[0]), int(mesh_xb[1]) + 1):
        p1 = transform.world_to_screen(x, mesh_xb[2])
        p2 = transform.world_to_screen(x, mesh_xb[3])
        pygame.draw.line(screen, (220, 220, 220), p1, p2, 1)

    for y in range(int(mesh_xb[2]), int(mesh_xb[3]) + 1):
        p1 = transform.world_to_screen(mesh_xb[0], y)
        p2 = transform.world_to_screen(mesh_xb[1], y)
        pygame.draw.line(screen, (220, 220, 220), p1, p2, 1)

    for obs in obstacles:
        rect = transform.rect_from_xb(obs["xb"], min_px=2)
        pygame.draw.rect(screen, (90, 90, 90), rect)
        pygame.draw.rect(screen, (20, 20, 20), rect, 1)

    for hole in holes:
        rect = transform.rect_from_xb(hole["xb"], min_px=4)
        pygame.draw.rect(screen, (250, 250, 250), rect)
        pygame.draw.rect(screen, (0, 180, 0), rect, 2)

    for vent in burner_vents:
        rect = transform.rect_from_xb(vent["xb"], min_px=6)
        pygame.draw.rect(screen, (255, 80, 60), rect)
        pygame.draw.rect(screen, (120, 0, 0), rect, 2)


def gas_get(gas_reading, key, default=None):
    if gas_reading is None:
        return default

    if isinstance(gas_reading, dict):
        return gas_reading.get(key, default)

    return getattr(gas_reading, key, default)


def gas_status_color(status):
    status = str(status).upper()

    if status == "BLOCKED":
        return (180, 0, 0)

    if status == "DANGER":
        return (220, 60, 0)

    if status == "WARNING":
        return (220, 160, 0)

    if status == "CAUTION":
        return (180, 120, 0)

    return (0, 140, 40)


# ============================================================
# MATPLOTLIB THERMAL WINDOW
# ============================================================

def setup_thermal_window():
    plt.ion()

    fig, ax = plt.subplots(figsize=(6, 4.5))

    initial_img = np.zeros((24, 32), dtype=float) + 25.0

    im = ax.imshow(
        initial_img,
        origin="upper",
        vmin=20.0,
        vmax=300.0,
        aspect="auto",
        cmap="inferno"
    )

    ax.set_title("Thermal Camera")
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature [°C]")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, ax, im


def update_thermal_window(fig, ax, im, thermal_img, sim_time, display_mode="fixed"):
    thermal_min = float(np.nanmin(thermal_img))
    thermal_max = float(np.nanmax(thermal_img))

    im.set_data(thermal_img)

    if display_mode == "auto":
        vmin = thermal_min
        vmax = thermal_max
        if abs(vmax - vmin) < 1.0:
            vmax = vmin + 1.0

    elif display_mode == "fire":
        # 화재 상황 보기용
        vmin = 20.0
        vmax = 300.0

    elif display_mode == "sensor":
        # MLX90640 센서 사양 전체 범위
        vmin = -40.0
        vmax = 300.0

    else:
        # fixed 모드 기존값 유지
        vmin = 20.0
        vmax = 120.0
    im.set_clim(vmin=vmin, vmax=vmax)

    ax.set_title(
        f"Thermal Camera  t={sim_time:.1f}s\n"
        f"image min/max = {thermal_min:.1f} ~ {thermal_max:.1f} °C, "
        f"display = {vmin:.1f} ~ {vmax:.1f} °C"
    )

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fds-dir", type=str, default=".")
    parser.add_argument("--fds-file", type=str, default="factory_v1.fds")
    parser.add_argument("--path-file", type=str, default="robot_path.json")
    parser.add_argument("--human-file", type=str, default="human_path.json")
    parser.add_argument("--human-detection-range", type=float, default=10.0)
    parser.add_argument(
        "--temp-npz",
        type=str,
        default="processed/fds_temperature_3d_timeseries.npz"
    )
    parser.add_argument("--sim-speed", type=float, default=1.0)
    parser.add_argument("--x-max", type=float, default=20.0)
    parser.add_argument("--y-max", type=float, default=30.0)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--sensor-hz", type=float, default=8.0)
    parser.add_argument(
        "--thermal-scale",
        type=str,
        default="fire",
        choices=["fixed", "auto", "fire", "sensor"],
        help="fixed: 20~120 C, auto: image min~max, fire: 20~300 C, sensor: -40~300 C"
    )
    args = parser.parse_args()

    fds_dir = Path(args.fds_dir).resolve()
    fds_file = Path(args.fds_file).resolve()
    path_file = Path(args.path_file).resolve()
    human_file = Path(args.human_file).resolve()
    temp_npz = Path(args.temp_npz).resolve()

    if not fds_file.exists():
        print(f"[ERROR] FDS file not found: {fds_file}")
        sys.exit(1)

    if not temp_npz.exists():
        print(f"[ERROR] temperature npz not found: {temp_npz}")
        sys.exit(1)

    # --------------------------------------------------------
    # Load FDS map
    # --------------------------------------------------------
    mesh_xb, obstacles, holes, burner_vents = load_factory_from_fds(fds_file)

    # --------------------------------------------------------
    # Load FDS data
    # --------------------------------------------------------
    temp_data, temp_times, temp_x, temp_y, temp_z, xy_res, z_res = load_temperature_npz(
        temp_npz,
        x_max=args.x_max,
        y_max=args.y_max,
        z_max=args.z_max,
    )

    co_data, co_times, co_x, co_y = load_co_slice(fds_dir)

    obstacle_volume_zyx = build_obstacle_volume_zyx(
        obstacles=obstacles,
        holes=holes,
        x_coords=temp_x,
        y_coords=temp_y,
        z_coords=temp_z,
    )

    # 사람 인식의 벽/장애물 가림 판단용 2D obstacle map
    # obstacle_volume_zyx shape = (z, y, x)이므로 z축으로 투영하면 (y, x)가 된다.
    obstacle_map_2d = np.any(obstacle_volume_zyx, axis=0)

    # --------------------------------------------------------
    # Load robot path
    # --------------------------------------------------------
    path_data = load_robot_path(path_file)
    path = path_data.get("path", [])

    motion_cfg = path_data.get("motion_config", {})
    linear_speed = float(motion_cfg.get("linear_speed", 0.45))
    angular_speed_deg = float(motion_cfg.get("angular_speed_deg", 45.0))
    angular_speed = math.radians(angular_speed_deg)

    start_pose = path_data.get("start_pose", {})
    start_theta = math.radians(float(start_pose.get("theta_deg", 90.0)))

    motion_segments = build_motion_segments(
        path=path,
        start_theta=start_theta,
        linear_speed=linear_speed,
        angular_speed=angular_speed,
    )

    # --------------------------------------------------------
    # Load human paths
    # --------------------------------------------------------
    human_data = load_human_paths(human_file)
    human_entities = build_human_entities(human_data)

    # --------------------------------------------------------
    # Sensors
    # --------------------------------------------------------
    gas_sensor = MQ135Sensor()
    thermal_camera = ThermalCameraMLX90640()
    human_detector = SimpleHumanDetector(
        detection_range=float(args.human_detection_range)
    )

    # --------------------------------------------------------
    # Matplotlib thermal window
    # --------------------------------------------------------
    thermal_fig, thermal_ax, thermal_im = setup_thermal_window()

    latest_thermal_img = np.zeros((24, 32), dtype=float) + 25.0
    latest_gas = None
    current_humans = []
    detected_humans = []

    # --------------------------------------------------------
    # Pygame
    # --------------------------------------------------------
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Fire Evacuation Robot - Map Viewer")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    big_font = pygame.font.SysFont("Arial", 24)

    transform = WorldTransform(
        mesh_xb=mesh_xb,
        left=MAP_LEFT,
        top=MAP_TOP,
        width=MAP_W,
        height=MAP_H,
    )

    running = True
    started = False
    paused = False

    sim_time = 0.0
    sensor_timer = 0.0
    sensor_period = 1.0 / max(args.sensor_hz, 0.1)

    print("\n[READY]")
    print("ENTER : start / reset")
    print("SPACE : pause / resume")
    print("R     : reset time")
    print(f"Human detection: distance <= {args.human_detection_range:.1f} m and no wall")
    print("ESC   : quit")

    while running:
        dt_real = clock.tick(FPS) / 1000.0
        dt_sim = dt_real * args.sim_speed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_RETURN:
                    started = True
                    paused = False
                    sim_time = 0.0
                    sensor_timer = sensor_period
                    gas_sensor = MQ135Sensor()
                    print("[START] Simulation reset and started")

                elif event.key == pygame.K_SPACE:
                    if started:
                        paused = not paused

                elif event.key == pygame.K_r:
                    sim_time = 0.0
                    sensor_timer = sensor_period
                    gas_sensor = MQ135Sensor()
                    latest_thermal_img = np.zeros((24, 32), dtype=float) + 25.0
                    latest_gas = None
                    print("[RESET] Simulation time reset")

        if started and not paused:
            sim_time += dt_sim
            sensor_timer += dt_sim

        # ----------------------------------------------------
        # Robot pose
        # ----------------------------------------------------
        robot_x, robot_y, robot_theta = pose_from_motion_segments(
            motion_segments,
            sim_time,
        )

        # ----------------------------------------------------
        # Human positions and simple recognition
        # ----------------------------------------------------
        current_humans = humans_at_time(
            human_entities,
            sim_time,
        )

        detected_humans = human_detector.detect(
            robot_position=(robot_x, robot_y),
            humans=current_humans,
            obstacle_map=obstacle_map_2d,
            map_origin=(0.0, 0.0),
            map_resolution=xy_res,
            use_line_of_sight=True,
        )

        detected_ids = {
            str(human.get("id", "victim"))
            for human in detected_humans
        }

        # ----------------------------------------------------
        # Sensor update
        # ----------------------------------------------------
        if sensor_timer >= sensor_period or latest_gas is None:
            sensor_dt = max(sensor_timer, 0.0)
            sensor_timer = 0.0

            # Gas sensor
            co_i = nearest_time_index(co_times, sim_time)
            co_i = min(co_i, co_data.shape[0] - 1)

            co_vf = sample_2d_xy(
                co_data[co_i],
                co_x,
                co_y,
                robot_x,
                robot_y,
            )

            true_co_ppm = 0.0 if np.isnan(co_vf) else co_vf * 1_000_000.0
            latest_gas = gas_sensor.read(true_co_ppm, sensor_dt)

            # Thermal camera
            temp_i = nearest_time_index(temp_times, sim_time)
            temp_i = min(temp_i, temp_data.shape[0] - 1)

            # load_temperature_npz 결과: (time, x, y, z)
            temp_volume_xyz = temp_data[temp_i]

            # thermal_camera.sense_3d() 입력: (z, y, x)
            temp_volume_zyx = np.transpose(temp_volume_xyz, (2, 1, 0))

            latest_thermal_img = thermal_camera.sense_3d(
                temperature_volume=temp_volume_zyx,
                robot_x=robot_x,
                robot_y=robot_y,
                robot_theta=robot_theta,
                xy_resolution=xy_res,
                z_resolution=z_res,
                obstacle_volume=obstacle_volume_zyx,
                camera_pitch=0.0,
            )

            update_thermal_window(
                fig=thermal_fig,
                ax=thermal_ax,
                im=thermal_im,
                thermal_img=latest_thermal_img,
                sim_time=sim_time,
                display_mode=args.thermal_scale,
            )
        else:
            # matplotlib 창이 멈춘 것처럼 보이지 않게 이벤트만 처리
            plt.pause(0.001)

        # ----------------------------------------------------
        # Draw pygame map
        # ----------------------------------------------------
        screen.fill((235, 235, 235))

        draw_map(
            screen=screen,
            transform=transform,
            mesh_xb=mesh_xb,
            obstacles=obstacles,
            holes=holes,
            burner_vents=burner_vents,
        )

        # path
        if len(path) >= 2:
            pts = [transform.world_to_screen(x, y) for x, y in path]
            pygame.draw.lines(screen, (255, 180, 0), False, pts, 3)

            for p in pts:
                pygame.draw.circle(screen, (255, 180, 0), p, 5)


        draw_detection_range(
            screen=screen,
            transform=transform,
            robot_x=robot_x,
            robot_y=robot_y,
            detection_range=float(args.human_detection_range),
        )

        draw_human_detection_lines(
            screen=screen,
            transform=transform,
            robot_x=robot_x,
            robot_y=robot_y,
            humans=current_humans,
            detected_ids=detected_ids,
        )

        for human in current_humans:
            draw_human(
                screen=screen,
                transform=transform,
                font=font,
                human=human,
                detected=str(human.get("id", "victim")) in detected_ids,
            )

        draw_robot(
            screen=screen,
            transform=transform,
            x=robot_x,
            y=robot_y,
            theta=robot_theta,
        )

        # side panel
        pygame.draw.rect(screen, (245, 245, 245), (730, 0, SCREEN_W - 730, SCREEN_H))
        pygame.draw.line(screen, (0, 0, 0), (730, 0), (730, SCREEN_H), 2)

        y_text = PANEL_Y

        draw_text(screen, big_font, "Robot Map", PANEL_X, y_text)
        y_text += 45

        draw_text(screen, font, f"started = {started}", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, f"paused = {paused}", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, f"t = {sim_time:.2f} s", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, f"x = {robot_x:.2f} m", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, f"y = {robot_y:.2f} m", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, f"theta = {math.degrees(robot_theta):.1f} deg", PANEL_X, y_text)
        y_text += 40

        draw_text(screen, big_font, "MQ-135", PANEL_X, y_text)
        y_text += 35

        if latest_gas is not None:
            true_ppm = float(gas_get(latest_gas, "true_ppm", 0.0))
            measured_ppm = float(gas_get(latest_gas, "measured_ppm", 0.0))
            voltage = float(gas_get(latest_gas, "voltage", 0.0))
            status = str(gas_get(latest_gas, "status", "UNKNOWN"))
            cost = gas_get(latest_gas, "cost", 0.0)

            cost_str = "inf" if math.isinf(cost) else f"{float(cost):.2f}"
            status_color = gas_status_color(status)

            draw_text(screen, font, f"true CO = {true_ppm:.1f} ppm", PANEL_X, y_text)
            y_text += 25

            draw_text(screen, font, f"MQ-135 = {measured_ppm:.1f} ppm", PANEL_X, y_text)
            y_text += 25

            draw_text(screen, font, f"voltage = {voltage:.3f} V", PANEL_X, y_text)
            y_text += 25

            draw_text(screen, font, f"status = {status}", PANEL_X, y_text, status_color)
            y_text += 25

            draw_text(screen, font, f"cost = {cost_str}", PANEL_X, y_text)
            y_text += 40

        thermal_min = float(np.nanmin(latest_thermal_img))
        thermal_max = float(np.nanmax(latest_thermal_img))

        draw_text(screen, big_font, "Thermal", PANEL_X, y_text)
        y_text += 35

        draw_text(screen, font, "Shown in matplotlib window", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, f"thermal min = {thermal_min:.1f} C", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, f"thermal max = {thermal_max:.1f} C", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, f"scale = {args.thermal_scale}", PANEL_X, y_text)
        y_text += 45

        draw_text(screen, big_font, "Human", PANEL_X, y_text)
        y_text += 35

        draw_text(screen, font, f"objects = {len(current_humans)}", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, f"detected = {len(detected_humans)}", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, f"range = {args.human_detection_range:.1f} m", PANEL_X, y_text)
        y_text += 25

        if len(detected_humans) > 0:
            nearest_human = detected_humans[0]
            nearest_id = str(nearest_human.get("id", "victim"))
            nearest_dist = float(nearest_human.get("distance", 0.0))

            draw_text(
                screen,
                font,
                f"nearest = {nearest_id}",
                PANEL_X,
                y_text,
                color=(180, 0, 40),
            )
            y_text += 25

            draw_text(
                screen,
                font,
                f"dist = {nearest_dist:.2f} m",
                PANEL_X,
                y_text,
                color=(180, 0, 40),
            )
            y_text += 25

        y_text += 20

        draw_text(screen, big_font, "Controls", PANEL_X, y_text)
        y_text += 35

        draw_text(screen, font, "ENTER : start / reset", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, "SPACE : pause / resume", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, "R : reset time", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, "ESC : quit", PANEL_X, y_text)

        if not started:
            draw_text(
                screen,
                big_font,
                "Press ENTER to start",
                240,
                420,
                color=(180, 0, 0),
            )

        pygame.display.flip()

    pygame.quit()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
