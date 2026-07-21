# pygame_map_matplotlib_thermal_viewer.py

import argparse
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
from mapping.grid_map import GridMap, robot_rotation_clearance
from planner.a_star import a_star


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

# Robot mission settings (FDS world coordinates, metres).
# The default goal is the centre of the right-hand exit in factory_v1.fds.
START_POSITION = (1.0, 1.0)
EXIT_POSITION = (20.0, 15.0)
START_THETA_DEG = 90.0
GRID_RESOLUTION = 0.5
ROBOT_LENGTH = 0.70
ROBOT_WIDTH = 0.45
ROBOT_LINEAR_SPEED = 0.45
ROBOT_ANGULAR_SPEED_DEG = 45.0


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
# ROBOT MOTION
# ============================================================
def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def update_robot_pose(x, y, theta, path, waypoint_index, dt):
    """Use frame dt to rotate toward a waypoint and then drive forward."""
    if waypoint_index >= len(path):
        return x, y, theta, waypoint_index, True

    target_x, target_y = path[waypoint_index]
    dx = target_x - x
    dy = target_y - y
    distance = math.hypot(dx, dy)

    if distance <= 1e-4:
        waypoint_index += 1
        return x, y, theta, waypoint_index, waypoint_index >= len(path)

    target_theta = math.atan2(dy, dx)
    angle_error = wrap_angle(target_theta - theta)
    angular_speed = math.radians(ROBOT_ANGULAR_SPEED_DEG)
    max_turn = angular_speed * dt

    if abs(angle_error) > math.radians(1.0):
        theta = wrap_angle(theta + max(-max_turn, min(max_turn, angle_error)))
        return x, y, theta, waypoint_index, False

    theta = target_theta
    step = min(ROBOT_LINEAR_SPEED * dt, distance)
    x += math.cos(theta) * step
    y += math.sin(theta) * step

    if step >= distance - 1e-9:
        x, y = target_x, target_y
        waypoint_index += 1

    return x, y, theta, waypoint_index, waypoint_index >= len(path)


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
    parser.add_argument("--start", type=float, nargs=2, default=START_POSITION,
                        metavar=("X", "Y"), help="robot start in world metres")
    parser.add_argument("--goal", type=float, nargs=2, default=EXIT_POSITION,
                        metavar=("X", "Y"), help="A* goal in world metres")
    parser.add_argument("--grid-resolution", type=float, default=GRID_RESOLUTION)
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

    # --------------------------------------------------------
    # Build the static occupancy grid and plan to the exit
    # --------------------------------------------------------
    start_position = tuple(args.start)
    goal_position = tuple(args.goal)
    clearance = robot_rotation_clearance(ROBOT_LENGTH, ROBOT_WIDTH)
    grid_map = GridMap(
        mesh_xb=mesh_xb,
        obstacles=obstacles,
        holes=holes,
        resolution=args.grid_resolution,
        clearance=clearance,
    )
    start_grid = grid_map.world_to_grid(*start_position)
    goal_grid = grid_map.world_to_grid(*goal_position)
    grid_path = a_star(grid_map, start_grid, goal_grid)
    if grid_path is None:
        print(f"[ERROR] A* path not found: {start_grid} -> {goal_grid}")
        sys.exit(1)

    path = [grid_map.grid_to_world(gx, gy) for gx, gy in grid_path]
    start_theta = math.radians(START_THETA_DEG)
    print(f"[A*] {start_grid} -> {goal_grid}, nodes={len(grid_path)}")

    # --------------------------------------------------------
    # Sensors
    # --------------------------------------------------------
    gas_sensor = MQ135Sensor()
    thermal_camera = ThermalCameraMLX90640()

    # --------------------------------------------------------
    # Matplotlib thermal window
    # --------------------------------------------------------
    thermal_fig, thermal_ax, thermal_im = setup_thermal_window()

    latest_thermal_img = np.zeros((24, 32), dtype=float) + 25.0
    latest_gas = None

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
    robot_x, robot_y = start_position
    robot_theta = start_theta
    waypoint_index = 1
    robot_stopped = len(path) <= 1

    print("\n[READY]")
    print("ENTER : start / reset")
    print("SPACE : pause / resume")
    print("R     : reset time")
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
                    robot_x, robot_y = start_position
                    robot_theta = start_theta
                    waypoint_index = 1
                    robot_stopped = len(path) <= 1
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
                    robot_x, robot_y = start_position
                    robot_theta = start_theta
                    waypoint_index = 1
                    robot_stopped = len(path) <= 1
                    print("[RESET] Simulation time reset")

        if started and not paused:
            sim_time += dt_sim
            sensor_timer += dt_sim
            if not robot_stopped:
                robot_x, robot_y, robot_theta, waypoint_index, robot_stopped = update_robot_pose(
                    robot_x, robot_y, robot_theta, path, waypoint_index, dt_sim
                )

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
        y_text += 25

        draw_text(screen, font, f"A* nodes = {len(grid_path)}", PANEL_X, y_text)
        y_text += 25

        draw_text(screen, font, f"goal reached = {robot_stopped}", PANEL_X, y_text)
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
