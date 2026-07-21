import argparse
import math
import time
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import fdsreader as fds


# ============================================================
# MQ-135 SENSOR MODEL
# ============================================================

@dataclass
class MQ135Config:
    min_ppm: float = 10.0
    max_ppm: float = 1000.0
    blocked_ppm: float = 950.0
    danger_ppm: float = 900.0
    warning_ppm: float = 700.0
    response_tau: float = 2.0
    noise_std_ratio: float = 0.03
    noise_std_min: float = 5.0
    vc: float = 5.0
    rl: float = 10_000.0
    rs_100ppm: float = 10_000.0
    alpha: float = 0.6


class MQ135Sensor:
    def __init__(self, config=None):
        self.config = config if config is not None else MQ135Config()
        self.filtered_ppm = self.config.min_ppm

    def read(self, true_ppm, dt):
        cfg = self.config

        true_ppm = max(0.0, float(true_ppm))
        saturated = true_ppm >= cfg.max_ppm

        target_ppm = np.clip(true_ppm, cfg.min_ppm, cfg.max_ppm)

        if dt <= 0:
            a = 1.0
        else:
            a = 1.0 - math.exp(-dt / cfg.response_tau)

        self.filtered_ppm += a * (target_ppm - self.filtered_ppm)

        noise_std = max(cfg.noise_std_min, abs(self.filtered_ppm) * cfg.noise_std_ratio)
        measured_ppm = self.filtered_ppm + np.random.normal(0.0, noise_std)
        measured_ppm = float(np.clip(measured_ppm, cfg.min_ppm, cfg.max_ppm))

        voltage = self.ppm_to_voltage(measured_ppm)
        cost = self.ppm_to_cost(measured_ppm)
        status = self.ppm_to_status(measured_ppm)

        return {
            "true_ppm": true_ppm,
            "measured_ppm": measured_ppm,
            "voltage": voltage,
            "cost": cost,
            "status": status,
            "saturated": saturated,
        }

    def ppm_to_cost(self, ppm):
        cfg = self.config

        if ppm >= cfg.blocked_ppm:
            return float("inf")

        normalized = ppm / cfg.blocked_ppm
        return float(1.0 + 99.0 * normalized ** 2)

    def ppm_to_status(self, ppm):
        cfg = self.config

        if ppm >= cfg.blocked_ppm:
            return "BLOCKED"
        elif ppm >= cfg.danger_ppm:
            return "DANGER"
        elif ppm >= cfg.warning_ppm:
            return "WARNING"
        elif ppm >= 300.0:
            return "CAUTION"
        else:
            return "SAFE"

    def ppm_to_voltage(self, ppm):
        cfg = self.config

        ppm = max(cfg.min_ppm, min(ppm, cfg.max_ppm))
        rs = cfg.rs_100ppm * ((100.0 / ppm) ** cfg.alpha)
        return float(cfg.vc * cfg.rl / (rs + cfg.rl))


# ============================================================
# TEMPERATURE NPZ LOAD
# ============================================================

def make_axis_coords(n, min_v, max_v, resolution):
    """
    데이터 개수에 따라 cell-center 좌표 또는 boundary 좌표를 자동 생성.
    """
    length = max_v - min_v
    n_center = int(round(length / resolution))
    n_boundary = n_center + 1

    if n == n_center:
        return min_v + resolution / 2.0 + np.arange(n) * resolution

    if n == n_boundary:
        return np.linspace(min_v, max_v, n)

    return np.linspace(min_v, max_v, n)


def load_temperature_npz(npz_path, x_max=20.0, y_max=30.0, z_max=5.0):
    """
    processed/fds_temperature_3d_timeseries.npz 읽기.

    npz 내부 key:
    - temperature
    - times
    - xy_resolution
    - z_resolution
    """

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
        raise RuntimeError(f"temperature는 4차원이어야 해. 현재 shape={temp.shape}")

    if temp.shape[0] != len(times):
        raise RuntimeError(
            "temperature의 첫 번째 축이 time이 아닌 것 같아. "
            f"temperature shape={temp.shape}, len(times)={len(times)}"
        )

    nx_expected = int(round(x_max / xy_res))
    ny_expected = int(round(y_max / xy_res))
    nz_expected = int(round(z_max / z_res))

    spatial_shape = temp.shape[1:]

    # spatial axes를 x, y, z 순서로 자동 정렬
    best_perm = None
    best_score = 1e9

    for perm in permutations(range(3)):
        sx = spatial_shape[perm[0]]
        sy = spatial_shape[perm[1]]
        sz = spatial_shape[perm[2]]

        def score_dim(value, expected):
            # center 개수 또는 boundary 개수 모두 허용
            return min(abs(value - expected), abs(value - (expected + 1)))

        score = (
            score_dim(sx, nx_expected)
            + score_dim(sy, ny_expected)
            + score_dim(sz, nz_expected)
        )

        if score < best_score:
            best_score = score
            best_perm = perm

    temp = np.transpose(temp, (0, 1 + best_perm[0], 1 + best_perm[1], 1 + best_perm[2]))

    nx, ny, nz = temp.shape[1:]

    x_coords = make_axis_coords(nx, 0.0, x_max, xy_res)
    y_coords = make_axis_coords(ny, 0.0, y_max, xy_res)
    z_coords = make_axis_coords(nz, 0.0, z_max, z_res)

    print("[INFO] Normalized temperature data")
    print(f"  normalized shape: {temp.shape}")
    print(f"  x coords: {len(x_coords)}, {x_coords.min():.2f} ~ {x_coords.max():.2f}")
    print(f"  y coords: {len(y_coords)}, {y_coords.min():.2f} ~ {y_coords.max():.2f}")
    print(f"  z coords: {len(z_coords)}, {z_coords.min():.2f} ~ {z_coords.max():.2f}")

    return temp, times, x_coords, y_coords, z_coords


# ============================================================
# FDS CO SLICE LOAD
# ============================================================

def find_co_slice(sim):
    print("\n[INFO] FDS slices found:")
    for i, slc in enumerate(sim.slices):
        print(f"  [{i}] {slc}")

    # 1순위: ID에 CO_Z130이 보이면 선택
    for slc in sim.slices:
        text = str(slc).upper()
        if "CO_Z130" in text:
            print("\n[INFO] Selected CO slice by ID: CO_Z130")
            return slc

    # 2순위: CARBON MONOXIDE가 포함된 slice 선택
    for slc in sim.slices:
        text = str(slc).upper()
        if "CARBON MONOXIDE" in text:
            print("\n[INFO] Selected CO slice by quantity: CARBON MONOXIDE")
            return slc

    raise RuntimeError("CO_Z130 또는 CARBON MONOXIDE slice를 찾지 못했어.")


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


def sample_3d_xyz(data_3d, x_coords, y_coords, z_coords, x, y, z):
    ix = int(np.nanargmin(np.abs(x_coords - x)))
    iy = int(np.nanargmin(np.abs(y_coords - y)))
    iz = int(np.nanargmin(np.abs(z_coords - z)))

    if data_3d.shape == (len(x_coords), len(y_coords), len(z_coords)):
        return float(data_3d[ix, iy, iz])

    raise RuntimeError(
        f"3D shape mismatch: data={data_3d.shape}, "
        f"x={len(x_coords)}, y={len(y_coords)}, z={len(z_coords)}"
    )


def nearest_time_index(times, t):
    return int(np.nanargmin(np.abs(times - t)))


# ============================================================
# ROBOT PATH
# ============================================================

def interpolate_waypoints(waypoints, speed_mps, times):
    waypoints = np.asarray(waypoints, dtype=float)

    segments = []
    total_length = 0.0

    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        length = np.linalg.norm(p1 - p0)
        segments.append((p0, p1, length))
        total_length += length

    positions = []

    for t in times:
        dist = speed_mps * t

        if dist >= total_length:
            positions.append(waypoints[-1])
            continue

        acc = 0.0
        for p0, p1, length in segments:
            if acc + length >= dist:
                ratio = (dist - acc) / length
                pos = p0 + ratio * (p1 - p0)
                positions.append(pos)
                break
            acc += length

    return np.asarray(positions)


def get_robot_yaw(positions, idx):
    if idx < len(positions) - 1:
        dx = positions[idx + 1, 0] - positions[idx, 0]
        dy = positions[idx + 1, 1] - positions[idx, 1]
    else:
        dx = positions[idx, 0] - positions[idx - 1, 0]
        dy = positions[idx, 1] - positions[idx - 1, 1]

    return math.atan2(dy, dx)


# ============================================================
# THERMAL CAMERA
# ============================================================

def make_thermal_image(
    temp_3d,
    x_coords,
    y_coords,
    z_coords,
    robot_x,
    robot_y,
    robot_z,
    yaw,
    image_w=32,
    image_h=24,
    hfov_deg=55.0,
    vfov_deg=35.0,
    max_range=8.0,
    num_depth_samples=35,
):
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)

    image = np.full((image_h, image_w), np.nan, dtype=float)
    depths = np.linspace(0.3, max_range, num_depth_samples)

    for v in range(image_h):
        pitch = (0.5 - v / (image_h - 1)) * vfov

        for u in range(image_w):
            yaw_pixel = yaw + (u / (image_w - 1) - 0.5) * hfov

            values = []

            for d in depths:
                x = robot_x + d * math.cos(pitch) * math.cos(yaw_pixel)
                y = robot_y + d * math.cos(pitch) * math.sin(yaw_pixel)
                z = robot_z + d * math.sin(pitch)

                if (
                    x < x_coords.min() or x > x_coords.max()
                    or y < y_coords.min() or y > y_coords.max()
                    or z < z_coords.min() or z > z_coords.max()
                ):
                    continue

                temp = sample_3d_xyz(
                    temp_3d,
                    x_coords,
                    y_coords,
                    z_coords,
                    x,
                    y,
                    z
                )

                if not np.isnan(temp):
                    values.append(temp)

            if values:
                image[v, u] = np.nanmax(values)

    return np.nan_to_num(image, nan=20.0)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fds-dir", type=str, default=".", help="FDS 결과 폴더")
    parser.add_argument(
        "--temp-npz",
        type=str,
        default="processed/fds_temperature_3d_timeseries.npz",
        help="3D 온도 npz 파일 경로"
    )
    parser.add_argument("--speed", type=float, default=0.45, help="로봇 속도 [m/s]")
    parser.add_argument("--delay", type=float, default=0.05, help="화면 업데이트 delay")
    parser.add_argument("--step", type=int, default=1, help="몇 step마다 화면 업데이트할지")
    args = parser.parse_args()

    fds_dir = Path(args.fds_dir).resolve()
    temp_npz = Path(args.temp_npz).resolve()

    # 온도 npz 읽기
    temp_data, temp_times, temp_x, temp_y, temp_z = load_temperature_npz(temp_npz)

    # CO slice 읽기
    co_data, co_times, co_x, co_y = load_co_slice(fds_dir)

    # 로봇 경로
    waypoints = [
        (1.0, 1.0),
        (1.0, 21.0),
        (7.0, 21.0),
        (16.5, 25.5),
    ]

    positions = interpolate_waypoints(
        waypoints=waypoints,
        speed_mps=args.speed,
        times=co_times
    )

    gas_sensor = MQ135Sensor()
    thermal_camera_z = 0.30

    # 화면 준비
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax_thermal = axes[0]
    ax_map = axes[1]

    image0 = np.zeros((24, 32))
    im = ax_thermal.imshow(
        image0,
        origin="upper",
        vmin=20,
        vmax=120,
        aspect="auto"
    )
    ax_thermal.set_title("Thermal Camera")
    plt.colorbar(im, ax=ax_thermal, label="Temperature [C]")

    ax_map.set_xlim(0, 20)
    ax_map.set_ylim(0, 30)
    ax_map.set_aspect("equal")
    ax_map.set_title("Robot + MQ-135 Gas Sensor")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")

    path_arr = np.asarray(waypoints)
    ax_map.plot(path_arr[:, 0], path_arr[:, 1], linestyle="--", label="path")
    robot_point, = ax_map.plot([], [], marker="o", label="robot")
    ax_map.plot([16.5], [25.5], marker="x", markersize=10, label="fire")

    info_text = ax_map.text(
        0.02,
        0.98,
        "",
        transform=ax_map.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax_map.legend(loc="lower right")

    print("\n[START] Thermal camera + MQ-135 replay")
    print("time[s], x[m], y[m], true_CO[ppm], measured[ppm], voltage[V], status, cost")

    prev_t = co_times[0]

    max_steps = min(len(co_times), co_data.shape[0], len(positions))

    for i in range(0, max_steps, args.step):
        t = co_times[i]
        dt = float(t - prev_t) if i > 0 else 0.0
        prev_t = t

        robot_x, robot_y = positions[i]
        yaw = get_robot_yaw(positions, i)

        # -----------------------------
        # Gas sensor
        # -----------------------------
        co_vf = sample_2d_xy(
            co_data[i],
            co_x,
            co_y,
            robot_x,
            robot_y
        )

        if np.isnan(co_vf):
            true_co_ppm = 0.0
        else:
            true_co_ppm = co_vf * 1_000_000.0

        gas = gas_sensor.read(true_co_ppm, dt)

        cost_str = "inf" if math.isinf(gas["cost"]) else f"{gas['cost']:.2f}"

        print(
            f"{t:6.1f}, "
            f"{robot_x:5.2f}, {robot_y:5.2f}, "
            f"{gas['true_ppm']:9.2f}, "
            f"{gas['measured_ppm']:9.2f}, "
            f"{gas['voltage']:6.3f}, "
            f"{gas['status']:8s}, "
            f"{cost_str}"
        )

        # -----------------------------
        # Thermal camera
        # -----------------------------
        temp_i = nearest_time_index(temp_times, t)

        thermal_img = make_thermal_image(
            temp_3d=temp_data[temp_i],
            x_coords=temp_x,
            y_coords=temp_y,
            z_coords=temp_z,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_z=thermal_camera_z,
            yaw=yaw,
            image_w=32,
            image_h=24,
            hfov_deg=55.0,
            vfov_deg=35.0,
            max_range=8.0,
            num_depth_samples=35,
        )

        im.set_data(thermal_img)
        im.set_clim(vmin=20, vmax=max(80, float(np.nanmax(thermal_img))))

        ax_thermal.set_title(f"Thermal Camera  t={t:.1f}s")

        robot_point.set_data([robot_x], [robot_y])

        info_text.set_text(
            f"t = {t:.1f} s\n"
            f"x = {robot_x:.2f} m\n"
            f"y = {robot_y:.2f} m\n"
            f"true CO = {gas['true_ppm']:.1f} ppm\n"
            f"MQ-135 = {gas['measured_ppm']:.1f} ppm\n"
            f"voltage = {gas['voltage']:.3f} V\n"
            f"status = {gas['status']}\n"
            f"cost = {cost_str}"
        )

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(args.delay)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()