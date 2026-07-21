# test_fds_thermal_3d.py

import re
import numpy as np
import matplotlib.pyplot as plt
from math import atan2

from sensors.thermal_camera import ThermalCameraMLX90640


def load_fds2ascii_temperature_3d_csv(csv_path):
    """
    fds2ascii에서 얻은 3D 온도 CSV를 읽어서
    temp_volume[z, y, x] 형태로 변환한다.

    예상 CSV 형식:
        x, y, z, temperature
    """

    points = []

    number_pattern = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"

    with open(csv_path, "r", errors="ignore") as f:
        for line in f:
            nums = re.findall(number_pattern, line)

            if len(nums) >= 4:
                values = [float(v.replace("D", "E").replace("d", "E")) for v in nums]

                x = values[0]
                y = values[1]
                z = values[2]
                temp = values[3]

                if 0.0 <= x <= 20.0 and 0.0 <= y <= 30.0 and 0.0 <= z <= 5.0:
                    if -100.0 <= temp <= 1500.0:
                        points.append((x, y, z, temp))

    if len(points) == 0:
        raise ValueError("x, y, z, temperature 형식의 3D 데이터를 찾지 못했습니다.")

    points = np.array(points, dtype=np.float32)

    xs = np.unique(np.round(points[:, 0], 6))
    ys = np.unique(np.round(points[:, 1], 6))
    zs = np.unique(np.round(points[:, 2], 6))

    xs.sort()
    ys.sort()
    zs.sort()

    nx = len(xs)
    ny = len(ys)
    nz = len(zs)

    temp_sum = np.zeros((nz, ny, nx), dtype=np.float32)
    temp_count = np.zeros((nz, ny, nx), dtype=np.float32)

    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_i = {y: i for i, y in enumerate(ys)}
    z_to_i = {z: i for i, z in enumerate(zs)}

    for x, y, z, temp in points:
        xr = round(float(x), 6)
        yr = round(float(y), 6)
        zr = round(float(z), 6)

        ix = x_to_i[xr]
        iy = y_to_i[yr]
        iz = z_to_i[zr]

        temp_sum[iz, iy, ix] += temp
        temp_count[iz, iy, ix] += 1.0

    temp_volume = temp_sum / np.maximum(temp_count, 1.0)

    empty = temp_count == 0
    if np.any(empty):
        temp_volume[empty] = np.mean(temp_volume[~empty])

    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.5
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 0.5
    dz = float(np.median(np.diff(zs))) if len(zs) > 1 else 0.25

    xy_resolution = (dx + dy) / 2.0
    z_resolution = dz

    print("CSV point count:", len(points))
    print("x range:", xs[0], "~", xs[-1], "nx:", nx)
    print("y range:", ys[0], "~", ys[-1], "ny:", ny)
    print("z range:", zs[0], "~", zs[-1], "nz:", nz)
    print("dx:", dx, "dy:", dy, "dz:", dz)

    return temp_volume, xy_resolution, z_resolution


def choose_robot_pose_near_hotspot_3d(temp_volume, xy_resolution):
    """
    3D 온도장에서 가장 뜨거운 지점을 찾고,
    그 근처에 로봇을 배치해서 화재 방향을 바라보게 한다.
    """

    iz, iy, ix = np.unravel_index(np.argmax(temp_volume), temp_volume.shape)

    hot_x = ix * xy_resolution
    hot_y = iy * xy_resolution

    robot_x = hot_x - 5.0
    robot_y = hot_y

    if robot_x < 1.0:
        robot_x = hot_x
        robot_y = hot_y - 5.0

    robot_x = np.clip(robot_x, 1.0, 19.0)
    robot_y = np.clip(robot_y, 1.0, 29.0)

    robot_theta = atan2(hot_y - robot_y, hot_x - robot_x)

    return robot_x, robot_y, robot_theta, hot_x, hot_y


def main():
    csv_path = "factory_v1_temp_3d_t60.csv"

    temp_volume, xy_resolution, z_resolution = load_fds2ascii_temperature_3d_csv(csv_path)

    print()
    print("Loaded 3D temperature volume")
    print("shape:", temp_volume.shape)
    print("xy_resolution:", xy_resolution)
    print("z_resolution:", z_resolution)
    print("min temp:", np.min(temp_volume))
    print("max temp:", np.max(temp_volume))

    robot_x, robot_y, robot_theta, hot_x, hot_y = choose_robot_pose_near_hotspot_3d(
        temp_volume,
        xy_resolution
    )

    print()
    print("Hotspot position:")
    print("hot_x:", hot_x, "hot_y:", hot_y)

    print()
    print("Robot pose:")
    print("robot_x:", robot_x)
    print("robot_y:", robot_y)
    print("robot_theta(rad):", robot_theta)

    camera = ThermalCameraMLX90640(
        width=32,
        height=24,
        fov_h_deg=110.0,
        fov_v_deg=75.0,
        camera_height=0.30,
        front_offset=0.30,
        min_range=0.20,
        max_range=7.0,
        noise_std=0.5,
        accuracy_limit=2.0,
        measurement_mode="max",
    )

    thermal_img = camera.sense_3d(
        temperature_volume=temp_volume,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_theta=robot_theta,
        xy_resolution=xy_resolution,
        z_resolution=z_resolution,
        obstacle_volume=None,
        camera_pitch=0.0,
    )

    print()
    print("Thermal image 3D")
    print("shape:", thermal_img.shape)
    print("min:", thermal_img.min())
    print("max:", thermal_img.max())

    np.savetxt(
        "thermal_mlx90640_from_fds_3d_t60.csv",
        thermal_img,
        delimiter=",",
        fmt="%.3f"
    )

    # 3D 온도장을 위에서 본 최대온도 지도
    topdown_max_temp = np.max(temp_volume, axis=0)

    plt.figure()
    plt.imshow(
        topdown_max_temp,
        origin="lower",
        extent=[
            0,
            topdown_max_temp.shape[1] * xy_resolution,
            0,
            topdown_max_temp.shape[0] * xy_resolution,
        ]
    )
    plt.colorbar(label="Max Temperature over Z (C)")
    plt.scatter(robot_x, robot_y, marker="x", label="Robot")
    plt.scatter(hot_x, hot_y, marker="o", label="Hotspot")
    plt.legend()
    plt.title("FDS 3D Temperature Volume - Top View Max Projection")

    plt.figure()
    plt.imshow(thermal_img)
    plt.colorbar(label="Temperature (C)")
    plt.title("Simulated MLX90640 Image from FDS 3D Data")

    plt.show()


if __name__ == "__main__":
    main()
