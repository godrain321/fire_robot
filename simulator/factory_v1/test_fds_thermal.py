# test_fds_thermal.py

import re
import numpy as np
import matplotlib.pyplot as plt
from math import atan2

from sensors.thermal_camera import ThermalCameraMLX90640


def load_fds2ascii_temperature_csv(
    csv_path,
    default_resolution=0.5,
):
    """
    fds2ascii에서 얻은 x, y, temperature 형식 CSV를 읽어서
    temp_map[y, x] 형태의 2D numpy array로 변환한다.

    CSV 형식 예:
        x, y, temperature
        0.25, 0.25, 25.3
        0.75, 0.25, 25.4
        ...
    """

    import re
    import numpy as np

    points = []

    number_pattern = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"

    with open(csv_path, "r", errors="ignore") as f:
        for line in f:
            nums = re.findall(number_pattern, line)

            # x, y, temperature 세 개 이상 있는 줄만 사용
            if len(nums) >= 3:
                values = [float(v.replace("D", "E").replace("d", "E")) for v in nums]

                x = values[0]
                y = values[1]
                temp = values[2]

                # factory_v1 범위 안의 점만 사용
                if 0.0 <= x <= 20.0 and 0.0 <= y <= 30.0:
                    if -100.0 <= temp <= 1500.0:
                        points.append((x, y, temp))

    if len(points) == 0:
        raise ValueError("x, y, temperature 형식의 데이터를 찾지 못했습니다.")

    points = np.array(points, dtype=np.float32)

    # 좌표값 오차 방지를 위해 반올림
    xs = np.unique(np.round(points[:, 0], 6))
    ys = np.unique(np.round(points[:, 1], 6))

    xs.sort()
    ys.sort()

    nx = len(xs)
    ny = len(ys)

    temp_sum = np.zeros((ny, nx), dtype=np.float32)
    temp_count = np.zeros((ny, nx), dtype=np.float32)

    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_i = {y: i for i, y in enumerate(ys)}

    for x, y, temp in points:
        xr = round(float(x), 6)
        yr = round(float(y), 6)

        ix = x_to_i[xr]
        iy = y_to_i[yr]

        temp_sum[iy, ix] += temp
        temp_count[iy, ix] += 1.0

    temp_map = temp_sum / np.maximum(temp_count, 1.0)

    # 혹시 빈 칸이 있으면 평균값으로 채움
    empty = temp_count == 0
    if np.any(empty):
        temp_map[empty] = np.mean(temp_map[~empty])

    # 해상도 계산
    if len(xs) > 1:
        dx = float(np.median(np.diff(xs)))
    else:
        dx = default_resolution

    if len(ys) > 1:
        dy = float(np.median(np.diff(ys)))
    else:
        dy = default_resolution

    map_resolution = (dx + dy) / 2.0

    print("CSV point count:", len(points))
    print("x range:", xs[0], "~", xs[-1], "nx:", nx)
    print("y range:", ys[0], "~", ys[-1], "ny:", ny)
    print("dx:", dx, "dy:", dy)

    return temp_map, map_resolution

def choose_robot_pose_near_hotspot(temp_map, map_resolution):
    """
    가장 뜨거운 지점을 찾고,
    그 지점에서 약 5m 떨어진 곳에 로봇을 배치한 뒤
    화재 방향을 바라보도록 theta를 설정한다.
    """

    hot_iy, hot_ix = np.unravel_index(np.argmax(temp_map), temp_map.shape)

    hot_x = hot_ix * map_resolution
    hot_y = hot_iy * map_resolution

    # 기본은 화재 왼쪽 5m 지점에서 오른쪽을 바라보게 설정
    robot_x = hot_x - 5.0
    robot_y = hot_y

    # 만약 맵 밖이면 아래쪽에서 위쪽을 바라보게 설정
    if robot_x < 1.0:
        robot_x = hot_x
        robot_y = hot_y - 5.0

    # 경계 안으로 제한
    robot_x = np.clip(robot_x, 1.0, 19.0)
    robot_y = np.clip(robot_y, 1.0, 29.0)

    robot_theta = atan2(hot_y - robot_y, hot_x - robot_x)

    return robot_x, robot_y, robot_theta, hot_x, hot_y


def main():
    csv_path = "factory_v1_temp_z050_t60.csv"

    temp_map, map_resolution = load_fds2ascii_temperature_csv(csv_path)

    print("Loaded temperature map")
    print("shape:", temp_map.shape)
    print("map_resolution:", map_resolution)
    print("min temp:", np.min(temp_map))
    print("max temp:", np.max(temp_map))

    robot_x, robot_y, robot_theta, hot_x, hot_y = choose_robot_pose_near_hotspot(
        temp_map,
        map_resolution
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

    # 지금은 FDS 온도 데이터 연결 테스트가 목적이라 obstacle_map은 일단 None
    thermal_img = camera.sense(
        temperature_map=temp_map,
        obstacle_map=None,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_theta=robot_theta,
        map_resolution=map_resolution,
    )

    print()
    print("Thermal image")
    print("shape:", thermal_img.shape)
    print("min:", thermal_img.min())
    print("max:", thermal_img.max())

    np.savetxt(
        "thermal_mlx90640_from_fds_t60.csv",
        thermal_img,
        delimiter=",",
        fmt="%.3f"
    )

    # Ground truth map
    plt.figure()
    plt.imshow(
        temp_map,
        origin="lower",
        extent=[0, temp_map.shape[1] * map_resolution,
                0, temp_map.shape[0] * map_resolution]
    )
    plt.colorbar(label="Temperature (C)")
    plt.scatter(robot_x, robot_y, marker="x", label="Robot")
    plt.scatter(hot_x, hot_y, marker="o", label="Hotspot")
    plt.legend()
    plt.title("FDS Ground Truth Temperature Map at z=0.5m, t=60s")

    # Simulated thermal camera image
    plt.figure()
    plt.imshow(thermal_img)
    plt.colorbar(label="Temperature (C)")
    plt.title("Simulated MLX90640 Image from FDS Data")

    plt.show()


if __name__ == "__main__":
    main()
