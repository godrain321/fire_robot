# test_thermal.py

import numpy as np
import matplotlib.pyplot as plt

from sensors.thermal_camera import ThermalCameraMLX90640


# ------------------------------------------------------------
# 1. 테스트용 온도맵 생성
# 실제 공장 크기: 20m x 30m
# 해상도: 0.5m/cell
# shape = (Y 방향 60칸, X 방향 40칸)
# ------------------------------------------------------------

map_resolution = 0.5
temp_map = np.ones((60, 40), dtype=np.float32) * 25.0

# ------------------------------------------------------------
# 2. 테스트용 장애물맵 생성
# True = 벽/장애물
# False = 빈 공간
# ------------------------------------------------------------

obstacle_map = np.zeros((60, 40), dtype=bool)

# 외벽
obstacle_map[0, :] = True
obstacle_map[-1, :] = True
obstacle_map[:, 0] = True
obstacle_map[:, -1] = True

# 선반 몇 줄 추가
# y index 기준으로 긴 선반 생성
obstacle_map[16:17, 5:35] = True
obstacle_map[22:23, 5:35] = True
obstacle_map[28:29, 5:35] = True
obstacle_map[34:35, 5:35] = True

# ------------------------------------------------------------
# 3. 화재/고온 영역 설정
# ------------------------------------------------------------

# 화재 중심 위치를 실제 좌표로 생각하면 대략 x=15m, y=18m 부근
# grid index로는 x=30, y=36 근처
temp_map[32:41, 26:35] = 70.0
temp_map[34:39, 28:33] = 120.0

# ------------------------------------------------------------
# 4. 로봇 위치 설정
# ------------------------------------------------------------

robot_x = 10.0
robot_y = 10.0

# 위쪽 방향을 바라봄
# 좌표계에서 +Y 방향
robot_theta = np.deg2rad(90.0)

# ------------------------------------------------------------
# 5. 열화상 카메라 생성
# ------------------------------------------------------------

camera = ThermalCameraMLX90640(
    width=32,
    height=24,
    fov_h_deg=110.0,
    fov_v_deg=75.0,
    camera_height=0.30,
    front_offset=0.30,
    min_range=0.20,
    max_range=12.0,
    noise_std=0.5,
    accuracy_limit=2.0,
    measurement_mode="max",
)

thermal_img = camera.sense(
    temperature_map=temp_map,
    obstacle_map=None,
    robot_x=robot_x,
    robot_y=robot_y,
    robot_theta=robot_theta,
    map_resolution=map_resolution,
)

print("Thermal image shape:", thermal_img.shape)
print("Min temp:", thermal_img.min())
print("Max temp:", thermal_img.max())

# ------------------------------------------------------------
# 6. 결과 시각화
# ------------------------------------------------------------

plt.figure()
plt.imshow(temp_map, origin="lower")
plt.colorbar(label="Temperature (C)")
plt.scatter(robot_x / map_resolution, robot_y / map_resolution, marker="x")
plt.title("Ground Truth Temperature Map")

plt.figure()
plt.imshow(thermal_img)
plt.colorbar(label="Temperature (C)")
plt.title("Simulated MLX90640 Thermal Image")

plt.show()