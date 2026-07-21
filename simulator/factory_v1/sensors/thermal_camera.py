# sensors/thermal_camera.py

import numpy as np
from math import sin, cos, radians


class ThermalCameraMLX90640:
    """
    MLX90640 열화상 카메라 시뮬레이션 모델

    실제 센서 사양 반영:
    - 해상도: 32 x 24
    - FOV: 110° x 75°
    - 측정 범위: -40 ~ 300 °C
    - 정확도: 약 ±2 °C
    - 위치: 로봇 전면 중앙, 지면에서 0.30 m

    현재 버전은 2D temperature map 기반의 단순 Ray Casting 모델이다.
    나중에 FDS 3D 온도장을 읽으면 3D Ray Casting으로 확장할 수 있다.
    """

    def __init__(
        self,
        width=32,
        height=24,
        fov_h_deg=110.0,
        fov_v_deg=75.0,
        camera_height=0.30,
        front_offset=0.30,
        min_range=0.20,
        max_range=7.0,
        temp_min=-40.0,
        temp_max=300.0,
        accuracy_limit=2.0,
        noise_std=0.5,
        ambient_temp=25.0,
        measurement_mode="max",
    ):
        self.width = width
        self.height = height

        self.fov_h = radians(fov_h_deg)
        self.fov_v = radians(fov_v_deg)

        self.camera_height = camera_height
        self.front_offset = front_offset

        self.min_range = min_range
        self.max_range = max_range

        self.temp_min = temp_min
        self.temp_max = temp_max

        self.accuracy_limit = accuracy_limit
        self.noise_std = noise_std
        self.ambient_temp = ambient_temp

        # "max" : ray 경로에서 가장 높은 온도 측정
        # "last": ray 끝 지점 온도 측정
        self.measurement_mode = measurement_mode

    def sense(
        self,
        temperature_map,
        robot_x,
        robot_y,
        robot_theta,
        map_resolution=0.5,
        obstacle_map=None,
    ):
        """
        temperature_map:
            2D numpy array.
            shape = (y_cells, x_cells)
            값 = 온도 [°C]

        robot_x, robot_y:
            로봇 중심 위치 [m]

        robot_theta:
            로봇 방향 [rad]
            예: np.deg2rad(90)

        map_resolution:
            temperature_map 한 칸의 실제 크기 [m/cell]

        obstacle_map:
            2D numpy array, bool.
            True = 벽/선반/장애물
            False = 빈 공간
            없으면 벽 가림 없이 온도만 샘플링함.

        return:
            thermal_image, shape = (24, 32)
        """

        if not isinstance(temperature_map, np.ndarray):
            temperature_map = np.array(temperature_map, dtype=np.float32)

        if obstacle_map is not None and not isinstance(obstacle_map, np.ndarray):
            obstacle_map = np.array(obstacle_map, dtype=bool)

        thermal_img = np.zeros((self.height, self.width), dtype=np.float32)

        # 로봇 중심이 아니라, 로봇 전면 중앙에 카메라가 있다고 가정
        cam_x = robot_x + self.front_offset * cos(robot_theta)
        cam_y = robot_y + self.front_offset * sin(robot_theta)

        for row in range(self.height):
            for col in range(self.width):

                # 가로 방향 FOV
                # col=0은 왼쪽 끝, col=31은 오른쪽 끝

                h_ratio = (col + 0.5) / self.width

                # col=0은 화면 왼쪽 픽셀이어야 하므로
                # 로봇 기준 왼쪽 방향, 즉 +h_angle을 보게 한다.
                h_angle = (0.5 - h_ratio) * self.fov_h
                ray_theta = robot_theta + h_angle

                # 세로 방향 FOV는 현재 2D map에서는 깊이 정보로 단순화
                # 위쪽 픽셀은 먼 곳, 아래쪽 픽셀은 가까운 곳을 본다고 가정
                v_ratio = (row + 0.5) / self.height
                target_distance = self.max_range - v_ratio * (self.max_range - self.min_range)

                temp = self._cast_ray(
                    temperature_map=temperature_map,
                    obstacle_map=obstacle_map,
                    cam_x=cam_x,
                    cam_y=cam_y,
                    ray_theta=ray_theta,
                    target_distance=target_distance,
                    map_resolution=map_resolution,
                )

                thermal_img[row, col] = self._apply_sensor_noise(temp)

        return thermal_img

    def _cast_ray(
        self,
        temperature_map,
        obstacle_map,
        cam_x,
        cam_y,
        ray_theta,
        target_distance,
        map_resolution,
    ):
        """
        하나의 픽셀에 해당하는 ray를 쏜다.
        벽/선반을 만나면 그 뒤는 보지 않는다.
        """

        ray_step = map_resolution * 0.5
        distances = np.arange(self.min_range, target_distance + ray_step, ray_step)

        sampled_temps = []

        for d in distances:
            x = cam_x + d * cos(ray_theta)
            y = cam_y + d * sin(ray_theta)

            ix, iy = self._world_to_grid(x, y, map_resolution)

            if not self._in_bounds(temperature_map, ix, iy):
                break

            temp = float(temperature_map[iy, ix])
            sampled_temps.append(temp)

            # 장애물을 만나면 그 뒤는 볼 수 없음
            if obstacle_map is not None:
                if self._in_bounds(obstacle_map, ix, iy) and obstacle_map[iy, ix]:
                    break

        if len(sampled_temps) == 0:
            return self.ambient_temp

        if self.measurement_mode == "last":
            return sampled_temps[-1]

        # 기본값: ray 경로 중 가장 뜨거운 지점
        # 화염/고온 공기가 시야 안에 들어오면 센서 이미지에 반영되도록 함
        return max(sampled_temps)

    def _world_to_grid(self, x, y, map_resolution):
        ix = int(x / map_resolution)
        iy = int(y / map_resolution)
        return ix, iy

    def _in_bounds(self, array, ix, iy):
        return 0 <= iy < array.shape[0] and 0 <= ix < array.shape[1]

    def _apply_sensor_noise(self, temp):
        """
        MLX90640의 오차를 단순 모델링.
        - 랜덤 노이즈
        - 절대 정확도 ±2 °C
        - 측정 범위 제한
        """

        noisy = temp + np.random.normal(0.0, self.noise_std)
        noisy += np.random.uniform(-self.accuracy_limit, self.accuracy_limit)

        return float(np.clip(noisy, self.temp_min, self.temp_max))
    def sense_3d(
        self,
        temperature_volume,
        robot_x,
        robot_y,
        robot_theta,
        xy_resolution=0.5,
        z_resolution=0.25,
        obstacle_volume=None,
        camera_pitch=0.0,
    ):
        """
        3D MLX90640 열화상 카메라 모델.

        temperature_volume:
            shape = (z, y, x)
            temperature_volume[iz, iy, ix] = temperature [C]

        robot_x, robot_y:
            로봇 중심 위치 [m]

        robot_theta:
            로봇 yaw 방향 [rad]
            0 rad    = +X 방향
            pi/2 rad = +Y 방향

        xy_resolution:
            x, y 방향 격자 크기 [m]

        z_resolution:
            z 방향 격자 크기 [m]

        obstacle_volume:
            shape = (z, y, x)
            True = 벽/선반/장애물
            False = 빈 공간

        camera_pitch:
            카메라 상하 기울기 [rad]
            0이면 수평
            양수면 위를 봄
            음수면 아래를 봄

        return:
            thermal_img, shape = (24, 32)
        """

        if not isinstance(temperature_volume, np.ndarray):
            temperature_volume = np.array(temperature_volume, dtype=np.float32)

        if obstacle_volume is not None and not isinstance(obstacle_volume, np.ndarray):
            obstacle_volume = np.array(obstacle_volume, dtype=bool)

        thermal_img = np.zeros((self.height, self.width), dtype=np.float32)

        # 로봇 중심이 아니라 전면 중앙에 카메라가 있다고 가정
        cam_x = robot_x + self.front_offset * cos(robot_theta)
        cam_y = robot_y + self.front_offset * sin(robot_theta)
        cam_z = self.camera_height

        for row in range(self.height):
            for col in range(self.width):

                # 가로 방향 각도
                h_ratio = (col + 0.5) / self.width

                # col=0은 화면 왼쪽 픽셀
                # 로봇 기준 왼쪽 방향을 보게 해야 하므로 +h_angle
                h_angle = (0.5 - h_ratio) * self.fov_h

                # 세로 방향 각도
                # row=0은 이미지 위쪽 → 위쪽 방향
                # row=height-1은 이미지 아래쪽 → 아래쪽 방향
                v_ratio = (row + 0.5) / self.height
                v_angle = (0.5 - v_ratio) * self.fov_v

                pitch = camera_pitch + v_angle
                yaw = robot_theta + h_angle

                # 3D ray 방향 벡터
                dir_x = cos(pitch) * cos(yaw)
                dir_y = cos(pitch) * sin(yaw)
                dir_z = sin(pitch)

                temp = self._cast_ray_3d(
                    temperature_volume=temperature_volume,
                    obstacle_volume=obstacle_volume,
                    cam_x=cam_x,
                    cam_y=cam_y,
                    cam_z=cam_z,
                    dir_x=dir_x,
                    dir_y=dir_y,
                    dir_z=dir_z,
                    xy_resolution=xy_resolution,
                    z_resolution=z_resolution,
                )

                thermal_img[row, col] = self._apply_sensor_noise(temp)

        return thermal_img

    def _cast_ray_3d(
        self,
        temperature_volume,
        obstacle_volume,
        cam_x,
        cam_y,
        cam_z,
        dir_x,
        dir_y,
        dir_z,
        xy_resolution,
        z_resolution,
    ):
        """
        3D 공간에서 하나의 ray를 쏜다.
        벽/선반을 만나면 그 뒤는 보지 않는다.
        """

        ray_step = min(xy_resolution, z_resolution) * 0.5
        distances = np.arange(self.min_range, self.max_range + ray_step, ray_step)

        sampled_temps = []

        for d in distances:
            x = cam_x + d * dir_x
            y = cam_y + d * dir_y
            z = cam_z + d * dir_z

            ix, iy, iz = self._world_to_grid_3d(
                x,
                y,
                z,
                xy_resolution,
                z_resolution,
            )

            if not self._in_bounds_3d(temperature_volume, ix, iy, iz):
                break

            temp = float(temperature_volume[iz, iy, ix])
            sampled_temps.append(temp)

            if obstacle_volume is not None:
                if self._in_bounds_3d(obstacle_volume, ix, iy, iz):
                    if obstacle_volume[iz, iy, ix]:
                        break

        if len(sampled_temps) == 0:
            return self.ambient_temp

        if self.measurement_mode == "last":
            return sampled_temps[-1]

        return max(sampled_temps)

    def _world_to_grid_3d(
        self,
        x,
        y,
        z,
        xy_resolution,
        z_resolution,
    ):
        ix = int(x / xy_resolution)
        iy = int(y / xy_resolution)
        iz = int(z / z_resolution)

        return ix, iy, iz

    def _in_bounds_3d(self, volume, ix, iy, iz):
        return (
            0 <= iz < volume.shape[0]
            and 0 <= iy < volume.shape[1]
            and 0 <= ix < volume.shape[2]
        )