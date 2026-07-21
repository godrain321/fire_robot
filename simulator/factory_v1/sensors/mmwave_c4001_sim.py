# mmwave_c4001_sim.py

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def normalize_angle(angle: float) -> float:
    """
    각도를 -pi ~ pi 범위로 정규화한다.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class C4001Detection:
    """
    C4001 mmWave 센서의 가상 탐지 결과
    """
    human_id: str
    detected: bool

    detection_type: str
    # "presence"       : 8 m 이내 사람 존재 감지
    # "motion_range"   : 12 m 이내 움직이는 사람 거리/속도 감지
    # "false_positive" : 오탐지

    x: Optional[float]
    y: Optional[float]
    z: Optional[float]

    distance: Optional[float]
    bearing: Optional[float]
    elevation: Optional[float]
    radial_speed: Optional[float]

    distance_measured: bool
    speed_measured: bool
    confidence: float


class VirtualC4001MmWaveSensor:
    """
    DFROBOT Gravity C4001 SEN0610 기반 단순화 mmWave 가상 센서 모델.

    실제 사양 반영:
    - 24 GHz
    - FMCW
    - 수평 감지각 100도
    - 수직 감지각 80도
    - 사람 존재 감지 최대 8 m
    - 동작 감지 및 거리 측정 최대 12 m
    - 거리 측정 가능 범위 1.2 m ~ 12 m
    - 속도 측정 가능 범위 0.1 m/s ~ 10 m/s

    단순화 가정:
    - 실제 range-Doppler FFT, point cloud, 다중경로 반사는 구현하지 않음
    - 사람인지 판단은 이미 된다고 가정
    - 시뮬레이션에서는 사람 위치와 속도 정보를 받아 센서 출력 형태로 변환
    - 벽/장애물 가림은 obstacle_map으로 선택 적용
    """

    def __init__(
        self,
        presence_range: float = 8.0,
        motion_range: float = 12.0,
        distance_measure_min: float = 1.2,
        distance_measure_max: float = 12.0,
        speed_measure_min: float = 0.1,
        speed_measure_max: float = 10.0,
        horizontal_fov_deg: float = 100.0,
        vertical_fov_deg: float = 80.0,
        sensor_height: float = 0.35,
        base_confidence: float = 0.95,
        distance_noise_std: float = 0.08,
        bearing_noise_std_deg: float = 2.0,
        elevation_noise_std_deg: float = 2.0,
        speed_noise_std: float = 0.05,
        position_noise_std: float = 0.10,
        presence_miss_probability: float = 0.03,
        motion_miss_probability: float = 0.05,
        false_positive_probability: float = 0.00,
        random_seed: Optional[int] = None,
    ):
        # 실제 사양성 파라미터
        self.frequency_hz = 24e9
        self.modulation = "FMCW"

        self.presence_range = presence_range
        self.motion_range = motion_range

        self.distance_measure_min = distance_measure_min
        self.distance_measure_max = distance_measure_max

        self.speed_measure_min = speed_measure_min
        self.speed_measure_max = speed_measure_max

        self.horizontal_fov = math.radians(horizontal_fov_deg)
        self.vertical_fov = math.radians(vertical_fov_deg)

        self.sensor_height = sensor_height

        # 센서 오차/노이즈 모델
        self.base_confidence = base_confidence
        self.distance_noise_std = distance_noise_std
        self.bearing_noise_std = math.radians(bearing_noise_std_deg)
        self.elevation_noise_std = math.radians(elevation_noise_std_deg)
        self.speed_noise_std = speed_noise_std
        self.position_noise_std = position_noise_std

        self.presence_miss_probability = presence_miss_probability
        self.motion_miss_probability = motion_miss_probability
        self.false_positive_probability = false_positive_probability

        self.rng = random.Random(random_seed)
        self.np_rng = np.random.default_rng(random_seed)

    def detect(
        self,
        robot_pose: Tuple[float, float, float],
        humans: List[Dict],
        robot_velocity: Tuple[float, float] = (0.0, 0.0),
        obstacle_map: Optional[np.ndarray] = None,
        map_origin: Tuple[float, float] = (0.0, 0.0),
        map_resolution: float = 0.1,
        use_occlusion: bool = True,
    ) -> List[Dict]:
        """
        robot_pose:
            (robot_x, robot_y, robot_yaw)
            yaw 단위는 radian.
            yaw = 0 이면 +x 방향을 바라보는 상태.

        humans:
            [
                {
                    "id": "victim_1",
                    "x": 4.0,
                    "y": 2.0,
                    "z": 0.8,       # 선택. 없으면 0.8 m로 가정
                    "vx": 0.0,      # 선택. m/s
                    "vy": 0.0,      # 선택. m/s
                }
            ]

        robot_velocity:
            (robot_vx, robot_vy), m/s.
            로봇이 움직이고 있을 때 상대속도 계산에 사용.

        obstacle_map:
            2D numpy array.
            obstacle_map[y, x] = True 이면 장애물.
            벽/금속/큰 가구처럼 mmWave를 막는 물체로 간주.

        return:
            탐지 결과 리스트.
            가까운 대상부터 정렬해서 반환.
        """

        robot_x, robot_y, robot_yaw = robot_pose
        robot_vx, robot_vy = robot_velocity

        detections: List[C4001Detection] = []

        for human in humans:
            human_id = str(human.get("id", "unknown"))

            human_x = float(human["x"])
            human_y = float(human["y"])
            human_z = float(human.get("z", 0.8))

            human_vx = float(human.get("vx", 0.0))
            human_vy = float(human.get("vy", 0.0))

            dx = human_x - robot_x
            dy = human_y - robot_y
            dz = human_z - self.sensor_height

            horizontal_distance = math.hypot(dx, dy)
            true_distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            if true_distance < 1e-6:
                continue

            global_angle = math.atan2(dy, dx)
            bearing = normalize_angle(global_angle - robot_yaw)
            elevation = math.atan2(dz, max(horizontal_distance, 1e-6))

            # 1. 수평 FOV 확인: C4001 수평 약 100도
            if abs(bearing) > self.horizontal_fov / 2.0:
                continue

            # 2. 수직 FOV 확인: C4001 수직 약 80도
            if abs(elevation) > self.vertical_fov / 2.0:
                continue

            # 3. 장애물 가림 확인
            if use_occlusion and obstacle_map is not None:
                blocked = self._is_line_blocked(
                    start=(robot_x, robot_y),
                    end=(human_x, human_y),
                    obstacle_map=obstacle_map,
                    map_origin=map_origin,
                    map_resolution=map_resolution,
                )

                if blocked:
                    continue

            # 4. 상대 방사속도 계산
            # FMCW/Doppler 관점에서는 센서와 대상 사이 방향의 속도 성분이 중요하다.
            rel_vx = human_vx - robot_vx
            rel_vy = human_vy - robot_vy

            radial_speed = (dx * rel_vx + dy * rel_vy) / max(horizontal_distance, 1e-6)
            radial_speed_abs = abs(radial_speed)

            # 5. 실제 C4001 사양에 맞춰 탐지 타입 결정
            in_presence_range = true_distance <= self.presence_range
            in_motion_range = true_distance <= self.motion_range

            speed_measurable = (
                self.speed_measure_min <= radial_speed_abs <= self.speed_measure_max
            )

            distance_measurable = (
                self.distance_measure_min <= true_distance <= self.distance_measure_max
            )

            detection_type: Optional[str] = None

            # 5-1. 움직이는 사람: 12 m까지 거리/속도 감지 가능
            if in_motion_range and speed_measurable:
                if self.rng.random() >= self.motion_miss_probability:
                    detection_type = "motion_range"

            # 5-2. 정지 또는 느리게 움직이는 사람: 8 m까지 존재 감지
            if detection_type is None and in_presence_range:
                if self.rng.random() >= self.presence_miss_probability:
                    detection_type = "presence"

            if detection_type is None:
                continue

            # 6. confidence 계산
            confidence = self._compute_confidence(
                distance=true_distance,
                bearing=bearing,
                elevation=elevation,
                radial_speed_abs=radial_speed_abs,
                detection_type=detection_type,
            )

            # 7. 측정값 생성
            measured_distance: Optional[float] = None
            measured_bearing: Optional[float] = None
            measured_elevation: Optional[float] = None
            measured_speed: Optional[float] = None
            measured_x: Optional[float] = None
            measured_y: Optional[float] = None
            measured_z: Optional[float] = None

            # C4001의 거리 측정 가능 범위는 1.2 m ~ 12 m
            if distance_measurable:
                measured_distance = true_distance + float(
                    self.np_rng.normal(0.0, self.distance_noise_std)
                )
                measured_distance = max(0.0, measured_distance)

                measured_bearing = bearing + float(
                    self.np_rng.normal(0.0, self.bearing_noise_std)
                )

                measured_elevation = elevation + float(
                    self.np_rng.normal(0.0, self.elevation_noise_std)
                )

                measured_global_angle = robot_yaw + measured_bearing

                measured_horizontal_distance = measured_distance * math.cos(
                    measured_elevation
                )

                measured_x = robot_x + measured_horizontal_distance * math.cos(
                    measured_global_angle
                )
                measured_y = robot_y + measured_horizontal_distance * math.sin(
                    measured_global_angle
                )
                measured_z = self.sensor_height + measured_distance * math.sin(
                    measured_elevation
                )

                measured_x += float(self.np_rng.normal(0.0, self.position_noise_std))
                measured_y += float(self.np_rng.normal(0.0, self.position_noise_std))
                measured_z += float(self.np_rng.normal(0.0, self.position_noise_std))

            # C4001의 속도 측정 가능 범위는 0.1 m/s ~ 10 m/s
            if speed_measurable:
                measured_speed = radial_speed + float(
                    self.np_rng.normal(0.0, self.speed_noise_std)
                )

            detections.append(
                C4001Detection(
                    human_id=human_id,
                    detected=True,
                    detection_type=detection_type,
                    x=measured_x,
                    y=measured_y,
                    z=measured_z,
                    distance=measured_distance,
                    bearing=measured_bearing,
                    elevation=measured_elevation,
                    radial_speed=measured_speed,
                    distance_measured=distance_measurable,
                    speed_measured=speed_measurable,
                    confidence=confidence,
                )
            )

        # 8. 오탐지 옵션
        if self.false_positive_probability > 0.0:
            if self.rng.random() < self.false_positive_probability:
                detections.append(self._generate_false_positive(robot_pose))

        detections.sort(
            key=lambda d: float("inf") if d.distance is None else d.distance
        )

        return [d.__dict__ for d in detections]

    def _compute_confidence(
        self,
        distance: float,
        bearing: float,
        elevation: float,
        radial_speed_abs: float,
        detection_type: str,
    ) -> float:
        """
        거리, 각도, 속도를 이용한 단순 confidence 모델.
        """

        if detection_type == "motion_range":
            max_range = self.motion_range
        else:
            max_range = self.presence_range

        # 거리가 멀수록 confidence 감소
        distance_factor = 1.0 - 0.45 * (distance / max_range)
        distance_factor = np.clip(distance_factor, 0.25, 1.0)

        # 수평 중심에서 벗어날수록 감소
        bearing_factor = 1.0 - 0.30 * (abs(bearing) / (self.horizontal_fov / 2.0))
        bearing_factor = np.clip(bearing_factor, 0.40, 1.0)

        # 수직 중심에서 벗어날수록 감소
        elevation_factor = 1.0 - 0.20 * (abs(elevation) / (self.vertical_fov / 2.0))
        elevation_factor = np.clip(elevation_factor, 0.50, 1.0)

        # motion_range에서는 속도가 너무 작거나 너무 크면 confidence 감소
        speed_factor = 1.0

        if detection_type == "motion_range":
            if radial_speed_abs < self.speed_measure_min:
                speed_factor = 0.5
            elif radial_speed_abs > self.speed_measure_max:
                speed_factor = 0.5
            else:
                speed_factor = 1.0

        confidence = (
            self.base_confidence
            * distance_factor
            * bearing_factor
            * elevation_factor
            * speed_factor
        )

        return float(np.clip(confidence, 0.05, 1.0))

    def _is_line_blocked(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        obstacle_map: np.ndarray,
        map_origin: Tuple[float, float],
        map_resolution: float,
    ) -> bool:
        """
        로봇과 사람 사이에 장애물이 있는지 검사한다.

        obstacle_map[y, x] = True 이면 mmWave를 막는 장애물로 간주한다.
        실제 mmWave는 얇은 플라스틱/천/유리 등은 일부 통과할 수 있지만,
        시뮬레이션에서는 벽, 선반, 금속 장애물은 차폐된다고 단순화한다.
        """

        x0, y0 = start
        x1, y1 = end

        distance = math.hypot(x1 - x0, y1 - y0)

        if distance < 1e-6:
            return False

        num_samples = max(2, int(distance / (map_resolution * 0.5)))

        for i in range(1, num_samples):
            t = i / num_samples

            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)

            grid_x = int((x - map_origin[0]) / map_resolution)
            grid_y = int((y - map_origin[1]) / map_resolution)

            if grid_y < 0 or grid_y >= obstacle_map.shape[0]:
                continue

            if grid_x < 0 or grid_x >= obstacle_map.shape[1]:
                continue

            if bool(obstacle_map[grid_y, grid_x]):
                return True

        return False

    def _generate_false_positive(
        self,
        robot_pose: Tuple[float, float, float],
    ) -> C4001Detection:
        """
        오탐지를 생성한다.
        기본 설정에서는 false_positive_probability = 0 이므로 사용되지 않는다.
        """

        robot_x, robot_y, robot_yaw = robot_pose

        distance = self.rng.uniform(
            self.distance_measure_min,
            self.distance_measure_max,
        )

        bearing = self.rng.uniform(
            -self.horizontal_fov / 2.0,
            self.horizontal_fov / 2.0,
        )

        elevation = self.rng.uniform(
            -self.vertical_fov / 2.0,
            self.vertical_fov / 2.0,
        )

        global_angle = robot_yaw + bearing

        horizontal_distance = distance * math.cos(elevation)

        x = robot_x + horizontal_distance * math.cos(global_angle)
        y = robot_y + horizontal_distance * math.sin(global_angle)
        z = self.sensor_height + distance * math.sin(elevation)

        radial_speed = self.rng.uniform(
            self.speed_measure_min,
            self.speed_measure_max,
        )

        return C4001Detection(
            human_id="false_positive",
            detected=True,
            detection_type="false_positive",
            x=x,
            y=y,
            z=z,
            distance=distance,
            bearing=bearing,
            elevation=elevation,
            radial_speed=radial_speed,
            distance_measured=True,
            speed_measured=True,
            confidence=0.20,
        )