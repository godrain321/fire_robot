# human_detection_sim.py

import math
from typing import Dict, List, Tuple, Optional

import numpy as np


class SimpleHumanDetector:
    """
    단순 요구조자 인식 모델

    가정:
    - 사람 인식 알고리즘은 별도 구현 대상
    - 시뮬레이션에서는 로봇과 사람 사이 거리가 detection_range 이내이고,
      중간에 벽/장애물이 없으면 사람을 인식했다고 판단
    """

    def __init__(
        self,
        detection_range: float = 10.0,
        horizontal_fov_deg: float = 100.0,
    ):
        self.detection_range = float(detection_range)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        if not math.isfinite(self.detection_range) or self.detection_range <= 0.0:
            raise ValueError("detection_range must be finite and positive")
        if (
            not math.isfinite(self.horizontal_fov_deg)
            or not 0.0 < self.horizontal_fov_deg <= 360.0
        ):
            raise ValueError("horizontal_fov_deg must be in (0, 360]")

    def detect(
        self,
        robot_position: Tuple[float, float],
        humans: List[Dict],
        robot_heading_rad: float = 0.0,
        obstacle_map: Optional[np.ndarray] = None,
        map_origin: Tuple[float, float] = (0.0, 0.0),
        map_resolution: float = 0.1,
        use_line_of_sight: bool = True,
    ) -> List[Dict]:
        """
        robot_position:
            (robot_x, robot_y)

        humans:
            [
                {"id": "victim_1", "x": 4.0, "y": 2.0},
                {"id": "victim_2", "x": 8.0, "y": 5.0},
            ]

        obstacle_map:
            obstacle_map[y, x] = True 이면 벽/장애물

        return:
            인식된 사람 리스트
        """

        robot_x, robot_y = robot_position
        detected_humans = []

        for human in humans:
            human_id = human.get("id", "unknown")
            human_x = float(human["x"])
            human_y = float(human["y"])

            dx = human_x - robot_x
            dy = human_y - robot_y
            distance = math.hypot(dx, dy)

            # 1. 거리 조건
            if distance > self.detection_range:
                continue

            # 2. 로봇 전방 수평 시야각 조건. atan2(sin, cos)로 각도
            # 차이를 [-pi, pi]에 정규화하여 +/-pi 경계도 처리한다.
            if distance > 1e-9 and self.horizontal_fov_deg < 360.0:
                bearing = math.atan2(dy, dx)
                angle_delta = math.atan2(
                    math.sin(bearing - robot_heading_rad),
                    math.cos(bearing - robot_heading_rad),
                )
                half_fov = math.radians(self.horizontal_fov_deg) / 2.0
                if abs(angle_delta) > half_fov + 1e-12:
                    continue

            # 3. 벽/장애물 가림 조건
            blocked = False

            if use_line_of_sight and obstacle_map is not None:
                blocked = self._is_line_blocked(
                    start=(robot_x, robot_y),
                    end=(human_x, human_y),
                    obstacle_map=obstacle_map,
                    map_origin=map_origin,
                    map_resolution=map_resolution,
                )

            if blocked:
                continue

            detected_humans.append(
                {
                    "id": human_id,
                    "detected": True,
                    "x": human_x,
                    "y": human_y,
                    "distance": distance,
                }
            )

        detected_humans.sort(key=lambda h: h["distance"])
        return detected_humans

    def _is_line_blocked(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        obstacle_map: np.ndarray,
        map_origin: Tuple[float, float],
        map_resolution: float,
    ) -> bool:
        """
        로봇과 사람 사이 직선 경로에 장애물이 있는지 확인
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

            if obstacle_map[grid_y, grid_x]:
                return True

        return False
