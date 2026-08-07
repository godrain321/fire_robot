"""Shared ROS, PGM and FDS coordinate transformations."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MapTransform:
    """Coordinate transform for an unrotated ROS occupancy-grid map."""

    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.resolution <= 0.0:
            raise ValueError("map dimensions and resolution must be positive")
        if not math.isclose(self.origin_yaw, 0.0, abs_tol=1e-12):
            raise ValueError("rotated occupancy-grid origins are not supported")

    @property
    def width_m(self) -> float:
        return self.width * self.resolution

    @property
    def height_m(self) -> float:
        return self.height * self.resolution

    def ros_to_fds(self, x_ros: float, y_ros: float) -> tuple[float, float]:
        return x_ros - self.origin_x, y_ros - self.origin_y

    def fds_to_ros(self, x_fds: float, y_fds: float) -> tuple[float, float]:
        return x_fds + self.origin_x, y_fds + self.origin_y

    def pixel_center_to_ros(self, row: int, col: int) -> tuple[float, float]:
        self._check_pixel(row, col)
        x_ros = self.origin_x + (col + 0.5) * self.resolution
        y_ros = self.origin_y + (self.height - row - 0.5) * self.resolution
        return x_ros, y_ros

    def ros_to_pixel(self, x_ros: float, y_ros: float) -> tuple[int, int]:
        col = math.floor((x_ros - self.origin_x) / self.resolution)
        y_from_bottom = math.floor((y_ros - self.origin_y) / self.resolution)
        row = self.height - 1 - y_from_bottom
        self._check_pixel(row, col)
        return row, col

    def fds_to_grid(
        self, x_fds: float, y_fds: float, grid_resolution: float
    ) -> tuple[int, int]:
        if grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be positive")
        return math.floor(x_fds / grid_resolution), math.floor(
            y_fds / grid_resolution
        )

    def _check_pixel(self, row: int, col: int) -> None:
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"pixel ({row}, {col}) is outside the map")

