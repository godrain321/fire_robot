"""Fast z-slice thermal sensor backend for the optional 2-D runner.

This module is simulator-only. It exposes only forward-FOV observations to the
robot; the complete FDS slice never enters mapping or planning.
"""

from __future__ import annotations

import math

import numpy as np

from sensors.thermal_camera import ThermalRayObservation, ThermalRaySample
from simulation.ground_truth import FDSGroundTruthEnvironment


class FDSGroundTruthEnvironment2D(FDSGroundTruthEnvironment):
    """Replace 24x32 3-D rays with 32 horizontal rays on one z slice."""

    observation_height_m = 0.30

    def __init__(self, *args, observation_height_m: float = 0.30, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(float(observation_height_m)):
            raise ValueError("observation_height_m must be finite")
        if not self.mesh_xb[4] <= observation_height_m <= self.mesh_xb[5]:
            raise ValueError("observation height is outside the FDS mesh")
        self.observation_height_m = float(observation_height_m)
        self._slice_z_index = self._nearest_index(
            self._temperature_z, self.observation_height_m
        )
        self._obstacle_slice_yx = self._obstacle_volume_zyx[
            self._slice_z_index
        ].copy()
        self._cached_time_index: int | None = None
        self._cached_temperature_slice_yx: np.ndarray | None = None
        self.slice_cache_build_count = 0

    def _temperature_slice(self, fds_time: float) -> tuple[np.ndarray, int]:
        time_index = self._nearest_index(self._temperature_times, fds_time)
        if time_index != self._cached_time_index:
            # Stored volume is [time,x,y,z]; expose this private sensor cache as
            # [row(y),col(x)] only inside the Ground Truth facade.
            self._cached_temperature_slice_yx = np.asarray(
                self._ground_truth_temperature_xyz[
                    time_index, :, :, self._slice_z_index
                ].T,
                dtype=float,
            )
            self._cached_time_index = time_index
            self.slice_cache_build_count += 1
        return self._cached_temperature_slice_yx, time_index

    def capture_thermal(self, camera, robot_state, fds_time: float):
        """Return forward-visible temperatures using horizontal 2-D rays.

        The returned image is retained only for compatibility with status and
        fire-localization code. Every row contains the same horizontal scan;
        no vertical or 3-D ray casting is performed.
        """
        temperature_yx, time_index = self._temperature_slice(fds_time)
        ray_count = int(camera.width)
        image = np.full((int(camera.height), ray_count), camera.ambient_temp)
        observations = []
        cam_x = robot_state.x + camera.front_offset * math.cos(robot_state.theta)
        cam_y = robot_state.y + camera.front_offset * math.sin(robot_state.theta)
        step = self._xy_resolution * 0.5

        for col in range(ray_count):
            ratio = (col + 0.5) / ray_count
            yaw = robot_state.theta + (0.5 - ratio) * camera.fov_h
            samples = []
            occluded = False
            for distance in np.arange(camera.min_range, camera.max_range + step, step):
                x = cam_x + float(distance) * math.cos(yaw)
                y = cam_y + float(distance) * math.sin(yaw)
                ix = self._nearest_index(self._temperature_x, x)
                iy = self._nearest_index(self._temperature_y, y)
                if not (
                    self.mesh_xb[0] <= x <= self.mesh_xb[1]
                    and self.mesh_xb[2] <= y <= self.mesh_xb[3]
                ):
                    break
                temperature = float(temperature_yx[iy, ix])
                sample = ThermalRaySample(
                    grid_position=(ix, iy, self._slice_z_index),
                    world_position=(x, y, self.observation_height_m),
                    distance=float(distance), temperature=temperature,
                )
                samples.append(sample)
                if self._obstacle_slice_yx[iy, ix]:
                    occluded = True
                    break
            finite_indices = [
                index for index, sample in enumerate(samples)
                if math.isfinite(sample.temperature)
            ]
            if not finite_indices:
                observations.append(ThermalRayObservation(
                    0, col, camera.ambient_temp, tuple(samples), None, None,
                    None, False, occluded,
                ))
                continue
            if camera.measurement_mode == "last":
                selected_index = finite_indices[-1]
            else:
                selected_index = max(
                    finite_indices, key=lambda index: samples[index].temperature
                )
            selected = samples[selected_index]
            measured = float(camera._apply_sensor_noise(selected.temperature))
            measured_samples = tuple(
                ThermalRaySample(
                    sample.grid_position, sample.world_position, sample.distance,
                    measured if index == selected_index else sample.temperature,
                )
                for index, sample in enumerate(samples)
            )
            image[:, col] = measured
            observations.append(ThermalRayObservation(
                0, col, measured, measured_samples,
                selected.world_position, selected.grid_position,
                selected.distance, True, occluded,
            ))
        selected_time = float(self._temperature_times[time_index])
        return image, tuple(observations), selected_time
