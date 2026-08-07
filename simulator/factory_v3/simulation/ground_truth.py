"""FDS Ground Truth facade; only simulated sensors and evaluation query it."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from mapping.fire_costmap import _axis_coordinates, load_factory_geometry


@dataclass(frozen=True)
class GasObservation:
    """One localized gas-sensor result."""

    valid: bool
    reading: Any | None
    selected_time: float
    world_position: tuple[float, float]
    reason: str = ""


@dataclass(frozen=True)
class GroundTruthExposure:
    """Ground Truth point sample used only for post-run evaluation."""

    temperature: float
    co_ppm: float
    temperature_time: float
    co_time: float


class FDSGroundTruthEnvironment:
    """Own complete FDS arrays without exposing them to mapping or planning."""

    def __init__(
        self,
        fds_path: str | Path,
        temperature_npz_path: str | Path,
        fds_result_dir: str | Path,
    ) -> None:
        self.mesh_xb, self.obstacles, self.holes = load_factory_geometry(fds_path)
        self._load_temperature(temperature_npz_path)
        self._load_co(fds_result_dir)
        self._obstacle_volume_zyx = self._build_obstacle_volume()

    @property
    def temperature_time_range(self) -> tuple[float, float]:
        return float(self._temperature_times[0]), float(self._temperature_times[-1])

    @property
    def co_time_range(self) -> tuple[float, float]:
        return float(self._co_times[0]), float(self._co_times[-1])

    @property
    def xy_resolution(self) -> float:
        return self._xy_resolution

    @property
    def z_resolution(self) -> float:
        return self._z_resolution

    def _load_temperature(self, npz_path: str | Path) -> None:
        with np.load(npz_path) as data:
            raw = np.asarray(data["temperature"], dtype=float)
            self._temperature_times = np.asarray(data["times"], dtype=float)
            self._xy_resolution = float(data["xy_resolution"])
            self._z_resolution = float(data["z_resolution"])
        if raw.ndim != 4 or raw.shape[0] != self._temperature_times.size:
            raise ValueError(f"Invalid temperature timeseries shape: {raw.shape}")

        extents = (
            self.mesh_xb[1] - self.mesh_xb[0],
            self.mesh_xb[3] - self.mesh_xb[2],
            self.mesh_xb[5] - self.mesh_xb[4],
        )
        resolutions = (self._xy_resolution, self._xy_resolution, self._z_resolution)
        expected = tuple(int(round(v / r)) for v, r in zip(extents, resolutions))
        matches = []
        for permutation in permutations(range(3)):
            reordered = tuple(raw.shape[1:][index] for index in permutation)
            if all(n in {cells, cells + 1} for n, cells in zip(reordered, expected)):
                matches.append(permutation)
        if len(matches) != 1:
            raise ValueError(f"Ambiguous temperature axes: {raw.shape}, {matches}")
        permutation = matches[0]
        self._ground_truth_temperature_xyz = np.transpose(
            raw, (0, 1 + permutation[0], 1 + permutation[1], 1 + permutation[2])
        )
        self._temperature_x = _axis_coordinates(
            self._ground_truth_temperature_xyz.shape[1],
            self.mesh_xb[0], self.mesh_xb[1], self._xy_resolution,
        )
        self._temperature_y = _axis_coordinates(
            self._ground_truth_temperature_xyz.shape[2],
            self.mesh_xb[2], self.mesh_xb[3], self._xy_resolution,
        )
        self._temperature_z = _axis_coordinates(
            self._ground_truth_temperature_xyz.shape[3],
            self.mesh_xb[4], self.mesh_xb[5], self._z_resolution,
        )

    def _load_co(self, fds_result_dir: str | Path) -> None:
        try:
            import fdsreader as fds
        except ImportError as exc:
            raise RuntimeError("fdsreader is required for FDS CO Ground Truth") from exc
        simulation = fds.Simulation(str(fds_result_dir))
        slices = [slc for slc in simulation.slices if "CO_Z130" in str(slc).upper()]
        if not slices:
            slices = [
                slc for slc in simulation.slices
                if "CARBON MONOXIDE" in str(slc).upper()
            ]
        if not slices:
            raise ValueError("Carbon-monoxide FDS slice not found")
        co_slice = slices[0]
        values, coords = co_slice.to_global(
            masked=True, fill=np.nan, return_coordinates=True
        )
        values = np.asarray(values, dtype=float)
        self._co_times = np.asarray(co_slice.times, dtype=float)
        self._co_x = np.asarray(coords["x"], dtype=float)
        self._co_y = np.asarray(coords["y"], dtype=float)
        self.co_height = float(np.asarray(coords.get("z", [np.nan]))[0])
        if values.shape[1:] == (self._co_x.size, self._co_y.size):
            self._ground_truth_co_yx = np.transpose(values, (0, 2, 1))
        elif values.shape[1:] == (self._co_y.size, self._co_x.size):
            self._ground_truth_co_yx = values
        else:
            raise ValueError(f"CO coordinate/data mismatch: {values.shape}")
        # factory_v1.fds and Smokeview identify this as mol/mol volume fraction.
        self._ground_truth_co_yx *= 1_000_000.0

    def _build_obstacle_volume(self) -> np.ndarray:
        shape = (
            self._temperature_z.size,
            self._temperature_y.size,
            self._temperature_x.size,
        )
        volume = np.zeros(shape, dtype=bool)

        def fill(xb, value):
            x1, x2, y1, y2, z1, z2 = xb
            xi = np.where(
                (self._temperature_x >= min(x1, x2))
                & (self._temperature_x <= max(x1, x2))
            )[0]
            yi = np.where(
                (self._temperature_y >= min(y1, y2))
                & (self._temperature_y <= max(y1, y2))
            )[0]
            zi = np.where(
                (self._temperature_z >= min(z1, z2))
                & (self._temperature_z <= max(z1, z2))
            )[0]
            if xi.size and yi.size and zi.size:
                volume[np.ix_(zi, yi, xi)] = value

        for obstacle in self.obstacles:
            fill(obstacle["xb"], True)
        for hole in self.holes:
            fill(hole["xb"], False)
        return volume

    @staticmethod
    def _nearest_index(axis: np.ndarray, value: float) -> int:
        return int(np.abs(axis - value).argmin())

    def capture_thermal(self, camera, robot_state, fds_time: float):
        """Invoke the camera internally so the full volume never reaches mapping."""
        time_index = self._nearest_index(self._temperature_times, fds_time)
        volume_zyx = np.transpose(
            self._ground_truth_temperature_xyz[time_index], (2, 1, 0)
        )
        image, rays = camera.sense_3d(
            temperature_volume=volume_zyx,
            robot_x=robot_state.x,
            robot_y=robot_state.y,
            robot_theta=robot_state.theta,
            xy_resolution=self._xy_resolution,
            z_resolution=self._z_resolution,
            obstacle_volume=self._obstacle_volume_zyx,
            camera_pitch=0.0,
            return_observations=True,
            origin=(self.mesh_xb[0], self.mesh_xb[2], self.mesh_xb[4]),
        )
        return image, rays, float(self._temperature_times[time_index])

    def measure_co(self, sensor, robot_state, fds_time: float, dt: float) -> GasObservation:
        """Sample one Ground Truth cell and return only the simulated reading."""
        time_index = self._nearest_index(self._co_times, fds_time)
        ix = self._nearest_index(self._co_x, robot_state.x)
        iy = self._nearest_index(self._co_y, robot_state.y)
        true_ppm = float(self._ground_truth_co_yx[time_index, iy, ix])
        selected_time = float(self._co_times[time_index])
        if not np.isfinite(true_ppm):
            return GasObservation(
                False, None, selected_time, (robot_state.x, robot_state.y),
                "FDS CO cell is invalid; belief remains unknown",
            )
        reading = sensor.read(true_ppm=true_ppm, dt=dt)
        return GasObservation(
            True, reading, selected_time, (robot_state.x, robot_state.y)
        )

    def evaluate_exposure(
        self, robot_x: float, robot_y: float, robot_height: float, fds_time: float
    ) -> GroundTruthExposure:
        """Point-sample Ground Truth for metrics only; never call from planning."""
        ti = self._nearest_index(self._temperature_times, fds_time)
        tx = self._nearest_index(self._temperature_x, robot_x)
        ty = self._nearest_index(self._temperature_y, robot_y)
        tz = self._nearest_index(self._temperature_z, robot_height)
        ci = self._nearest_index(self._co_times, fds_time)
        cx = self._nearest_index(self._co_x, robot_x)
        cy = self._nearest_index(self._co_y, robot_y)
        return GroundTruthExposure(
            temperature=float(self._ground_truth_temperature_xyz[ti, tx, ty, tz]),
            co_ppm=float(self._ground_truth_co_yx[ci, cy, cx]),
            temperature_time=float(self._temperature_times[ti]),
            co_time=float(self._co_times[ci]),
        )

    def sample_map_yx(
        self, x_world: np.ndarray, y_world: np.ndarray,
        height: float, fds_time: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return an evaluation-only FDS plane in ``[y,x]`` planner order.

        The caller must not pass these complete Ground Truth arrays to mapping
        or planning. Virtual sensors continue to query this facade directly.
        """
        targets_x = np.asarray(x_world, dtype=float)
        targets_y = np.asarray(y_world, dtype=float)
        if targets_x.ndim != 1 or targets_y.ndim != 1:
            raise ValueError("sample_map_yx coordinates must be 1-D")
        ti = self._nearest_index(self._temperature_times, fds_time)
        ci = self._nearest_index(self._co_times, fds_time)
        tz = self._nearest_index(self._temperature_z, height)
        tx = np.abs(self._temperature_x[:, None] - targets_x[None, :]).argmin(axis=0)
        ty = np.abs(self._temperature_y[:, None] - targets_y[None, :]).argmin(axis=0)
        cx = np.abs(self._co_x[:, None] - targets_x[None, :]).argmin(axis=0)
        cy = np.abs(self._co_y[:, None] - targets_y[None, :]).argmin(axis=0)
        temperature_xy = self._ground_truth_temperature_xyz[ti, :, :, tz]
        temperature_yx = temperature_xy[np.ix_(tx, ty)].T.copy()
        co_yx = self._ground_truth_co_yx[ci][np.ix_(cy, cx)].copy()
        return temperature_yx, co_yx, float(self._temperature_times[ti])
