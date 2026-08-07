"""Build a static fire-risk costmap from complete FDS temperature and CO data."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import re
from typing import Any

import numpy as np

from mapping.grid_map import GridMap


@dataclass(frozen=True)
class FireCostmapConfig:
    """Configuration for the ideal, full-information static costmap."""

    fds_time: float = 60.0
    robot_height: float = 1.3
    grid_resolution: float = 0.5
    base_cost: float = 1.0
    temperature_safe: float = 20.0
    temperature_blocked: float = 60.0
    temperature_weight: float = 8.0
    temperature_power: float = 2.0
    co_safe: float = 0.0
    co_blocked: float = 1600.0
    co_weight: float = 8.0
    co_power: float = 2.0
    use_inflation: bool = True
    inflation_radius: float = 0.45

    def __post_init__(self) -> None:
        if self.grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be positive")
        if self.base_cost <= 0.0:
            raise ValueError("base_cost must be positive")
        if self.temperature_blocked <= self.temperature_safe:
            raise ValueError("temperature_blocked must exceed temperature_safe")
        if self.co_blocked <= self.co_safe:
            raise ValueError("co_blocked must exceed co_safe")
        if self.temperature_weight < 0.0 or self.co_weight < 0.0:
            raise ValueError("risk weights must be non-negative")


@dataclass(frozen=True)
class FireCostmapLayers:
    """All costmap layers, stored in NumPy ``array[gy, gx]`` order."""

    grid_map: GridMap
    static_obstacle_map: np.ndarray
    temperature_map: np.ndarray
    co_map: np.ndarray
    temperature_cost_map: np.ndarray
    co_cost_map: np.ndarray
    blocked_mask: np.ndarray
    final_cost_map: np.ndarray
    selected_temperature_time: float
    selected_co_time: float
    selected_temperature_height: float
    selected_co_height: float

    def validate(self) -> None:
        expected = (self.grid_map.height, self.grid_map.width)
        arrays = {
            "static_obstacle_map": self.static_obstacle_map,
            "temperature_map": self.temperature_map,
            "co_map": self.co_map,
            "temperature_cost_map": self.temperature_cost_map,
            "co_cost_map": self.co_cost_map,
            "blocked_mask": self.blocked_mask,
            "final_cost_map": self.final_cost_map,
        }
        mismatches = {
            name: value.shape for name, value in arrays.items()
            if value.shape != expected
        }
        if mismatches:
            raise ValueError(
                f"Costmap layer shape mismatch: expected={expected}, "
                f"actual={mismatches}"
            )


def _read_fds_namelists(fds_path: Path) -> list[tuple[str, str]]:
    text = fds_path.read_text(encoding="utf-8")
    cleaned = []
    for line in text.splitlines():
        uncommented = line.split("!", maxsplit=1)[0]
        if uncommented.strip():
            cleaned.append(uncommented)
    blocks = re.findall(
        r"&([A-Z0-9_]+)\s+(.*?)/", "\n".join(cleaned), flags=re.I | re.S
    )
    return [
        (name.upper(), " ".join(body.replace("\n", " ").split()))
        for name, body in blocks
    ]


def _parse_numbers(body: str, key: str, count: int) -> list[float] | None:
    match = re.search(rf"\b{key}\s*=\s*([^/]+)", body, flags=re.I)
    if match is None:
        return None
    numbers = re.findall(
        r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", match.group(1)
    )
    if len(numbers) < count:
        return None
    return [float(value) for value in numbers[:count]]


def load_factory_geometry(
    fds_path: str | Path,
) -> tuple[list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load mesh, OBST and HOLE geometry from the authoritative FDS file."""

    fds_path = Path(fds_path)
    mesh_xb: list[float] | None = None
    obstacles: list[dict[str, Any]] = []
    holes: list[dict[str, Any]] = []

    for name, body in _read_fds_namelists(fds_path):
        if name == "MESH":
            mesh_xb = _parse_numbers(body, "XB", 6)
        elif name in {"OBST", "HOLE"}:
            xb = _parse_numbers(body, "XB", 6)
            if xb is None:
                continue
            target = obstacles if name == "OBST" else holes
            target.append({"xb": xb})

    if mesh_xb is None:
        raise ValueError(f"MESH XB metadata not found in {fds_path}")
    return mesh_xb, obstacles, holes


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.ndim != 1 or source.size == 0:
        raise ValueError("Source coordinate axis must be a non-empty 1-D array")
    if np.any(np.diff(source) <= 0.0):
        raise ValueError("Source coordinate axis must be strictly increasing")
    return np.abs(source[:, None] - target[None, :]).argmin(axis=0)


def _axis_coordinates(
    count: int, minimum: float, maximum: float, resolution: float
) -> np.ndarray:
    cell_count = int(round((maximum - minimum) / resolution))
    if count == cell_count + 1:
        return np.linspace(minimum, maximum, count)
    if count == cell_count:
        return minimum + resolution / 2.0 + np.arange(count) * resolution
    raise ValueError(
        f"Axis length {count} does not match {cell_count} cells or "
        f"{cell_count + 1} boundaries"
    )


def _load_temperature_plane(
    npz_path: str | Path,
    mesh_xb: list[float],
    requested_time: float,
    requested_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    with np.load(npz_path) as data:
        required = {"temperature", "times", "xy_resolution", "z_resolution"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Temperature NPZ missing keys: {sorted(missing)}")
        temperature = np.asarray(data["temperature"], dtype=float)
        times = np.asarray(data["times"], dtype=float)
        xy_resolution = float(data["xy_resolution"])
        z_resolution = float(data["z_resolution"])

    if temperature.ndim != 4 or temperature.shape[0] != times.size:
        raise ValueError(
            "Temperature must have four axes with time first: "
            f"shape={temperature.shape}, times={times.shape}"
        )

    extents = (
        mesh_xb[1] - mesh_xb[0],
        mesh_xb[3] - mesh_xb[2],
        mesh_xb[5] - mesh_xb[4],
    )
    resolutions = (xy_resolution, xy_resolution, z_resolution)
    expected = tuple(int(round(v / r)) for v, r in zip(extents, resolutions))
    spatial_shape = temperature.shape[1:]
    matches = []
    for permutation in permutations(range(3)):
        reordered = tuple(spatial_shape[index] for index in permutation)
        if all(size in {cells, cells + 1} for size, cells in zip(reordered, expected)):
            matches.append(permutation)
    if len(matches) != 1:
        raise ValueError(
            "Cannot determine temperature spatial axes uniquely: "
            f"shape={spatial_shape}, expected xyz cells={expected}, matches={matches}"
        )

    permutation = matches[0]
    temperature_xyz = np.transpose(
        temperature, (0, 1 + permutation[0], 1 + permutation[1], 1 + permutation[2])
    )
    x_coords = _axis_coordinates(
        temperature_xyz.shape[1], mesh_xb[0], mesh_xb[1], xy_resolution
    )
    y_coords = _axis_coordinates(
        temperature_xyz.shape[2], mesh_xb[2], mesh_xb[3], xy_resolution
    )
    z_coords = _axis_coordinates(
        temperature_xyz.shape[3], mesh_xb[4], mesh_xb[5], z_resolution
    )
    time_index = int(np.abs(times - requested_time).argmin())
    z_index = int(np.abs(z_coords - requested_height).argmin())
    # Convert normalized (x, y) storage to the project-wide map[y, x] rule.
    plane_yx = temperature_xyz[time_index, :, :, z_index].T
    return plane_yx, x_coords, y_coords, float(times[time_index]), float(z_coords[z_index])


def _find_co_slice(simulation: Any) -> Any:
    for slc in simulation.slices:
        if "CO_Z130" in str(slc).upper():
            return slc
    for slc in simulation.slices:
        if "CARBON MONOXIDE" in str(slc).upper():
            return slc
    raise ValueError("FDS result does not contain a carbon-monoxide slice")


def _load_co_plane(
    fds_dir: str | Path, requested_time: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    try:
        import fdsreader as fds
    except ImportError as exc:
        raise RuntimeError(
            "fdsreader is required to load the FDS CO slice"
        ) from exc

    simulation = fds.Simulation(str(fds_dir))
    co_slice = _find_co_slice(simulation)
    description = str(co_slice).upper()
    unit = str(getattr(getattr(co_slice, "quantity", None), "unit", "")).lower()
    if "VOLUME FRACTION" not in description and unit not in {"mol/mol", "ppm"}:
        raise ValueError(
            "CO unit is not a verified volume fraction or ppm: "
            f"description={co_slice}, unit={unit!r}"
        )

    values, coords = co_slice.to_global(
        masked=True, fill=np.nan, return_coordinates=True
    )
    values = np.asarray(values, dtype=float)
    times = np.asarray(co_slice.times, dtype=float)
    x_coords = np.asarray(coords["x"], dtype=float)
    y_coords = np.asarray(coords["y"], dtype=float)
    z_coords = np.asarray(coords.get("z", [np.nan]), dtype=float)
    if values.ndim != 3 or values.shape[0] != times.size:
        raise ValueError(
            f"CO data must be (time, x, y) or (time, y, x): {values.shape}"
        )

    time_index = int(np.abs(times - requested_time).argmin())
    plane = values[time_index]
    if plane.shape == (x_coords.size, y_coords.size):
        plane_yx = plane.T
    elif plane.shape == (y_coords.size, x_coords.size):
        plane_yx = plane
    else:
        raise ValueError(
            f"CO coordinate/data mismatch: data={plane.shape}, "
            f"x={x_coords.size}, y={y_coords.size}"
        )

    # factory_v1.fds declares VOLUME FRACTION and Smokeview reports mol/mol.
    # Volume fraction (mol/mol) converts to ppm by multiplying by 1e6.
    if unit == "ppm":
        co_ppm = plane_yx
    else:
        co_ppm = plane_yx * 1_000_000.0
    return co_ppm, x_coords, y_coords, float(times[time_index]), float(z_coords[0])


def _resample_yx_nearest(
    source_yx: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    expected = (source_y.size, source_x.size)
    if source_yx.shape != expected:
        raise ValueError(
            f"Source map/coordinate mismatch: map={source_yx.shape}, expected={expected}"
        )
    x_indices = _nearest_indices(source_x, target_x)
    y_indices = _nearest_indices(source_y, target_y)
    return source_yx[np.ix_(y_indices, x_indices)]


def _risk_cost(
    values: np.ndarray, safe: float, blocked: float, weight: float, power: float
) -> np.ndarray:
    normalized = np.clip((values - safe) / (blocked - safe), 0.0, 1.0)
    return weight * np.power(normalized, power)


def build_fire_costmap(
    fds_path: str | Path,
    temperature_npz_path: str | Path,
    fds_result_dir: str | Path,
    config: FireCostmapConfig,
) -> FireCostmapLayers:
    """Build all static cost layers on one FDS-derived planner grid."""

    mesh_xb, obstacles, holes = load_factory_geometry(fds_path)
    clearance = config.inflation_radius if config.use_inflation else 0.0
    grid_map = GridMap(
        mesh_xb=mesh_xb,
        obstacles=obstacles,
        holes=holes,
        resolution=config.grid_resolution,
        clearance=clearance,
    )
    target_x = np.array(
        [grid_map.grid_to_world(gx, 0)[0] for gx in range(grid_map.width)]
    )
    target_y = np.array(
        [grid_map.grid_to_world(0, gy)[1] for gy in range(grid_map.height)]
    )

    temperature, temp_x, temp_y, temp_time, temp_height = _load_temperature_plane(
        temperature_npz_path, mesh_xb, config.fds_time, config.robot_height
    )
    co_ppm, co_x, co_y, co_time, co_height = _load_co_plane(
        fds_result_dir, config.fds_time
    )
    temperature_map = _resample_yx_nearest(
        temperature, temp_x, temp_y, target_x, target_y
    )
    co_map = _resample_yx_nearest(co_ppm, co_x, co_y, target_x, target_y)
    static_map = np.asarray(grid_map.occupancy, dtype=bool)

    temperature_cost = _risk_cost(
        temperature_map,
        config.temperature_safe,
        config.temperature_blocked,
        config.temperature_weight,
        config.temperature_power,
    )
    co_cost = _risk_cost(
        co_map,
        config.co_safe,
        config.co_blocked,
        config.co_weight,
        config.co_power,
    )
    invalid_environment = ~np.isfinite(temperature_map) | ~np.isfinite(co_map)
    blocked = (
        static_map
        | invalid_environment
        | (temperature_map >= config.temperature_blocked)
        | (co_map >= config.co_blocked)
    )
    final_cost = config.base_cost + temperature_cost + co_cost
    final_cost[blocked] = np.inf

    layers = FireCostmapLayers(
        grid_map=grid_map,
        static_obstacle_map=static_map,
        temperature_map=temperature_map,
        co_map=co_map,
        temperature_cost_map=temperature_cost,
        co_cost_map=co_cost,
        blocked_mask=blocked,
        final_cost_map=final_cost,
        selected_temperature_time=temp_time,
        selected_co_time=co_time,
        selected_temperature_height=temp_height,
        selected_co_height=co_height,
    )
    layers.validate()
    return layers
