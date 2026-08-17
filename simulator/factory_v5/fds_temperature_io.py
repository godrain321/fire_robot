"""Strict reader for factory_v5 fds2ascii 3-D temperature CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")


@dataclass(frozen=True)
class TemperatureFrame:
    """One fds2ascii frame in project-standard ``temperature[z,y,x]`` order."""

    temperature: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    parsed_rows: int


def load_temperature_csv(
    csv_path: str | Path,
    mesh_xb: tuple[float, float, float, float, float, float],
) -> TemperatureFrame:
    """Read x,y,z,T records, reject incomplete grids, and return ``[z,y,x]``."""
    path = Path(csv_path)
    points = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            values = NUMBER_RE.findall(line)
            if len(values) < 4:
                continue
            x, y, z, temperature = (
                float(value.replace("D", "E").replace("d", "E"))
                for value in values[:4]
            )
            if not (
                mesh_xb[0] - 1e-6 <= x <= mesh_xb[1] + 1e-6
                and mesh_xb[2] - 1e-6 <= y <= mesh_xb[3] + 1e-6
                and mesh_xb[4] - 1e-6 <= z <= mesh_xb[5] + 1e-6
            ):
                continue
            # FDS gas/flame cells can legitimately exceed 2000 °C. Keep only
            # a broad sanity bound here; scenario-specific plausibility belongs
            # in the post-pack statistics, not in the CSV parser.
            if not np.isfinite(temperature) or not -273.15 <= temperature <= 5000.0:
                raise ValueError(
                    f"nonphysical temperature in {path}: {temperature}"
                )
            points.append((x, y, z, temperature))
    if not points:
        raise ValueError(f"no x,y,z,temperature records found in {path}")

    array = np.asarray(points, dtype=np.float64)
    x = np.unique(np.round(array[:, 0], 6))
    y = np.unique(np.round(array[:, 1], 6))
    z = np.unique(np.round(array[:, 2], 6))
    expected_rows = x.size * y.size * z.size
    if len(points) != expected_rows:
        raise ValueError(
            f"CSV is not one complete Cartesian grid: rows={len(points)}, "
            f"expected={expected_rows}, axes={(x.size, y.size, z.size)}, file={path}"
        )

    xi = {value: index for index, value in enumerate(x)}
    yi = {value: index for index, value in enumerate(y)}
    zi = {value: index for index, value in enumerate(z)}
    volume = np.full((z.size, y.size, x.size), np.nan, dtype=np.float32)
    for px, py, pz, temperature in array:
        index = (zi[round(float(pz), 6)], yi[round(float(py), 6)], xi[round(float(px), 6)])
        if np.isfinite(volume[index]):
            raise ValueError(f"duplicate coordinate {index} in {path}")
        volume[index] = temperature
    if np.isnan(volume).any():
        raise ValueError(f"incomplete temperature grid in {path}")
    return TemperatureFrame(volume, x, y, z, len(points))


def axis_resolution(axis: np.ndarray, name: str) -> float:
    """Return constant axis spacing and reject nonuniform coordinates."""
    if axis.size < 2:
        raise ValueError(f"{name} axis has fewer than two coordinates")
    differences = np.diff(axis)
    spacing = float(np.median(differences))
    if not np.allclose(differences, spacing, atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} coordinates are not uniformly spaced")
    return spacing
