#!/usr/bin/env python3
"""Pack factory_v5 temperature CSV frames atomically and delete only on success."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from fds_temperature_io import axis_resolution, load_temperature_csv
from mapping.fire_costmap import load_factory_geometry


TIME_RE = re.compile(r"factory_v5_cat_temp_3d_t(\d+(?:\.\d+)?)\.csv$")


def extract_time(path: Path) -> float:
    """Extract a numeric simulation time from one dedicated CSV filename."""
    match = TIME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected temperature CSV filename: {path.name}")
    return float(match.group(1))


def verify_npz(path: Path, expected_frames: int) -> dict[str, object]:
    """Reopen and validate the packed result before it replaces the final NPZ."""
    with np.load(path, allow_pickle=False) as data:
        required = {"temperature", "times", "xy_resolution", "z_resolution"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"NPZ missing keys: {sorted(missing)}")
        temperature = np.asarray(data["temperature"])
        times = np.asarray(data["times"])
        if temperature.ndim != 4 or temperature.shape[0] != expected_frames:
            raise ValueError(f"invalid temperature shape: {temperature.shape}")
        if times.shape != (expected_frames,) or np.any(np.diff(times) <= 0.0):
            raise ValueError(f"invalid or unsorted time axis: {times}")
        if temperature.dtype != np.float32:
            raise ValueError(f"temperature dtype must be float32: {temperature.dtype}")
        stats = {
            "keys": list(data.files),
            "shape": tuple(temperature.shape),
            "dtype": str(temperature.dtype),
            "time_count": int(times.size),
            "time_min": float(times.min()),
            "time_max": float(times.max()),
            "nan_count": int(np.isnan(temperature).sum()),
            "temperature_min": float(np.nanmin(temperature)),
            "temperature_max": float(np.nanmax(temperature)),
        }
        if stats["nan_count"]:
            raise ValueError(f"temperature NPZ contains NaN: {stats}")
        return stats


def read_mesh_ijk(fds_path: Path) -> tuple[int, int, int]:
    """Read the single factory-v5 MESH cell counts from the authoritative input."""
    text = fds_path.read_text(encoding="utf-8")
    match = re.search(
        r"&MESH\b.*?\bIJK\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
        text, flags=re.I | re.S,
    )
    if match is None:
        raise ValueError(f"MESH IJK not found in {fds_path}")
    return tuple(int(value) for value in match.groups())


def pack_direct_from_fds(
    base: Path, output: Path, mesh_tuple: tuple[float, ...]
) -> dict[str, object]:
    """Pack the complete cell-centred TEMP_3D slice without CSV expansion."""
    try:
        import fdsreader as fds
    except ImportError as exc:
        raise RuntimeError("fdsreader is required for --from-fds") from exc

    simulation = fds.Simulation(str(base))
    slices = [item for item in simulation.slices if "TEMPERATURE" in str(item).upper()]
    if not slices:
        raise ValueError("FDS result does not contain a temperature slice")
    temperature_slice = slices[0]
    values, coords = temperature_slice.to_global(
        masked=False, fill=np.nan, return_coordinates=True
    )
    values = np.asarray(values, dtype=float)
    times = np.asarray(temperature_slice.times, dtype=np.float32)
    x = np.asarray(coords["x"], dtype=float)
    y = np.asarray(coords["y"], dtype=float)
    z = np.asarray(coords["z"], dtype=float)
    if values.shape[1:] == (x.size, y.size, z.size):
        temperature = np.transpose(values, (0, 3, 2, 1))
    elif values.shape[1:] == (z.size, y.size, x.size):
        temperature = values
    else:
        raise ValueError(
            f"temperature coordinate/data mismatch: {values.shape}, "
            f"coordinates={(x.size, y.size, z.size)}"
        )
    if not np.isfinite(temperature).all():
        raise ValueError("temperature FDS slice contains NaN or infinity")
    # fdsreader exposes boundary-to-boundary linspace coordinates for a
    # cell-centred slice. Derive the authoritative cell size from MESH XB and
    # the actual cell counts instead of treating those display coordinates as
    # cell centres.
    dx = (mesh_tuple[1] - mesh_tuple[0]) / x.size
    dy = (mesh_tuple[3] - mesh_tuple[2]) / y.size
    dz = (mesh_tuple[5] - mesh_tuple[4]) / z.size
    if not np.isclose(dx, dy, atol=1e-6, rtol=0.0):
        raise ValueError(f"x/y resolutions differ: dx={dx}, dy={dy}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="fds_temperature_3d_timeseries.", suffix=".npz",
            dir=output.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary_path,
            temperature=np.asarray(temperature, dtype=np.float32),
            times=times,
            xy_resolution=np.asarray(dx, dtype=np.float32),
            z_resolution=np.asarray(dz, dtype=np.float32),
            x_coordinates=x.astype(np.float32),
            y_coordinates=y.astype(np.float32),
            z_coordinates=z.astype(np.float32),
            mesh_xb=np.asarray(mesh_tuple, dtype=np.float32),
            axis_order=np.asarray("time,z,y,x"),
        )
        stats = verify_npz(temporary_path, times.size)
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return stats


def main() -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-csv", action="store_true", help="retain source CSV files")
    parser.add_argument(
        "--from-fds", action="store_true",
        help="read the complete cell-centred TEMP_3D slice directly with fdsreader",
    )
    args = parser.parse_args()
    csv_dir = base / "csv_temp3d"
    output = base / "processed" / "fds_temperature_3d_timeseries.npz"
    fds_path = base / "factory_v5.fds"
    mesh_xb, _, _ = load_factory_geometry(fds_path)
    mesh_ijk = read_mesh_ijk(fds_path)
    mesh_tuple = tuple(float(value) for value in mesh_xb)

    if args.from_fds:
        stats = pack_direct_from_fds(base, output, mesh_tuple)
        print(f"Saved and verified directly from FDS: {output}")
        for key, value in stats.items():
            print(f"{key}: {value}")
        return 0

    files = []
    for path in csv_dir.iterdir() if csv_dir.is_dir() else ():
        if path.is_file() and TIME_RE.fullmatch(path.name):
            files.append((extract_time(path), path))
    files.sort(key=lambda item: item[0])
    if not files:
        raise FileNotFoundError(f"no factory_v5 temperature CSV files in {csv_dir}")
    times = np.asarray([item[0] for item in files], dtype=np.float32)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"duplicate or unsorted CSV times: {times}")

    frames = []
    reference_axes = None
    row_count = None
    print(
        f"Packing {len(files)} temperature CSV files from {csv_dir}",
        flush=True,
    )
    for frame_index, (time_value, path) in enumerate(files, start=1):
        if frame_index == 1 or frame_index % 10 == 0 or frame_index == len(files):
            print(
                f"[{frame_index}/{len(files)}] loading t={time_value:g}s: {path.name}",
                flush=True,
            )
        frame = load_temperature_csv(path, mesh_tuple)
        axes = (frame.x, frame.y, frame.z)
        if reference_axes is None:
            actual_counts = (frame.x.size, frame.y.size, frame.z.size)
            if any(
                actual not in {cells, cells + 1}
                for actual, cells in zip(actual_counts, mesh_ijk)
            ):
                raise ValueError(
                    f"CSV grid {actual_counts} does not match MESH IJK {mesh_ijk}"
                )
            reference_axes = axes
            row_count = frame.parsed_rows
        elif frame.parsed_rows != row_count or any(
            not np.array_equal(current, reference)
            for current, reference in zip(axes, reference_axes)
        ):
            raise ValueError(f"coordinate or row-count mismatch at t={time_value}: {path}")
        frames.append(frame.temperature)
    assert reference_axes is not None
    x, y, z = reference_axes
    dx, dy = axis_resolution(x, "x"), axis_resolution(y, "y")
    dz = axis_resolution(z, "z")
    if not np.isclose(dx, dy, atol=1e-6, rtol=0.0):
        raise ValueError(f"x/y resolutions differ: dx={dx}, dy={dy}")
    print("Stacking frames into temperature[time,z,y,x]...", flush=True)
    temperature = np.stack(frames).astype(np.float32, copy=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="fds_temperature_3d_timeseries.", suffix=".npz",
            dir=output.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        print(f"Compressing temporary NPZ: {temporary_path}", flush=True)
        np.savez_compressed(
            temporary_path,
            temperature=temperature,
            times=times,
            xy_resolution=np.asarray(dx, dtype=np.float32),
            z_resolution=np.asarray(dz, dtype=np.float32),
            x_coordinates=x.astype(np.float32),
            y_coordinates=y.astype(np.float32),
            z_coordinates=z.astype(np.float32),
            mesh_xb=np.asarray(mesh_tuple, dtype=np.float32),
            axis_order=np.asarray("time,z,y,x"),
        )
        print("Reopening and verifying temporary NPZ...", flush=True)
        stats = verify_npz(temporary_path, len(files))
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Saved and verified: {output}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    if not args.keep_csv:
        for _, path in files:
            path.unlink()
        print(f"Deleted {len(files)} verified input files only from {csv_dir}")
    else:
        print("CSV files retained (--keep-csv).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
