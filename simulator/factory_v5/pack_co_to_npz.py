#!/usr/bin/env python3
"""Pack the factory_v5 CO slice into a portable, runtime-ready NPZ."""

from __future__ import annotations

from pathlib import Path
import tempfile
import os

import numpy as np


def main() -> int:
    try:
        import fdsreader as fds
    except ImportError as exc:
        raise RuntimeError("fdsreader is required to pack the CO result") from exc

    base = Path(__file__).resolve().parent
    simulation = fds.Simulation(str(base))
    slices = [item for item in simulation.slices if "CO_Z130" in str(item).upper()]
    if not slices:
        slices = [
            item for item in simulation.slices
            if "CARBON MONOXIDE" in str(item).upper()
        ]
    if not slices:
        raise ValueError("FDS result does not contain a carbon-monoxide slice")
    co_slice = slices[0]
    values, coords = co_slice.to_global(
        masked=False, fill=np.nan, return_coordinates=True
    )
    values = np.asarray(values, dtype=float)
    times = np.asarray(co_slice.times, dtype=float)
    x_coords = np.asarray(coords["x"], dtype=float)
    y_coords = np.asarray(coords["y"], dtype=float)
    height = float(np.asarray(coords.get("z", [np.nan]), dtype=float)[0])
    if values.shape[1:] == (x_coords.size, y_coords.size):
        co_yx = np.transpose(values, (0, 2, 1))
    elif values.shape[1:] == (y_coords.size, x_coords.size):
        co_yx = values
    else:
        raise ValueError(f"CO coordinate/data mismatch: {values.shape}")
    unit = str(getattr(getattr(co_slice, "quantity", None), "unit", "")).lower()
    if unit == "ppm":
        co_ppm = co_yx
    elif unit in {"mol/mol", ""} or "VOLUME FRACTION" in str(co_slice).upper():
        co_ppm = co_yx * 1_000_000.0
    else:
        raise ValueError(f"unsupported CO unit: {unit!r}")

    output = base / "processed" / "fds_co_2d_timeseries.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="fds_co_2d_timeseries.", suffix=".tmp.npz",
            dir=output.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary_path,
            co_ppm=np.asarray(co_ppm, dtype=np.float32),
            times=np.asarray(times, dtype=np.float32),
            x_coordinates=np.asarray(x_coords, dtype=np.float32),
            y_coordinates=np.asarray(y_coords, dtype=np.float32),
            height=np.asarray(height, dtype=np.float32),
            axis_order=np.asarray("time,y,x"),
        )
        with np.load(temporary_path, allow_pickle=False) as packed:
            if packed["co_ppm"].shape != (times.size, y_coords.size, x_coords.size):
                raise ValueError("packed CO NPZ has an invalid shape")
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"Saved {output}: shape={co_ppm.shape}, time={times[0]}..{times[-1]} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
