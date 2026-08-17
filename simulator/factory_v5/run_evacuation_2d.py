#!/usr/bin/env python3
"""Run factory_v5 with a cached 0.30 m 2-D thermal observation slice."""

from __future__ import annotations

import sys

import run_partial_costmap_evacuation as runner
from simulation.thermal_slice_2d import FDSGroundTruthEnvironment2D


def main() -> int:
    # Keep the original integrated runner untouched and replace only its
    # simulator-side sensor facade. The 2-D mode intentionally has no thermal
    # image window.
    runner.FDSGroundTruthEnvironment = FDSGroundTruthEnvironment2D
    if "--no-thermal-window" not in sys.argv:
        sys.argv.append("--no-thermal-window")
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
