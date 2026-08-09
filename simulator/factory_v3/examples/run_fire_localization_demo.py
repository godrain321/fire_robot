"""Small fire-localization demo that needs neither FDS nor Pygame."""

import json
import math
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from mapping.fire_localization import FireLocalizationConfig, FireLocalizer
from sensors.thermal_camera import ThermalRayObservation, ThermalRaySample
from world.fire_maps import MapMetadata


def make_ray(cells, temperature=85.0):
    samples = tuple(
        ThermalRaySample((x, y, 0), (x, y, 0.3), index + 1, temperature)
        for index, (x, y) in enumerate(cells)
    )
    return ThermalRayObservation(
        0, 0, temperature, samples, samples[-1].world_position,
        samples[-1].grid_position, samples[-1].distance, True, False,
    )


def main():
    metadata = MapMetadata(0, 6, 0, 6, 1, 7, 7, (0, 0))
    localizer = FireLocalizer(
        metadata, np.zeros((7, 7), bool),
        FireLocalizationConfig(
            possible_probability_threshold=0.2,
            likely_probability_threshold=0.4,
            confirm_probability_threshold=0.6,
            release_probability_threshold=0.5,
            candidate_probability_threshold=0.15,
            maximum_confirmed_region_cells=20,
        ),
    )
    views = (
        ((0, 3, 0), [(1, 3), (2, 3), (3, 3)]),
        ((3, 0, math.pi / 2), [(3, 1), (3, 2), (3, 3)]),
        ((6, 3, math.pi), [(5, 3), (4, 3), (3, 3)]),
    )
    for index, (pose, cells) in enumerate(views):
        result = localizer.add_thermal_observation(
            f"thermal-{index}", index + 1, pose, (make_ray(cells),)
        )
        print(index + 1, result.state.name, result.highest_probability_grid,
              f"p={result.highest_probability:.3f}")
    print(json.dumps(localizer.latest_result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
