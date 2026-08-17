import json

import numpy as np
import yaml

from mapping.fire_localization import FireLocalizationConfig, FireLocalizer
from mapping.grid_map import GridMap
from mapping.partial_costmap import PartialCostmapConfig, PartialFireCostmap
from sensors.thermal_camera import ThermalRayObservation, ThermalRaySample
from world.fire_maps import EstimatedFireMap, MapMetadata


def test_estimated_map_copies_layers_and_costmap_revision_changes_only_for_cost_change():
    metadata = MapMetadata(0, 2, 0, 2, 1, 3, 3, (0, 0))
    static = np.zeros((3, 3), bool)
    cfg = FireLocalizationConfig(
        prior_probability=0.05,
        possible_probability_threshold=0.10,
        likely_probability_threshold=0.30,
        confirm_probability_threshold=0.60,
        release_probability_threshold=0.50,
        candidate_probability_threshold=0.10,
    )
    localizer = FireLocalizer(metadata, static, cfg)
    sample = ThermalRaySample((1, 1, 0), (1.0, 1.0, 0.3), 1.0, 80.0)
    observation = ThermalRayObservation(
        0, 0, 80, (sample,), sample.world_position, sample.grid_position,
        sample.distance, True, False,
    )
    localizer.add_thermal_observation("thermal", 1, (0, 1, 0), (observation,))
    estimated = EstimatedFireMap(metadata, temperature_blocked_c=60, co_blocked_ppm=1600)
    estimated.sync_fire_localization(localizer)
    assert estimated.fire_probability is not localizer.fire_probability
    assert estimated.fire_probability[1, 1] == localizer.fire_probability[1, 1]

    grid = GridMap((0, 2, 0, 2, 0, 1), [], [], 1, 0)
    belief = PartialFireCostmap(grid, static, PartialCostmapConfig(grid_resolution=1))
    first = belief.update_estimated_fire_probability(
        localizer.fire_probability, cost_weight=50, minimum_probability=0.1
    )
    revision = belief.revision
    assert first.changed_cells
    assert np.isfinite(belief.final_cost_map[1, 1])
    assert not belief.blocked_mask[1, 1]
    second = belief.update_estimated_fire_probability(
        localizer.fire_probability.copy(), cost_weight=50, minimum_probability=0.1
    )
    assert not second.changed_cells
    assert belief.revision == revision


def test_yaml_loading_and_json_serialization():
    from pathlib import Path
    base = Path(__file__).resolve().parents[1]
    scenario = yaml.safe_load((base / "config/evacuation.yaml").read_text())
    cfg = FireLocalizationConfig.from_mapping(scenario["fire_localization"])
    assert cfg.enabled
    metadata = MapMetadata(0, 1, 0, 1, 1, 2, 2, (0, 0))
    localizer = FireLocalizer(metadata, np.zeros((2, 2), bool), cfg)
    json.dumps(localizer.to_dict(), allow_nan=True)


def test_localizer_api_has_no_ground_truth_dependency():
    import inspect
    import mapping.fire_localization as module
    source = inspect.getsource(module)
    assert "GroundTruth" not in source
    assert "fds_result" not in source
