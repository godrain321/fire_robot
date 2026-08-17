from pathlib import Path

import yaml

from run_partial_costmap_evacuation import SimulationMetrics, export_actual_path


def test_actual_path_export_contains_only_supplied_actual_poses(tmp_path):
    metrics = SimulationMetrics(
        travelled_distance=1.25,
        actual_path_world=[(1.0, 2.0), (1.5, 2.0), (2.0, 2.75)],
    )
    yaml_path = tmp_path / "path.yaml"
    image_path = tmp_path / "path.png"

    export_actual_path(metrics, yaml_path, image_path, elapsed=3.5)

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["coordinate_frame"] == "factory_v5_world_xy_m"
    assert data["points"] == [
        {"x": 1.0, "y": 2.0},
        {"x": 1.5, "y": 2.0},
        {"x": 2.0, "y": 2.75},
    ]
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
