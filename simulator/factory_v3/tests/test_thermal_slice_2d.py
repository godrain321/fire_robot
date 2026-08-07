from types import SimpleNamespace

import numpy as np

from sensors.thermal_camera import ThermalCameraMLX90640
from simulation.thermal_slice_2d import FDSGroundTruthEnvironment2D


def environment():
    item = FDSGroundTruthEnvironment2D.__new__(FDSGroundTruthEnvironment2D)
    item.mesh_xb = (0.0, 4.0, 0.0, 4.0, 0.0, 1.0)
    item._temperature_times = np.array([0.0, 1.0])
    item._temperature_x = np.arange(5, dtype=float)
    item._temperature_y = np.arange(5, dtype=float)
    item._temperature_z = np.array([0.0, 0.3, 0.6, 0.9])
    item._xy_resolution = 1.0
    item._z_resolution = 0.3
    item._ground_truth_temperature_xyz = np.full((2, 5, 5, 4), 25.0)
    item._ground_truth_temperature_xyz[:, 2, 2, 1] = 80.0
    item._obstacle_volume_zyx = np.zeros((4, 5, 5), dtype=bool)
    item.observation_height_m = 0.3
    item._slice_z_index = 1
    item._obstacle_slice_yx = item._obstacle_volume_zyx[1].copy()
    item._cached_time_index = None
    item._cached_temperature_slice_yx = None
    item.slice_cache_build_count = 0
    return item


def camera():
    return ThermalCameraMLX90640(
        width=1, height=24, fov_h_deg=0, front_offset=0,
        min_range=0.2, max_range=3.0, accuracy_limit=0, noise_std=0,
    )


def test_2d_sensor_uses_one_horizontal_ray_not_24_vertical_rays():
    item = environment()
    image, rays, selected_time = item.capture_thermal(
        camera(), SimpleNamespace(x=0.0, y=2.0, theta=0.0), 0.0
    )
    assert image.shape == (24, 1)
    assert len(rays) == 1
    assert np.all(image == 80.0)
    assert rays[0].hit_world_position[2] == 0.3
    assert selected_time == 0.0


def test_temperature_slice_is_cached_for_same_fds_time_index():
    item = environment()
    sensor = camera()
    state = SimpleNamespace(x=0.0, y=2.0, theta=0.0)
    item.capture_thermal(sensor, state, 0.0)
    item.capture_thermal(sensor, state, 0.1)
    assert item.slice_cache_build_count == 1
    item.capture_thermal(sensor, state, 1.0)
    assert item.slice_cache_build_count == 2


def test_2d_obstacle_slice_stops_visibility():
    item = environment()
    item._obstacle_slice_yx[2, 1] = True
    image, rays, _ = item.capture_thermal(
        camera(), SimpleNamespace(x=0.0, y=2.0, theta=0.0), 0.0
    )
    assert rays[0].occluded
    assert image.max() == 25.0
    assert all(sample.world_position[0] < 2.0 for sample in rays[0].ray_cells)


def test_2d_runner_import_does_not_replace_integrated_runner_globally():
    import run_partial_costmap_evacuation as runner
    from simulation.ground_truth import FDSGroundTruthEnvironment
    import run_evacuation_2d

    assert runner.FDSGroundTruthEnvironment is FDSGroundTruthEnvironment
    assert run_evacuation_2d.FDSGroundTruthEnvironment2D is not FDSGroundTruthEnvironment
