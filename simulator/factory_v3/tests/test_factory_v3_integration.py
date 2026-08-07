"""Factory-v3 coordinate, CATF, scenario, and sensor integration checks."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml

from human_detection_sim import SimpleHumanDetector
from mapping.fire_costmap import load_factory_geometry
from mapping.grid_map import GridMap
from sensors.thermal_camera import ThermalCameraMLX90640
from simulation.ground_truth import FDSGroundTruthEnvironment


BASE = Path(__file__).resolve().parents[1]


def _scenario_grid():
    mesh, obstacles, holes = load_factory_geometry(BASE / "factory_v3.fds")
    scenario = yaml.safe_load(
        (BASE / "config" / "evacuation.yaml").read_text(encoding="utf-8")
    )
    grid = GridMap(
        mesh, obstacles, holes,
        resolution=scenario["planner"]["grid_resolution_m"],
        clearance=scenario["planner"]["inflation_radius_m"],
    )
    return mesh, obstacles, grid, scenario


def test_catf_geometry_is_loaded():
    mesh, obstacles, holes = load_factory_geometry(BASE / "factory_v3.fds")
    assert mesh == [1.8, 30.0, 5.6, 28.0, 0.0, 3.0]
    assert len(obstacles) > 600


def test_world_grid_roundtrip_and_boundaries():
    mesh, _, grid, _ = _scenario_grid()
    for x, y in (
        (mesh[0], mesh[2]),
        (mesh[1] - 1e-6, mesh[3] - 1e-6),
        (13.0, 16.0),
    ):
        node = grid.world_to_grid(x, y)
        assert grid.in_bounds(node)
        roundtrip = grid.grid_to_world(*node)
        assert abs(roundtrip[0] - x) <= grid.resolution / 2 + 1e-6
        assert abs(roundtrip[1] - y) <= grid.resolution / 2 + 1e-6
    assert not grid.in_bounds(grid.world_to_grid(mesh[1] + grid.resolution, mesh[3]))


def test_smokeview_boundary_display_is_separate_from_inflated_planner_map():
    mesh, obstacles, holes = load_factory_geometry(BASE / "factory_v3.fds")
    point = (13.2, 11.4)
    planner = GridMap(mesh, obstacles, holes, resolution=0.2, clearance=0.45)
    display = GridMap(
        mesh, obstacles, holes, resolution=0.2, clearance=0.0,
        include_lower_obstacle_boundary=False,
    )
    node = planner.world_to_grid(*point)
    assert planner.is_blocked(node)  # Robot clearance remains conservative.
    assert not display.is_blocked(node)  # Smokeview/FDS lower boundary stays open.


def test_configured_mission_points_are_explicit_and_free():
    _, _, grid, scenario = _scenario_grid()
    points = [
        ("robot_start", scenario["robot_start"]),
        *[(human["id"], human) for human in scenario["humans"]],
        *[(item["id"], item["approach"]) for item in scenario["exits"]],
    ]
    for name, point in points:
        node = grid.world_to_grid(float(point["x"]), float(point["y"]))
        assert grid.in_bounds(node), name
        assert not grid.is_blocked(node), name


def test_thermal_camera_respects_nonzero_mesh_origin():
    volume = np.full((3, 4, 6), 20.0, dtype=np.float32)
    volume[:, :, 4] = 90.0
    camera = ThermalCameraMLX90640(
        width=1, height=1, fov_h_deg=0.0, fov_v_deg=0.0,
        camera_height=0.2, front_offset=0.0, min_range=0.1,
        max_range=1.0, noise_std=0.0, accuracy_limit=0.0,
    )
    image, observations = camera.sense_3d(
        volume, robot_x=1.8, robot_y=5.8, robot_theta=0.0,
        xy_resolution=0.2, z_resolution=0.2,
        origin=(1.8, 5.6, 0.0), return_observations=True,
    )
    assert image.shape == (1, 1)
    assert observations[0].ray_cells
    assert observations[0].ray_cells[0].grid_position[0] >= 0


def test_victim_starts_outside_initial_detection_range():
    _, _, grid, scenario = _scenario_grid()
    detector = SimpleHumanDetector(scenario["human_detection_range_m"])
    start = scenario["robot_start"]
    victim = scenario["humans"][0]
    detections = detector.detect(
        (start["x"], start["y"]), scenario["humans"],
        obstacle_map=np.asarray(grid.occupancy, dtype=bool),
        map_origin=(grid.x_min, grid.y_min), map_resolution=grid.resolution,
    )
    distance = np.hypot(victim["x"] - start["x"], victim["y"] - start["y"])
    assert distance > scenario["human_detection_range_m"]
    assert detections == []


def test_ground_truth_loads_mock_temperature_and_co(monkeypatch, tmp_path):
    temperature_path = tmp_path / "temperature.npz"
    np.savez_compressed(
        temperature_path,
        temperature=np.full((2, 16, 113, 142), 25.0, dtype=np.float32),
        times=np.asarray([0.0, 1.0], dtype=np.float32),
        xy_resolution=np.asarray(0.2, dtype=np.float32),
        z_resolution=np.asarray(0.2, dtype=np.float32),
    )
    x = np.linspace(1.8, 30.0, 142)
    y = np.linspace(5.6, 28.0, 113)

    class MockSlice:
        times = np.asarray([0.0, 1.0])
        quantity = SimpleNamespace(unit="mol/mol")

        def __str__(self):
            return "CO_Z130 CARBON MONOXIDE VOLUME FRACTION"

        def to_global(self, **_kwargs):
            values = np.full((2, x.size, y.size), 0.001, dtype=float)
            return values, {"x": x, "y": y, "z": np.asarray([1.3])}

    mock_fdsreader = SimpleNamespace(
        Simulation=lambda _path: SimpleNamespace(slices=[MockSlice()])
    )
    monkeypatch.setitem(sys.modules, "fdsreader", mock_fdsreader)
    environment = FDSGroundTruthEnvironment(
        BASE / "factory_v3.fds", temperature_path, tmp_path
    )
    assert environment.temperature_time_range == (0.0, 1.0)
    assert environment.co_time_range == (0.0, 1.0)
    state = SimpleNamespace(x=13.0, y=16.0, theta=0.0)
    camera = ThermalCameraMLX90640(
        width=1, height=1, noise_std=0.0, accuracy_limit=0.0
    )
    image, observations, selected_time = environment.capture_thermal(camera, state, 0.0)
    assert image.shape == (1, 1)
    assert observations
    assert selected_time == 0.0
    exposure = environment.evaluate_exposure(13.0, 16.0, 1.3, 0.0)
    assert exposure.co_ppm == 1000.0
    temperature_yx, co_yx, map_time = environment.sample_map_yx(
        np.asarray([1.8, 30.0]), np.asarray([5.6, 28.0]), 1.3, 0.0
    )
    assert temperature_yx.shape == (2, 2)
    assert co_yx.shape == (2, 2)
    assert np.all(temperature_yx == 25.0)
    assert np.all(co_yx == 1000.0)
    assert map_time == 0.0
