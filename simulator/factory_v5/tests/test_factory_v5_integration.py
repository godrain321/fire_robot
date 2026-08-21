"""Factory-v3 coordinate, CATF, scenario, and sensor integration checks."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml

from human_detection_sim import SimpleHumanDetector
from mapping.fire_costmap import (
    load_factory_geometry, obstacles_for_initial_robot_map,
)
from mapping.grid_map import GridMap
from navigation.victim_scripted_motion import (
    ScriptedVictimMotionConfig, ScriptedVictimMotionController,
)
from sensors.thermal_camera import ThermalCameraMLX90640
from simulation.ground_truth import FDSGroundTruthEnvironment
from world.fire_maps import MapMetadata


BASE = Path(__file__).resolve().parents[1]


def _scenario_grid():
    mesh, obstacles, holes = load_factory_geometry(BASE / "factory_v5.fds")
    scenario = yaml.safe_load(
        (BASE / "config" / "evacuation.yaml").read_text(encoding="utf-8")
    )
    obstacles = obstacles_for_initial_robot_map(obstacles, scenario)
    grid = GridMap(
        mesh, obstacles, holes,
        resolution=scenario["planner"]["grid_resolution_m"],
        clearance=scenario["planner"]["inflation_radius_m"],
    )
    return mesh, obstacles, grid, scenario


def test_catf_geometry_is_loaded():
    mesh, obstacles, holes = load_factory_geometry(BASE / "factory_v5.fds")
    assert mesh == [1.8, 30.0, 5.6, 28.0, 0.0, 3.0]
    assert len(obstacles) > 600


def test_exit2_fire_line_replaces_strong_remote_oil_source():
    scenario_text = (BASE / "generated" / "scenario.inc").read_text(
        encoding="utf-8"
    )
    assert (
        "ID='MACHINERY_OIL_FIRE_OIL', HRRPUA=250.000"
        in scenario_text
    )
    assert (
        "ID='EXIT2_BLOCKING_FIRE', HRRPUA=1200.000"
        in scenario_text
    )
    fire_line = next(
        line for line in scenario_text.splitlines()
        if "ID='V3_EXIT2_BLOCKING_FIRE_001'" in line
    )
    assert "XB=19.200,21.400,9.400,9.600,0.000,0.000" in fire_line
    assert "XYZ=21.200,9.500,0.000" in fire_line
    assert "SPREAD_RATE=0.0300" in fire_line


def test_sensor_costmap_makes_observed_developing_fire_costly():
    scenario = yaml.safe_load(
        (BASE / "config" / "evacuation.yaml").read_text(encoding="utf-8")
    )
    config = scenario["sensor_costmap"]
    assert config["temperature_weight"] == 24.0
    assert config["temperature_power"] == 1.5

    # An observed 40 C cell is cautionary well before the unchanged 60 C
    # hard block. Unknown cells are not involved in this calculation.
    normalized = (40.0 - 20.0) / (60.0 - 20.0)
    observed_temperature_cost = (
        config["temperature_weight"]
        * normalized ** config["temperature_power"]
    )
    assert observed_temperature_cost > 8.0


def test_exit1_scenario_blocker_is_loaded_without_changing_base_includes():
    _, obstacles, _ = load_factory_geometry(BASE / "factory_v5.fds")
    blocker_xb = [7.8, 10.8, 17.8, 18.0, 0.0, 1.4]
    assert any(item["xb"] == blocker_xb for item in obstacles)
    scenario = yaml.safe_load(
        (BASE / "config" / "evacuation.yaml").read_text(encoding="utf-8")
    )
    exit1 = next(item for item in scenario["exits"] if item["id"] == "EXIT1")
    assert exit1["initial_status"] == "unknown"
    assert exit1["approach"] == {"x": 9.4, "y": 17.4}
    planner_obstacles = obstacles_for_initial_robot_map(obstacles, scenario)
    assert not any(item["xb"] == blocker_xb for item in planner_obstacles)


def test_optional_unobserved_obstacle_may_be_absent_from_scenario():
    obstacles = [{"id": "KNOWN_WALL", "xb": [0, 1, 0, 1, 0, 1]}]
    scenario = {
        "robot_map": {
            "initially_unobserved_fds_obstacle_ids": [
                "EXIT1_FALLEN_STORAGE_RACK"
            ]
        }
    }
    assert obstacles_for_initial_robot_map(obstacles, scenario) == obstacles


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
    mesh, obstacles, holes = load_factory_geometry(BASE / "factory_v5.fds")
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


def test_victim_starts_hidden_by_slam_occupancy_before_scripted_motion():
    _, _, grid, scenario = _scenario_grid()
    detector = SimpleHumanDetector(
        scenario["human_detection_range_m"],
        scenario["human_detection_horizontal_fov_deg"],
    )
    start = scenario["robot_start"]
    victim = scenario["humans"][0]
    detections = detector.detect(
        (start["x"], start["y"]), scenario["humans"],
        robot_heading_rad=np.radians(start["yaw_deg"]),
        obstacle_map=np.asarray(grid.occupancy, dtype=bool),
        map_origin=(grid.x_min, grid.y_min), map_resolution=grid.resolution,
    )
    assert (victim["x"], victim["y"]) == (19.4, 19.0)
    assert detections == []


def test_configured_victim_wall_route_reaches_requested_destination():
    _, _, grid, scenario = _scenario_grid()
    victim = scenario["humans"][0]
    motion = ScriptedVictimMotionConfig.from_mapping(victim["scripted_motion"])
    assert motion.speed_mps == 1.2
    assert scenario["victim_following"]["victim_speed_mps"] == 1.2
    controller = ScriptedVictimMotionController(
        # The WorldState metadata contract uses the same origin/resolution.
        MapMetadata(
            grid.x_min, grid.x_max, grid.y_min, grid.y_max,
            grid.resolution, grid.width, grid.height, (grid.x_min, grid.y_min),
        ),
        victim["id"], (victim["x"], victim["y"]), motion,
    )
    static = np.asarray(grid.occupancy, dtype=bool)
    dynamic = np.zeros_like(static)
    for _ in range(200):
        controller.update(
            dt=0.1, static_obstacle_map=static,
            dynamic_obstacle_map=dynamic,
        )
        if controller.completed:
            break
    assert controller.completed
    assert controller.position_world == (14.4, 15.0)


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
        BASE / "factory_v5.fds", temperature_path, tmp_path
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
