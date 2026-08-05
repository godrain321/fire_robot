"""Unit and small integration tests for stage-2 partial costmap evacuation."""

import inspect
import math
from types import SimpleNamespace

import numpy as np

from mapping.grid_map import GridMap
from mapping.partial_costmap import (
    PartialCostmapConfig,
    PartialFireCostmap,
    path_has_new_block,
)
from planner.a_star import weighted_a_star_with_escape
from robot.path_follower import RobotState, ReplannablePathFollower
from sensors.thermal_camera import ThermalCameraMLX90640, ThermalRaySample


def make_belief(**config_overrides):
    grid = GridMap(
        mesh_xb=[0.0, 6.0, 0.0, 6.0, 0.0, 2.0],
        obstacles=[], holes=[], resolution=1.0, clearance=0.0,
    )
    config = PartialCostmapConfig(
        grid_resolution=1.0, use_inflation=False, **config_overrides
    )
    belief = PartialFireCostmap(
        grid, np.zeros((grid.height, grid.width), dtype=bool), config
    )
    return grid, config, belief


def ray_at(gx, gy, temperature):
    sample = ThermalRaySample(
        grid_position=(gx, gy, 0),
        world_position=(float(gx), float(gy), 0.25),
        distance=1.0,
        temperature=float(temperature),
    )
    return SimpleNamespace(
        row=0,
        col=0,
        valid=True,
        hit_world_position=sample.world_position,
        ray_cells=(sample,),
    )


def thermal_update(belief, observations, temperature, sim_time=1.0):
    image = np.full((1, 1), float(temperature))
    return belief.update_thermal_observations(image, observations, sim_time)


def plan(belief, start=(0, 3), goal=(6, 3)):
    return weighted_a_star_with_escape(
        belief.final_cost_map, start, goal, belief.static_obstacle_map
    )


def test_initial_unknown_map_still_produces_path():
    _, config, belief = make_belief()

    result = plan(belief)

    assert result.path
    assert np.isnan(belief.temperature_belief_map).all()
    assert np.isnan(belief.co_belief_map).all()
    assert np.allclose(
        belief.final_cost_map,
        config.base_cost + config.unknown_penalty,
    )


def test_high_thermal_observation_on_path_causes_detour():
    _, _, belief = make_belief()
    original = plan(belief)
    assert (3, 3) in original.path

    update = thermal_update(belief, [ray_at(3, 3, 80.0)], 80.0)
    replanned = plan(belief)

    assert (3, 3) in update.newly_blocked_cells
    assert replanned.path
    assert (3, 3) not in replanned.path
    assert replanned.path != original.path


def test_high_finite_co_cost_causes_detour():
    _, _, belief = make_belief(co_weight=1000.0)
    belief.update_co_observation(3.0, 3.0, 1500.0, 1.0)

    result = plan(belief)

    assert result.path
    assert (3, 3) not in result.path
    assert np.isfinite(belief.final_cost_map[3, 3])


def test_temperature_at_threshold_is_infinite_and_avoided():
    _, _, belief = make_belief()
    thermal_update(belief, [ray_at(3, 3, 60.0)], 60.0)

    result = plan(belief)

    assert np.isinf(belief.final_cost_map[3, 3])
    assert (3, 3) not in result.path


def test_co_at_threshold_is_infinite_and_avoided():
    _, _, belief = make_belief()
    belief.update_co_observation(3.0, 3.0, 1600.0, 1.0)

    result = plan(belief)

    assert np.isinf(belief.final_cost_map[3, 3])
    assert (3, 3) not in result.path


def test_unobserved_distant_ground_truth_does_not_leak_into_belief():
    _, _, belief = make_belief()
    ground_truth_temperature = np.full(belief.shape, 20.0)
    ground_truth_temperature[6, 6] = 500.0

    thermal_update(belief, [ray_at(1, 1, 25.0)], 25.0)

    assert belief.temperature_belief_map[1, 1] == 25.0
    assert np.isnan(belief.temperature_belief_map[6, 6])
    assert not belief.temperature_observed_mask[6, 6]
    assert ground_truth_temperature[6, 6] == 500.0


def test_new_risk_on_path_requests_immediate_replan():
    _, _, belief = make_belief()
    current = plan(belief)
    update = thermal_update(belief, [ray_at(3, 3, 60.0)], 60.0, 0.1)

    assert path_has_new_block(current.path, update.newly_blocked_cells, lookahead=20)


def test_fully_observed_blocked_exit_returns_no_path():
    _, _, belief = make_belief()
    goal = (5, 3)
    surrounding = [
        (gx, gy) for gy in range(2, 5) for gx in range(4, 7)
        if (gx, gy) != goal
    ]
    observations = []
    for index, (gx, gy) in enumerate(surrounding):
        observation = ray_at(gx, gy, 80.0)
        observation.row = 0
        observation.col = index
        observations.append(observation)
    belief.update_thermal_observations(
        np.full((1, len(observations)), 80.0), observations, 1.0
    )

    result = plan(belief, start=(0, 3), goal=goal)

    assert result.path == []
    assert "no traversable path" in result.reason


def test_thermal_camera_does_not_trace_cells_behind_obstacle():
    volume = np.full((2, 3, 8), 20.0)
    volume[:, :, 6] = 200.0
    obstacles = np.zeros_like(volume, dtype=bool)
    obstacles[:, :, 3] = True
    camera = ThermalCameraMLX90640(
        width=1, height=1, fov_h_deg=0.0, fov_v_deg=0.0,
        camera_height=0.25, front_offset=0.0, min_range=0.1,
        max_range=7.0, noise_std=0.0, accuracy_limit=0.0,
    )

    image, observations = camera.sense_3d(
        volume, robot_x=0.0, robot_y=1.0, robot_theta=0.0,
        xy_resolution=1.0, z_resolution=0.5,
        obstacle_volume=obstacles, return_observations=True,
    )
    sampled_x = {sample.grid_position[0] for sample in observations[0].ray_cells}

    assert observations[0].occluded
    assert max(sampled_x) == 3
    assert 6 not in sampled_x
    assert image.shape == (1, 1)


def test_replanned_path_replaces_old_path_for_motion():
    grid, config, belief = make_belief(robot_speed=1.0)
    follower = ReplannablePathFollower(grid, config)
    state = RobotState(0.0, 0.0, math.pi / 2.0)
    follower.set_path([(0, 0), (1, 0)])
    follower.set_path([(0, 0), (0, 1)])

    moved, _ = follower.update(state, 0.5, belief.final_cost_map)

    assert moved > 0.0
    assert state.y > 0.0
    assert math.isclose(state.x, 0.0, abs_tol=1e-9)
    assert follower.grid_path == [(0, 0), (0, 1)]


def test_planner_interface_contains_no_ground_truth_argument():
    parameters = inspect.signature(weighted_a_star_with_escape).parameters

    assert set(parameters) == {
        "cost_map", "start", "goal", "static_obstacle_map"
    }


def test_legacy_thermal_api_still_returns_only_image():
    camera = ThermalCameraMLX90640(
        width=2, height=2, noise_std=0.0, accuracy_limit=0.0
    )
    image = camera.sense_3d(
        np.full((2, 3, 3), 20.0), 1.0, 1.0, 0.0,
        xy_resolution=1.0, z_resolution=0.5,
    )

    assert isinstance(image, np.ndarray)
    assert image.shape == (2, 2)


def test_belief_uses_raw_pixel_array_not_ray_sample_or_display_color():
    _, _, belief = make_belief()
    observation = ray_at(2, 2, 999.0)
    raw_temperature_celsius = np.array([[42.5]])

    belief.update_thermal_observations(
        raw_temperature_celsius, [observation], 1.0
    )

    assert belief.temperature_belief_map[2, 2] == 42.5
    assert belief.temperature_belief_map[2, 2] != 999.0
