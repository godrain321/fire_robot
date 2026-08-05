"""Stage-2 evacuation using only sensor-updated robot belief for planning."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
import time

import numpy as np
import yaml

from human_detection_sim import SimpleHumanDetector
from mapping.fire_costmap import load_factory_geometry
from mapping.grid_map import GridMap
from mapping.partial_costmap import (
    PartialCostmapConfig,
    PartialFireCostmap,
    path_has_new_block,
)
from planner.a_star import weighted_a_star_with_escape
from robot.path_follower import RobotState, ReplannablePathFollower
from sensors.mq135_sensor import MQ135Config, MQ135Sensor
from sensors.thermal_camera import ThermalCameraMLX90640
from simulation.ground_truth import FDSGroundTruthEnvironment
from visualization.partial_costmap_viewers import (
    MatplotlibThermalViewer,
    PygameSimulationViewer,
    show_debug_costmaps,
)


@dataclass
class SimulationMetrics:
    replan_reasons: Counter = field(default_factory=Counter)
    replan_count: int = 0
    no_path_count: int = 0
    astar_time: float = 0.0
    first_path_cost: float | None = None
    final_path_cost: float | None = None
    travelled_distance: float = 0.0
    threshold_entries: int = 0
    observed_path_temperatures: list[float] = field(default_factory=list)
    observed_path_co: list[float] = field(default_factory=list)
    ground_truth_temperatures: list[float] = field(default_factory=list)
    ground_truth_co: list[float] = field(default_factory=list)
    detected_victim: str | None = None
    selected_exit: str | None = None


def _finite_stats(values: list[float]) -> tuple[str, str]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "N/A", "N/A"
    return f"{finite.max():.3f}", f"{finite.mean():.3f}"


def _combine_updates(*updates):
    changed = set()
    newly_blocked = set()
    for update in updates:
        changed.update(update.changed_cells)
        newly_blocked.update(update.newly_blocked_cells)
    return changed, newly_blocked


def _viewer_snapshot(
    fds_time, thermal, co_text, metrics, last_replan_reason, status,
):
    """Create display-only scalar state without feeding it back to simulation."""
    path_cost = (
        "N/A" if metrics.final_path_cost is None
        else f"{metrics.final_path_cost:.3f}"
    )
    return {
        "fds_time": fds_time,
        "thermal_min": float(np.nanmin(thermal)),
        "thermal_max": float(np.nanmax(thermal)),
        "co_text": co_text,
        "replan_count": metrics.replan_count,
        "last_replan_reason": last_replan_reason,
        "observed_ratio": 0.0,  # Filled by the caller's belief in draw().
        "path_cost": path_cost,
        "status": status,
    }


def _point(mapping) -> tuple[float, float]:
    return float(mapping["x"]), float(mapping["y"])


def _validate_free_point(name, point, grid_map) -> None:
    node = grid_map.world_to_grid(*point)
    if not grid_map.in_bounds(node):
        raise ValueError(f"{name} is outside the planner map: world={point}, grid={node}")
    if grid_map.is_blocked(node):
        raise ValueError(f"{name} is inside an inflated static obstacle: world={point}, grid={node}")


def _select_safest_exit(state, exits, belief):
    """Choose the currently reachable exit approach with the lowest A* cost."""
    start = belief.grid_map.world_to_grid(state.x, state.y)
    candidates = []
    for exit_item in exits:
        goal_world = _point(exit_item["approach"])
        result = weighted_a_star_with_escape(
            belief.final_cost_map,
            start,
            belief.grid_map.world_to_grid(*goal_world),
            belief.static_obstacle_map,
        )
        if result.path:
            candidates.append((result.total_cost, exit_item["id"], goal_world))
    if not candidates:
        raise RuntimeError("no configured exit is reachable in the current partial costmap")
    return min(candidates, key=lambda item: (item[0], item[1]))


def run_simulation(args) -> tuple[bool, SimulationMetrics, PartialFireCostmap, float]:
    config = PartialCostmapConfig(
        grid_resolution=args.grid_resolution,
        temperature_weight=args.temperature_weight,
        co_weight=args.co_weight,
        unknown_penalty=args.unknown_penalty,
        replan_interval_seconds=args.replan_interval,
        sensor_update_interval_seconds=args.sensor_interval,
        selected_fds_start_time=args.fds_start_time,
        simulation_dt=args.dt,
        robot_speed=args.robot_speed,
        gas_update_radius=args.gas_update_radius,
        use_inflation=not args.no_inflation,
        inflation_radius=args.inflation_radius,
    )
    ground_truth = FDSGroundTruthEnvironment(
        args.fds_file, args.temperature_npz, args.fds_dir
    )
    mesh_xb, obstacles, holes = load_factory_geometry(args.fds_file)
    grid_map = GridMap(
        mesh_xb, obstacles, holes, config.grid_resolution,
        config.inflation_radius if config.use_inflation else 0.0,
    )
    static_map = np.asarray(grid_map.occupancy, dtype=bool)
    belief = PartialFireCostmap(grid_map, static_map, config)
    for name, point in (("robot_start", args.start), ("search_waypoint", args.search_waypoint)):
        _validate_free_point(name, point, grid_map)
    for human in args.humans:
        _validate_free_point(f"human {human['id']}", _point(human), grid_map)
    for exit_item in args.exits:
        _validate_free_point(
            f"exit approach {exit_item['id']}", _point(exit_item["approach"]), grid_map
        )

    state = RobotState(args.start[0], args.start[1], math.radians(args.start_theta))
    goal = tuple(args.search_waypoint)
    follower = ReplannablePathFollower(grid_map, config)
    thermal_camera = ThermalCameraMLX90640()
    # Stage 2 must be able to represent the specified 1600 ppm threshold.
    gas_sensor = MQ135Sensor(MQ135Config(
        max_ppm=2000.0, warning_ppm=1000.0,
        danger_ppm=1400.0, blocked_ppm=config.co_blocked,
    ))

    metrics = SimulationMetrics()
    human_detector = SimpleHumanDetector(args.human_detection_range)
    active_victim = None
    victim_reached = False
    selected_exit = None
    detected_ids: set[str] = set()
    trajectory = [(state.x, state.y)]
    latest_thermal = np.full((24, 32), 25.0)
    latest_co_text = "unknown"
    sim_elapsed = 0.0
    last_sensor_time = -math.inf
    last_replan_time = -math.inf
    last_replan_position = (state.x, state.y)
    status = "INITIALIZING"
    no_path_active = False
    previous_grid = grid_map.world_to_grid(state.x, state.y)
    latest_newly_observed_cells: set[tuple[int, int]] = set()
    last_replan_reason = "none"

    pygame_viewer = None
    thermal_viewer = None
    if not args.headless:
        pygame_viewer = PygameSimulationViewer(grid_map, config)
        if not args.no_thermal_window:
            try:
                thermal_viewer = MatplotlibThermalViewer()
            except (ImportError, AttributeError) as exc:
                print(
                    "WARNING: Matplotlib thermal window is unavailable; "
                    "continuing with the Pygame simulation only. "
                    f"Use --no-thermal-window to suppress this attempt. ({exc})"
                )
                thermal_viewer = None

    while sim_elapsed <= args.max_time:
        if pygame_viewer is not None:
            running, paused = pygame_viewer.process_events()
            if not running:
                status = "USER_QUIT"
                break
            if paused:
                continue
        fds_time = config.selected_fds_start_time + sim_elapsed
        newly_blocked = set()
        sensor_due = sim_elapsed - last_sensor_time >= config.sensor_update_interval_seconds - 1e-9
        if sensor_due:
            observed_before = belief.observed_mask.copy()
            latest_thermal, rays, _ = ground_truth.capture_thermal(
                thermal_camera, state, fds_time
            )
            thermal_update = belief.update_thermal_observations(
                latest_thermal, rays, fds_time
            )
            gas = ground_truth.measure_co(
                gas_sensor, state, fds_time,
                config.sensor_update_interval_seconds,
            )
            if gas.valid:
                co_update = belief.update_co_observation(
                    state.x, state.y, gas.reading.measured_ppm, fds_time
                )
                latest_co_text = f"{gas.reading.measured_ppm:.1f} ppm"
            else:
                from mapping.partial_costmap import BeliefUpdate
                co_update = BeliefUpdate(frozenset(), frozenset())
                latest_co_text = "invalid"
            _, newly_blocked = _combine_updates(
                thermal_update, co_update
            )
            newly_observed_mask = belief.observed_mask & ~observed_before
            latest_newly_observed_cells = {
                (gx, gy) for gy, gx in np.argwhere(newly_observed_mask)
            }
            last_sensor_time = sim_elapsed
            if thermal_viewer is not None:
                thermal_viewer.update(latest_thermal, fds_time)

        detections = human_detector.detect(
            robot_position=(state.x, state.y),
            humans=args.humans,
            obstacle_map=static_map,
            map_origin=(grid_map.x_min, grid_map.y_min),
            map_resolution=grid_map.resolution,
            use_line_of_sight=True,
        )
        detected_ids.update(item["id"] for item in detections)
        if active_victim is None and detections:
            active_victim = detections[0]
            metrics.detected_victim = active_victim["id"]
            goal = (active_victim["x"], active_victim["y"])
            follower.clear()
            status = f"APPROACHING {active_victim['id']}"
        if active_victim is not None and not victim_reached and math.hypot(
            state.x - active_victim["x"], state.y - active_victim["y"]
        ) <= args.victim_approach_distance:
            victim_reached = True
            _, exit_id, exit_goal = _select_safest_exit(state, args.exits, belief)
            selected_exit = exit_id
            metrics.selected_exit = exit_id
            goal = exit_goal
            follower.clear()
            status = f"EVACUATING VIA {exit_id}"

        remaining_path = follower.remaining_grid_path()
        replan_reason = None
        if not remaining_path:
            replan_reason = "initial_or_missing_path"
        elif path_has_new_block(remaining_path, newly_blocked):
            replan_reason = "new_risk_on_path"
        elif sim_elapsed - last_replan_time >= config.replan_interval_seconds - 1e-9:
            replan_reason = "periodic"
        elif config.replan_distance > 0.0 and math.hypot(
            state.x - last_replan_position[0], state.y - last_replan_position[1]
        ) >= config.replan_distance:
            replan_reason = "distance"

        if replan_reason is not None:
            start_grid = grid_map.world_to_grid(state.x, state.y)
            goal_grid = grid_map.world_to_grid(*goal)
            astar_started = time.perf_counter()
            # Only robot belief arrays enter the planner. Ground Truth is not an argument.
            result = weighted_a_star_with_escape(
                belief.final_cost_map, start_grid, goal_grid,
                belief.static_obstacle_map,
            )
            metrics.astar_time += time.perf_counter() - astar_started
            metrics.replan_count += 1
            metrics.replan_reasons[replan_reason] += 1
            last_replan_reason = replan_reason
            last_replan_time = sim_elapsed
            last_replan_position = (state.x, state.y)
            if result.path:
                follower.set_path(result.path, result.escape_path, goal_world=goal)
                metrics.final_path_cost = result.total_cost
                if metrics.first_path_cost is None:
                    metrics.first_path_cost = result.total_cost
                status = f"EVACUATING ({replan_reason})"
                no_path_active = False
            else:
                follower.clear()
                metrics.no_path_count += 1
                status = f"NO_PATH: {result.reason}"
                no_path_active = True

        moved, motion_status = follower.update(state, config.simulation_dt, belief.final_cost_map)
        metrics.travelled_distance += moved
        trajectory.append((state.x, state.y))

        current_grid = grid_map.world_to_grid(state.x, state.y)
        if grid_map.in_bounds(current_grid):
            gy, gx = current_grid[1], current_grid[0]
            if belief.temperature_observed_mask[gy, gx]:
                metrics.observed_path_temperatures.append(
                    float(belief.temperature_belief_map[gy, gx])
                )
            if belief.co_observed_mask[gy, gx]:
                metrics.observed_path_co.append(float(belief.co_belief_map[gy, gx]))

        exposure = ground_truth.evaluate_exposure(
            state.x, state.y, config.robot_height, fds_time
        )
        metrics.ground_truth_temperatures.append(exposure.temperature)
        metrics.ground_truth_co.append(exposure.co_ppm)
        if current_grid != previous_grid and (
            exposure.temperature >= config.temperature_blocked
            or (np.isfinite(exposure.co_ppm) and exposure.co_ppm >= config.co_blocked)
        ):
            metrics.threshold_entries += 1
        previous_grid = current_grid

        if victim_reached and selected_exit is not None and follower.goal_reached(state, goal):
            status = "EVACUATION_SUCCESS"
            if pygame_viewer is not None:
                pygame_viewer.draw(
                    belief, state, tuple(args.start), goal, follower, trajectory,
                    thermal_camera, latest_newly_observed_cells,
                    _viewer_snapshot(
                        fds_time, latest_thermal, latest_co_text, metrics,
                        last_replan_reason, status,
                    ),
                    args.humans, args.exits, detected_ids,
                )
                pygame_viewer.close()
            if thermal_viewer is not None:
                thermal_viewer.close()
            return True, metrics, belief, sim_elapsed

        if pygame_viewer is not None:
            pygame_viewer.draw(
                belief, state, tuple(args.start), goal, follower, trajectory,
                thermal_camera, latest_newly_observed_cells,
                _viewer_snapshot(
                    fds_time, latest_thermal, latest_co_text, metrics,
                    last_replan_reason,
                    status if no_path_active else motion_status,
                ),
                args.humans, args.exits, detected_ids,
            )
        sim_elapsed += config.simulation_dt

    if pygame_viewer is not None:
        pygame_viewer.close()
    if thermal_viewer is not None:
        thermal_viewer.close()
    return False, metrics, belief, sim_elapsed


def print_summary(success, metrics, belief, elapsed):
    temp_max, temp_mean = _finite_stats(metrics.observed_path_temperatures)
    co_max, co_mean = _finite_stats(metrics.observed_path_co)
    gt_temp_max, _ = _finite_stats(metrics.ground_truth_temperatures)
    gt_co_max, _ = _finite_stats(metrics.ground_truth_co)
    print("\n=== Partial costmap evacuation summary ===")
    print(f"Evacuation success: {success}")
    print(f"Simulation time: {elapsed:.3f} s")
    print(f"Travelled distance: {metrics.travelled_distance:.3f} m")
    print(f"Replan count: {metrics.replan_count}")
    print(f"Replan reasons: {dict(metrics.replan_reasons)}")
    print(f"First path cost: {metrics.first_path_cost}")
    print(f"Final path cost: {metrics.final_path_cost}")
    print(f"Observed cells: {int(belief.observed_mask.sum())}")
    print(f"Observed ratio: {belief.observed_mask.mean()*100:.3f}%")
    print(f"Thermal observed cells: {int(belief.temperature_observed_mask.sum())}")
    print(f"CO observed cells: {int(belief.co_observed_mask.sum())}")
    print(f"Observed path temperature max/mean: {temp_max} / {temp_mean} °C")
    print(f"Observed path CO max/mean: {co_max} / {co_mean} ppm")
    print(f"Ground Truth maximum exposure temperature: {gt_temp_max} °C")
    print(f"Ground Truth maximum exposure CO: {gt_co_max} ppm")
    print(f"Threshold-exceeding cell entries: {metrics.threshold_entries}")
    print(f"No-path occurrences: {metrics.no_path_count}")
    print(f"A* total time: {metrics.astar_time:.6f} s")
    print(f"Detected victim: {metrics.detected_victim}")
    print(f"Selected exit: {metrics.selected_exit}")
    average = metrics.astar_time / metrics.replan_count if metrics.replan_count else 0.0
    print(f"A* average time: {average:.6f} s")


def parse_args():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", type=Path, default=base / "config/evacuation.yaml")
    parser.add_argument("--fds-file", type=Path, default=None)
    parser.add_argument("--fds-dir", type=Path, default=base)
    parser.add_argument(
        "--temperature-npz", type=Path,
        default=base / "processed/fds_temperature_3d_timeseries.npz",
    )
    parser.add_argument("--start", type=float, nargs=2, default=None)
    parser.add_argument("--start-theta", type=float, default=None)
    parser.add_argument("--grid-resolution", type=float, default=None)
    parser.add_argument("--fds-start-time", type=float, default=0.0)
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--sensor-interval", type=float, default=0.25)
    parser.add_argument("--replan-interval", type=float, default=1.0)
    parser.add_argument("--robot-speed", type=float, default=0.45)
    parser.add_argument("--unknown-penalty", type=float, default=2.0)
    parser.add_argument("--temperature-weight", type=float, default=8.0)
    parser.add_argument("--co-weight", type=float, default=8.0)
    parser.add_argument("--gas-update-radius", type=float, default=0.0)
    parser.add_argument("--inflation-radius", type=float, default=None)
    parser.add_argument("--no-inflation", action="store_true")
    parser.add_argument(
        "--headless", "--no-show", dest="headless", action="store_true",
        help="run simulation and tests without Pygame or Matplotlib viewers",
    )
    parser.add_argument(
        "--no-thermal-window", action="store_true",
        help="keep the Pygame main viewer but disable the thermal Matplotlib window",
    )
    parser.add_argument(
        "--debug-costmap-plots", action="store_true",
        help="show static Matplotlib cost-layer plots after the run",
    )
    return parser.parse_args()


def apply_scenario_config(args):
    """Resolve all v3 paths and mission positions from one explicit YAML file."""
    config_path = args.scenario_config.resolve()
    scenario = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent.parent
    args.fds_file = args.fds_file or base / scenario["fds_file"]
    args.temperature_npz = (
        args.temperature_npz
        if args.temperature_npz != Path(__file__).resolve().parent / "processed/fds_temperature_3d_timeseries.npz"
        else base / scenario["temperature_npz"]
    )
    args.start = tuple(args.start or _point(scenario["robot_start"]))
    args.start_theta = (
        float(scenario["robot_start"]["yaw_deg"])
        if args.start_theta is None else args.start_theta
    )
    args.grid_resolution = float(
        args.grid_resolution or scenario["planner"]["grid_resolution_m"]
    )
    args.inflation_radius = float(
        scenario["planner"]["inflation_radius_m"]
        if args.inflation_radius is None else args.inflation_radius
    )
    args.humans = list(scenario["humans"])
    args.exits = list(scenario["exits"])
    args.search_waypoint = _point(scenario["search_waypoint"])
    args.human_detection_range = float(scenario["human_detection_range_m"])
    args.victim_approach_distance = float(scenario["victim_approach_distance_m"])
    return args


def main() -> int:
    args = apply_scenario_config(parse_args())
    success, metrics, belief, elapsed = run_simulation(args)
    print_summary(success, metrics, belief, elapsed)
    if args.debug_costmap_plots:
        show_debug_costmaps(belief)
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
