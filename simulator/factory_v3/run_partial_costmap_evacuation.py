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
from mission.mission_manager import MissionEvent, MissionManager, MissionState
from navigation.return_path_planner import ReturnPathConfig, ReturnPathPlanner
from navigation.evacuation_strategy_selector import (
    EvacuationRouteSelectionConfig, EvacuationStrategy,
    EvacuationStrategySelector, HazardKnowledgeState, HazardKnowledgeTracker,
    PathValidationConfig, ReplanningConfig,
)
from navigation.travel_history import TravelHistory, TravelHistoryConfig
from navigation.path_simplifier import (
    PathSimplificationConfig, SafePathSimplifier,
)
from planner.a_star import weighted_a_star_with_escape
from planner.evacuation_planner import EvacuationPlanner, ExitSelectionConfig
from planner.exit_evaluator import ExitEvaluationConfig, ExitEvaluator
from robot.path_follower import RobotState, ReplannablePathFollower
from sensors.mq135_sensor import MQ135Config, MQ135Sensor
from sensors.thermal_camera import ThermalCameraMLX90640
from simulation.ground_truth import FDSGroundTruthEnvironment
from visualization.partial_costmap_viewers import (
    MatplotlibThermalViewer,
    PygameSimulationViewer,
    show_debug_costmaps,
)
from world.entities import ExitStatus, VictimStatus
from world.world_state import WorldState


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
    mission_state, navigation_mode="NORMAL", evacuation_plan=None,
    route_decision=None, route_failure=None, path_simplification=None,
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
        "mission_state": mission_state.name,
        "navigation_mode": navigation_mode,
        "hazard_knowledge": (
            "UNDECIDED" if route_decision is None
            else route_decision.hazard_knowledge.state.name
        ),
        "evacuation_strategy": (
            "UNDECIDED" if route_decision is None
            else route_decision.strategy.name
        ),
        "route_failure": route_failure or "none",
        "path_simplification": (
            "N/A" if path_simplification is None else
            f"{path_simplification.original_point_count}/"
            f"{path_simplification.corner_point_count}/"
            f"{path_simplification.simplified_point_count} points, "
            f"{path_simplification.reduction_ratio * 100:.1f}% reduced, "
            f"length {path_simplification.original_length_m:.2f}->"
            f"{path_simplification.simplified_length_m:.2f} m, "
            f"risk {path_simplification.original_risk_cost:.2f}->"
            f"{path_simplification.simplified_risk_cost:.2f}, "
            f"fallback={path_simplification.fallback_used}"
        ),
        "exit_plan": (
            "N/A" if evacuation_plan is None or not evacuation_plan.success
            else (
                f"{evacuation_plan.selected_exit_id}: "
                f"{evacuation_plan.selected_evaluation.path_length_m:.2f} m, "
                f"risk {evacuation_plan.selected_evaluation.accumulated_risk_cost:.3f}"
            )
        ),
    }


def _point(mapping) -> tuple[float, float]:
    return float(mapping["x"]), float(mapping["y"])


def _validate_free_point(name, point, grid_map) -> None:
    node = grid_map.world_to_grid(*point)
    if not grid_map.in_bounds(node):
        raise ValueError(f"{name} is outside the planner map: world={point}, grid={node}")
    if grid_map.is_blocked(node):
        raise ValueError(f"{name} is inside an inflated static obstacle: world={point}, grid={node}")


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
    mission = MissionManager(
        victim_reached_distance_m=args.victim_approach_distance,
        exit_reached_distance_m=args.exit_reached_distance,
        victim_wait_timeout_s=args.victim_wait_timeout,
        max_replan_count=args.max_mission_replans,
    )
    print(f"Mission state: {mission.current_state.name}")
    mesh_xb, obstacles, holes = load_factory_geometry(args.fds_file)
    grid_map = GridMap(
        mesh_xb, obstacles, holes, config.grid_resolution,
        config.inflation_radius if config.use_inflation else 0.0,
    )
    static_map = np.asarray(grid_map.occupancy, dtype=bool)
    belief = PartialFireCostmap(grid_map, static_map, config)
    world = WorldState.from_scenario(args.scenario, grid_map, config)
    # Existing detector and viewer APIs remain dictionary-based adapters.
    args.humans = world.legacy_humans()
    args.exits = world.legacy_exits()
    travel_history = TravelHistory(
        world.map_metadata, TravelHistoryConfig.from_mapping(args.travel_history_config)
    )
    return_planner = ReturnPathPlanner(
        world.map_metadata, ReturnPathConfig.from_mapping(args.return_path_config)
    )
    exit_evaluator = ExitEvaluator(
        world.map_metadata,
        ExitEvaluationConfig.from_mapping(args.exit_evaluation_config),
        temperature_blocked_c=config.temperature_blocked,
        co_blocked_ppm=config.co_blocked,
        base_cost=config.base_cost,
    )
    evacuation_planner = EvacuationPlanner(
        exit_evaluator,
        ExitSelectionConfig.from_mapping(args.exit_selection_config),
    )
    hazard_tracker = HazardKnowledgeTracker(
        temperature_elevated_c=config.temperature_safe,
        co_elevated_ppm=config.co_safe,
    )
    strategy_selector = EvacuationStrategySelector(
        world.map_metadata, hazard_tracker, return_planner, evacuation_planner,
        EvacuationRouteSelectionConfig.from_mapping(
            args.evacuation_route_selection_config
        ),
    )
    path_validation_config = PathValidationConfig.from_mapping(
        args.path_validation_config
    )
    replanning_config = ReplanningConfig.from_mapping(args.replanning_config)
    path_simplifier = SafePathSimplifier(
        world.map_metadata,
        PathSimplificationConfig.from_mapping(args.path_simplification_config),
    )
    world.attach_travel_history(travel_history)
    world.set_mission_entry(args.mission_entry_id, args.start)
    world.record_robot_position(args.start, sim_time=0.0)
    map_x = np.asarray([
        world.map_metadata.grid_to_world(col, 0)[0]
        for col in range(world.map_metadata.width)
    ])
    map_y = np.asarray([
        world.map_metadata.grid_to_world(0, row)[1]
        for row in range(world.map_metadata.height)
    ])
    for name, point in (("robot_start", args.start), ("search_waypoint", args.search_waypoint)):
        _validate_free_point(name, point, grid_map)
    for human in args.humans:
        _validate_free_point(f"human {human['id']}", _point(human), grid_map)
    for exit_item in world.exits.values():
        if exit_item.approach_position_world is not None:
            _validate_free_point(
                f"exit approach {exit_item.exit_id}",
                exit_item.approach_position_world,
                grid_map,
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
    returning_by_history = False
    returning_to_entrance = False
    blocked_return_grid = None
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
    last_route_environment_revision = world.environment_revision

    def activate_simplified_path(
        original_path_grid, *, goal_world, escape_path=(),
    ):
        """Validate, simplify, store, then atomically activate waypoints."""
        path = tuple((int(col), int(row)) for col, row in original_path_grid)
        current_grid = grid_map.world_to_grid(state.x, state.y)
        if path and path[0] != current_grid:
            path = (current_grid,) + path
        result = path_simplifier.simplify(
            path, costmap=belief.final_cost_map,
            static_obstacle_map=belief.static_obstacle_map,
            dynamic_obstacle_map=world.dynamic_obstacle_mask(),
            estimated_fire_map=world.estimated_fire_map,
            costmap_revision=belief.revision,
            start_world=(state.x, state.y), goal_world=goal_world,
        )
        if not result.success:
            follower.clear()
            world.final_route_failure_reason = result.failure_reason
            return None
        world.record_path_simplification(result)
        simplified_escape = tuple(
            node for node in result.simplified_path_grid if node in set(escape_path)
        )
        follower.set_path(
            result.simplified_path_grid, simplified_escape,
            goal_world=goal_world, world_path=result.waypoints_world,
        )
        return result

    def activate_replacement_route(replacement) -> bool:
        """Activate a Stage-6 replacement only after Stage-7 validation."""
        nonlocal goal, no_path_active, returning_to_entrance
        nonlocal selected_exit, status
        goal = replacement.target_position_world
        simplified = activate_simplified_path(
            replacement.path_grid, goal_world=goal
        )
        if simplified is None:
            mission.handle_event(
                MissionEvent.NO_SAFE_ROUTE_FOUND,
                sim_time=sim_elapsed,
                reason=world.final_route_failure_reason,
            )
            status = "NO_SAFE_ROUTE: path simplification failed"
            no_path_active = True
            return False
        world.set_active_route_decision(replacement)
        no_path_active = False
        if replacement.strategy is EvacuationStrategy.REPLAN_TO_ENTRANCE:
            mission.handle_event(
                MissionEvent.ENTRANCE_ROUTE_CREATED,
                path=simplified.simplified_path_grid,
                sim_time=sim_elapsed,
            )
            returning_to_entrance = True
            selected_exit = None
            status = "RETURNING_TO_ENTRANCE_BY_ASTAR"
        else:
            mission.handle_event(
                MissionEvent.EXIT_EVALUATION_REQUESTED,
                sim_time=sim_elapsed,
            )
            mission.handle_event(
                MissionEvent.SAFE_EXIT_SELECTED,
                exit_id=replacement.target_exit_id,
                exit_position=goal,
                selection_reason=replacement.reasons[0],
                sim_time=sim_elapsed,
            )
            mission.handle_event(
                MissionEvent.EVACUATION_PLAN_CREATED,
                exit_id=replacement.target_exit_id,
                exit_position=goal,
                path=simplified.simplified_path_grid,
                sim_time=sim_elapsed,
            )
            selected_exit = replacement.target_exit_id
            metrics.selected_exit = selected_exit
            returning_to_entrance = False
            status = f"EVACUATING VIA {selected_exit}"
        return True
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
        world.set_simulation_time(sim_elapsed)
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
            world.estimated_fire_map.sync_from_belief(belief)
            world.update_costmap_revision(belief.revision)
            # Capture hazard knowledge at observation time so later normal
            # readings do not erase evidence seen earlier in the mission.
            world.hazard_knowledge_decision = hazard_tracker.evaluate(
                world.estimated_fire_map, evaluated_at=sim_elapsed
            )
            if (
                returning_by_history
                and return_planner.config.validate_during_return
                and world.active_return_plan is not None
            ):
                rechecked = None
                simplified_recheck = None
                if world.active_path_simplification is not None:
                    remaining_simplified = (
                        world.active_path_simplification.simplified_path_grid[
                            max(0, follower.waypoint_index - 1):
                        ]
                    )
                    if remaining_simplified:
                        simplified_recheck = path_simplifier.validate_path(
                            remaining_simplified,
                            costmap=belief.final_cost_map,
                            static_obstacle_map=belief.static_obstacle_map,
                            dynamic_obstacle_map=world.dynamic_obstacle_mask(),
                            estimated_fire_map=world.estimated_fire_map,
                        )
                else:
                    rechecked = return_planner.validate_plan(
                        world.active_return_plan,
                        cost_map=belief.final_cost_map,
                        static_obstacle_map=belief.static_obstacle_map,
                        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
                        estimated_fire_map=world.estimated_fire_map,
                        current_time=sim_elapsed,
                        start_index=max(0, follower.waypoint_index - 1),
                        current_position_world=(state.x, state.y),
                    )
                if (rechecked is not None and not rechecked.success) or (
                    simplified_recheck is not None and not simplified_recheck.safe
                ):
                    invalid_reason = (
                        rechecked.failure_reason.value
                        if rechecked is not None and not rechecked.success else
                        simplified_recheck.rejection_reasons[0].value
                    )
                    invalid_grid = (
                        rechecked.blocked_grid
                        if rechecked is not None and not rechecked.success
                        else simplified_recheck.first_rejected_cell
                    )
                    # Safety order is intentional: stop and deactivate first,
                    # then evaluate a replacement route from the current pose.
                    mission.handle_event(
                        MissionEvent.RETURN_PATH_INVALIDATED,
                        sim_time=sim_elapsed,
                        reason=invalid_reason,
                    )
                    follower.clear()
                    world.clear_active_return_plan()
                    world.invalidate_active_route(
                        invalid_reason, invalid_grid
                    )
                    returning_by_history = False
                    blocked_return_grid = invalid_grid
                    status = f"RETURN_BLOCKED: {invalid_reason}"
                    no_path_active = True
                    if (
                        replanning_config.enabled
                        and world.route_replan_count
                        < replanning_config.max_replan_attempts
                    ):
                        mission.handle_event(
                            MissionEvent.REPLAN_REQUESTED,
                            sim_time=sim_elapsed,
                            reason=invalid_reason,
                        )
                        world.route_replan_count += 1
                        replacement = strategy_selector.replan_after_return_invalidated(
                            world_state=world,
                            current_position_world=(state.x, state.y),
                            cost_map=belief.final_cost_map,
                            costmap_revision=belief.revision,
                            created_at=sim_elapsed,
                        )
                        world.hazard_knowledge_decision = replacement.hazard_knowledge
                        if replacement.success:
                            activate_replacement_route(replacement)
                        else:
                            mission.handle_event(
                                MissionEvent.NO_SAFE_ROUTE_FOUND,
                                sim_time=sim_elapsed,
                                reason=replacement.failure_reason.value,
                            )
                            world.final_route_failure_reason = replacement.failure_reason.value
                            status = f"NO_SAFE_ROUTE: {replacement.failure_reason.value}"
            if (
                not returning_by_history
                and world.active_route_decision is not None
                and world.active_route_valid
                and (
                    (
                        path_validation_config.validate_on_costmap_revision
                        or path_validation_config.validate_on_hazard_update
                    )
                    and world.active_route_costmap_revision != belief.revision
                    or path_validation_config.validate_on_dynamic_obstacle_event
                    and last_route_environment_revision != world.environment_revision
                )
            ):
                active_grid_path = (
                    world.active_path_simplification.simplified_path_grid
                    if world.active_path_simplification is not None
                    else world.active_route_decision.path_grid
                )
                remaining_active_path = active_grid_path[
                    max(0, follower.waypoint_index - 1):
                ]
                route_validation = path_simplifier.validate_path(
                    remaining_active_path,
                    costmap=belief.final_cost_map,
                    static_obstacle_map=belief.static_obstacle_map,
                    dynamic_obstacle_map=world.dynamic_obstacle_mask(),
                    estimated_fire_map=world.estimated_fire_map,
                )
                valid = route_validation.safe
                blocked_grid = route_validation.first_rejected_cell
                invalid_reason = (
                    None if valid else route_validation.rejection_reasons[0].value
                )
                world.record_route_validation(
                    sim_time=sim_elapsed, costmap_revision=belief.revision
                )
                last_route_environment_revision = world.environment_revision
                if not valid:
                    # Movement is stopped before any planner is invoked.
                    follower.clear()
                    world.invalidate_active_route(invalid_reason, blocked_grid)
                    world.clear_active_evacuation_plan()
                    world.clear_active_return_plan()
                    mission.handle_event(
                        MissionEvent.ACTIVE_PATH_INVALIDATED,
                        sim_time=sim_elapsed, reason=invalid_reason,
                    )
                    blocked_return_grid = blocked_grid
                    no_path_active = True
                    status = f"PATH_INVALIDATED: {invalid_reason}"
                    if (
                        replanning_config.enabled
                        and world.route_replan_count
                        < replanning_config.max_replan_attempts
                    ):
                        mission.handle_event(
                            MissionEvent.REPLAN_REQUESTED,
                            sim_time=sim_elapsed, reason=invalid_reason,
                        )
                        world.route_replan_count += 1
                        if returning_to_entrance:
                            replacement = strategy_selector.replan_after_return_invalidated(
                                world_state=world,
                                current_position_world=(state.x, state.y),
                                cost_map=belief.final_cost_map,
                                costmap_revision=belief.revision,
                                created_at=sim_elapsed,
                            )
                        else:
                            replacement = strategy_selector.replan_to_safe_exit(
                                world_state=world,
                                current_position_world=(state.x, state.y),
                                cost_map=belief.final_cost_map,
                                costmap_revision=belief.revision,
                                created_at=sim_elapsed,
                            )
                        if replacement.success:
                            activate_replacement_route(replacement)
                        else:
                            mission.handle_event(
                                MissionEvent.NO_SAFE_ROUTE_FOUND,
                                sim_time=sim_elapsed,
                                reason=replacement.failure_reason.value,
                            )
                            world.final_route_failure_reason = replacement.failure_reason.value
                            status = f"NO_SAFE_ROUTE: {replacement.failure_reason.value}"
            if (
                not returning_by_history
                and world.active_route_decision is None
                and world.active_path_simplification is not None
                and (
                    world.active_path_simplification.used_costmap_revision
                    != belief.revision
                    or last_route_environment_revision
                    != world.environment_revision
                )
            ):
                # Search/approach paths do not yet have a Mission route object,
                # but their non-adjacent shortcut segments still require the
                # same supercover revalidation after sensor updates.
                remaining_search_path = (
                    world.active_path_simplification.simplified_path_grid[
                        max(0, follower.waypoint_index - 1):
                    ]
                )
                if remaining_search_path:
                    search_validation = path_simplifier.validate_path(
                        remaining_search_path,
                        costmap=belief.final_cost_map,
                        static_obstacle_map=belief.static_obstacle_map,
                        dynamic_obstacle_map=world.dynamic_obstacle_mask(),
                        estimated_fire_map=world.estimated_fire_map,
                    )
                    last_route_environment_revision = world.environment_revision
                    if not search_validation.safe:
                        follower.clear()
                        blocked_return_grid = search_validation.first_rejected_cell
                        status = (
                            "SEARCH_PATH_INVALIDATED: "
                            + search_validation.rejection_reasons[0].value
                        )
                        no_path_active = True
                        world.clear_path_simplification()
            gt_temperature, gt_co, gt_time = ground_truth.sample_map_yx(
                map_x, map_y, config.robot_height, fds_time
            )
            gt_observed = np.isfinite(gt_temperature) | np.isfinite(gt_co)
            gt_last_time = np.full(gt_temperature.shape, np.nan, dtype=float)
            gt_last_time[gt_observed] = gt_time
            world.ground_truth_fire_map.replace_layers(
                gt_temperature, gt_co, gt_observed, gt_last_time
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
            victim = world.get_victim(active_victim["id"])
            world.update_victim_status(
                victim.victim_id, VictimStatus.DETECTED,
                sim_time=sim_elapsed,
                distance_from_robot_m=active_victim["distance"],
            )
            world.update_victim_status(
                victim.victim_id, VictimStatus.APPROACHING, sim_time=sim_elapsed
            )
            transition = mission.handle_event(
                MissionEvent.VICTIM_DETECTED,
                victim_id=active_victim["id"],
                victim_position=(active_victim["x"], active_victim["y"]),
                sim_time=sim_elapsed,
            )
            print(
                f"Mission: {transition.previous_state.name} --{transition.event.name}--> "
                f"{transition.next_state.name}: {transition.reason}"
            )
            metrics.detected_victim = active_victim["id"]
            goal = (active_victim["x"], active_victim["y"])
            follower.clear()
            status = f"APPROACHING {active_victim['id']}"
        if active_victim is not None and not victim_reached and math.hypot(
            state.x - active_victim["x"], state.y - active_victim["y"]
        ) <= args.victim_approach_distance:
            victim_reached = True
            victim = world.get_victim(active_victim["id"])
            world.update_victim_status(
                victim.victim_id, VictimStatus.REACHED, sim_time=sim_elapsed
            )
            mission.handle_event(MissionEvent.VICTIM_REACHED, sim_time=sim_elapsed)
            print("대피 경로를 생성합니다. 물수건으로 코와 입을 막고 저를 따라오십시오")
            mission.handle_event(
                MissionEvent.ANNOUNCEMENT_FINISHED, sim_time=sim_elapsed
            )
            world.update_victim_status(
                victim.victim_id, VictimStatus.WAITING, sim_time=sim_elapsed
            )
            # TODO: Replace this immediate event with tracked victim motion data.
            mission.handle_event(
                MissionEvent.VICTIM_STARTED_MOVING,
                sim_time=sim_elapsed,
                reason="legacy simulation assumes the victim follows immediately",
            )
            world.update_victim_status(
                victim.victim_id, VictimStatus.MOVABLE, sim_time=sim_elapsed
            )
            mission.handle_event(
                MissionEvent.VICTIM_READY_FOR_EVACUATION,
                victim_id=victim.victim_id, sim_time=sim_elapsed,
            )
            route_decision = strategy_selector.select_initial_route(
                world_state=world, travel_history=travel_history,
                start_position_world=(state.x, state.y),
                victim_position_world=victim.position_world,
                cost_map=belief.final_cost_map,
                costmap_revision=belief.revision,
                created_at=sim_elapsed,
            )
            world.hazard_knowledge_decision = route_decision.hazard_knowledge
            if route_decision.hazard_knowledge.state is HazardKnowledgeState.FIRE_INFORMATION_AVAILABLE:
                mission.handle_event(
                    MissionEvent.HAZARD_INFORMATION_AVAILABLE,
                    sim_time=sim_elapsed,
                )
            else:
                mission.handle_event(
                    MissionEvent.NO_HAZARD_INFORMATION, sim_time=sim_elapsed,
                )
            if route_decision.success:
                if route_decision.strategy is EvacuationStrategy.SAFE_EXIT_PLANNING:
                    evacuation_plan = route_decision.evacuation_plan
                    selected = evacuation_plan.selected_evaluation
                    selected_exit = route_decision.target_exit_id
                    metrics.selected_exit = selected_exit
                    goal = route_decision.target_position_world
                    mission.handle_event(
                        MissionEvent.SAFE_EXIT_SELECTED,
                        exit_id=selected_exit, exit_position=goal,
                        selection_reason=evacuation_plan.selection_reason,
                        sim_time=sim_elapsed,
                    )
                    world.update_exit_status(
                        selected_exit, ExitStatus.USABLE,
                        sim_time=sim_elapsed,
                        temperature_c=selected.exit_temperature_c
                        if selected.exit_temperature_c is not None else np.nan,
                        co_ppm=selected.exit_co_ppm
                        if selected.exit_co_ppm is not None else np.nan,
                        path_cost=selected.accumulated_risk_cost,
                    )
                    simplified = activate_simplified_path(
                        evacuation_plan.path_grid, goal_world=goal
                    )
                    if simplified is None:
                        mission.handle_event(
                            MissionEvent.PATH_PLANNING_FAILED,
                            sim_time=sim_elapsed,
                            reason=world.final_route_failure_reason,
                        )
                        status = "NO_PATH: simplification rejected original path"
                        no_path_active = True
                    else:
                        world.set_active_route_decision(route_decision)
                        world.update_victim_status(
                            victim.victim_id, VictimStatus.EVACUATING,
                            sim_time=sim_elapsed, assigned_exit_id=selected_exit,
                        )
                        mission.handle_event(
                            MissionEvent.EVACUATION_PLAN_CREATED,
                            exit_id=selected_exit,
                            exit_position=goal,
                            path=simplified.simplified_path_grid,
                            selection_reason=evacuation_plan.selection_reason,
                            sim_time=sim_elapsed,
                        )
                        last_replan_time = sim_elapsed
                        last_replan_position = (state.x, state.y)
                        status = f"EVACUATING VIA {selected_exit}"
                        no_path_active = False
                else:
                    return_plan = route_decision.return_plan
                    selected_exit = None
                    goal = route_decision.target_position_world
                    simplified = activate_simplified_path(
                        route_decision.path_grid, goal_world=goal
                    )
                    if simplified is None:
                        mission.handle_event(
                            MissionEvent.NO_SAFE_ROUTE_FOUND,
                            sim_time=sim_elapsed,
                            reason=world.final_route_failure_reason,
                        )
                        status = "NO_SAFE_ROUTE: history path simplification failed"
                        no_path_active = True
                    else:
                        world.set_active_route_decision(route_decision)
                        mission.handle_event(
                            MissionEvent.RETURN_PATH_CREATED,
                            path=simplified.simplified_path_grid,
                            sim_time=sim_elapsed,
                        )
                        world.set_active_return_plan(return_plan)
                        world.update_victim_status(
                            victim.victim_id, VictimStatus.EVACUATING,
                            sim_time=sim_elapsed,
                        )
                        returning_by_history = True
                        returning_to_entrance = False
                        status = "RETURNING_BY_HISTORY"
                        no_path_active = False
            else:
                selected_exit = None
                follower.clear()
                mission.handle_event(
                    MissionEvent.NO_SAFE_ROUTE_FOUND, sim_time=sim_elapsed,
                    reason=route_decision.failure_reason.value,
                )
                world.final_route_failure_reason = route_decision.failure_reason.value
                status = f"NO_SAFE_ROUTE: {route_decision.failure_reason.value}"
                no_path_active = True

        remaining_path = follower.remaining_grid_path()
        replan_reason = None
        if mission.current_state in (
            MissionState.NO_SAFE_EXIT,
            MissionState.RETURN_PATH_BLOCKED,
            MissionState.RETURN_FAILED,
            MissionState.NO_SAFE_ROUTE,
            MissionState.REPLANNING_TO_ENTRANCE,
        ) or returning_by_history:
            # TODO: A later sensor/costmap update may explicitly issue
            # RETRY_REQUESTED. Until then, do not plan back to the victim goal.
            replan_reason = None
        elif not remaining_path:
            replan_reason = "initial_or_missing_path"
        elif path_has_new_block(remaining_path, newly_blocked):
            replan_reason = "new_risk_on_path"
        elif world.active_route_decision is not None and world.active_route_valid:
            # Stage-6 routes are revalidated on explicit belief/environment
            # revisions. Do not replace them with unrelated periodic A* runs.
            replan_reason = None
        elif sim_elapsed - last_replan_time >= config.replan_interval_seconds - 1e-9:
            replan_reason = "periodic"
        elif config.replan_distance > 0.0 and math.hypot(
            state.x - last_replan_position[0], state.y - last_replan_position[1]
        ) >= config.replan_distance:
            replan_reason = "distance"

        if replan_reason is not None:
            if mission.current_state is MissionState.REPLAN:
                mission.handle_event(
                    MissionEvent.RETRY_REQUESTED,
                    sim_time=sim_elapsed,
                    reason=f"planner retry triggered by {replan_reason}",
                )
            start_grid = grid_map.world_to_grid(state.x, state.y)
            goal_grid = grid_map.world_to_grid(*goal)
            astar_started = time.perf_counter()
            # Only robot belief arrays enter the planner. Ground Truth is not an argument.
            planning_cost_map = belief.final_cost_map.copy()
            planning_cost_map[world.dynamic_obstacle_mask()] = np.inf
            result = weighted_a_star_with_escape(
                planning_cost_map, start_grid, goal_grid,
                belief.static_obstacle_map,
            )
            metrics.astar_time += time.perf_counter() - astar_started
            metrics.replan_count += 1
            metrics.replan_reasons[replan_reason] += 1
            last_replan_reason = replan_reason
            last_replan_time = sim_elapsed
            last_replan_position = (state.x, state.y)
            if result.path:
                simplified = activate_simplified_path(
                    result.path, escape_path=result.escape_path,
                    goal_world=goal,
                )
                if simplified is not None and mission.current_state is MissionState.PLAN_EVACUATION:
                    exit_position = goal if selected_exit is not None else None
                    mission.handle_event(
                        MissionEvent.PATH_PLANNED,
                        exit_id=selected_exit,
                        exit_position=exit_position,
                        path=simplified.simplified_path_grid,
                        sim_time=sim_elapsed,
                    )
                if simplified is not None:
                    metrics.final_path_cost = result.total_cost
                    if metrics.first_path_cost is None:
                        metrics.first_path_cost = result.total_cost
                    status = f"EVACUATING ({replan_reason})"
                    no_path_active = False
                else:
                    if mission.current_state is MissionState.PLAN_EVACUATION:
                        mission.handle_event(
                            MissionEvent.PATH_PLANNING_FAILED,
                            sim_time=sim_elapsed,
                            reason=world.final_route_failure_reason,
                        )
                    metrics.no_path_count += 1
                    status = "NO_PATH: unsafe A* result"
                    no_path_active = True
            else:
                follower.clear()
                if mission.current_state is MissionState.PLAN_EVACUATION:
                    mission.handle_event(
                        MissionEvent.PATH_PLANNING_FAILED,
                        sim_time=sim_elapsed,
                        reason=result.reason,
                        failed_exits={selected_exit: result.reason}
                        if selected_exit else None,
                    )
                metrics.no_path_count += 1
                status = f"NO_PATH: {result.reason}"
                no_path_active = True

        moved, motion_status = follower.update(state, config.simulation_dt, belief.final_cost_map)
        metrics.travelled_distance += moved
        trajectory.append((state.x, state.y))
        if moved > 0.0:
            world.record_robot_position(
                (state.x, state.y), sim_time=sim_elapsed,
                is_returning=(returning_by_history or returning_to_entrance),
            )

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
        if grid_map.in_bounds(current_grid):
            world.ground_truth_fire_map.update_cell(
                current_grid[0], current_grid[1],
                temperature_c=exposure.temperature,
                co_ppm=exposure.co_ppm,
                sim_time=fds_time,
            )
        metrics.ground_truth_temperatures.append(exposure.temperature)
        metrics.ground_truth_co.append(exposure.co_ppm)
        if current_grid != previous_grid and (
            exposure.temperature >= config.temperature_blocked
            or (np.isfinite(exposure.co_ppm) and exposure.co_ppm >= config.co_blocked)
        ):
            metrics.threshold_entries += 1
        previous_grid = current_grid

        if (
            victim_reached
            and (selected_exit is not None or returning_by_history or returning_to_entrance)
            and follower.goal_reached(state, goal)
        ):
            world.update_victim_status(
                active_victim["id"], VictimStatus.RESCUED,
                sim_time=sim_elapsed,
            )
            remaining_victims = bool(world.get_unrescued_victims())
            if returning_by_history or returning_to_entrance:
                mission.handle_event(
                    MissionEvent.RETURN_COMPLETED,
                    sim_time=sim_elapsed,
                    remaining_victims=remaining_victims,
                )
                world.clear_active_return_plan()
            else:
                mission.handle_event(
                    MissionEvent.EXIT_REACHED,
                    exit_id=selected_exit,
                    sim_time=sim_elapsed,
                    remaining_victims=remaining_victims,
                )
                world.clear_active_evacuation_plan()
            world.clear_active_route()
            status = "EVACUATION_SUCCESS"
            if pygame_viewer is not None:
                pygame_viewer.draw(
                    belief, state, tuple(args.start), goal, follower, trajectory,
                    thermal_camera, latest_newly_observed_cells,
                    _viewer_snapshot(
                        fds_time, latest_thermal, latest_co_text, metrics,
                        last_replan_reason, status, mission.current_state,
                        (
                            "HISTORY_RETURN" if returning_by_history
                            else "ENTRANCE_ASTAR" if returning_to_entrance
                            else "NORMAL"
                        ),
                        world.active_evacuation_plan,
                        world.active_route_decision,
                        world.final_route_failure_reason,
                        world.active_path_simplification,
                    ),
                    args.humans, args.exits, detected_ids,
                    travel_history.get_points_world(),
                    () if world.active_return_plan is None else world.active_return_plan.path_world[max(0, follower.waypoint_index - 1):],
                    blocked_return_grid,
                    tuple(world.latest_exit_evaluations.values()),
                    selected_exit,
                    world.active_path_simplification,
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
                    mission.current_state,
                    (
                        "HISTORY_RETURN" if returning_by_history
                        else "ENTRANCE_ASTAR" if returning_to_entrance
                        else "NORMAL"
                    ),
                    world.active_evacuation_plan,
                    world.active_route_decision,
                    world.final_route_failure_reason,
                    world.active_path_simplification,
                ),
                args.humans, args.exits, detected_ids,
                travel_history.get_points_world(),
                () if world.active_return_plan is None else world.active_return_plan.path_world[max(0, follower.waypoint_index - 1):],
                blocked_return_grid,
                tuple(world.latest_exit_evaluations.values()),
                selected_exit,
                world.active_path_simplification,
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
    args.scenario = scenario
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
    mission_config = scenario.get("mission", {})
    args.exit_reached_distance = float(
        mission_config.get("exit_reached_distance_m", 1.0)
    )
    args.victim_wait_timeout = float(
        mission_config.get("victim_wait_timeout_s", 10.0)
    )
    args.max_mission_replans = int(mission_config.get("max_replan_count", 5))
    args.travel_history_config = scenario.get("travel_history", {})
    args.return_path_config = scenario.get("return_path", {})
    args.exit_evaluation_config = scenario.get("exit_evaluation", {})
    args.exit_selection_config = scenario.get("exit_selection", {})
    args.evacuation_route_selection_config = scenario.get(
        "evacuation_route_selection", {}
    )
    args.path_validation_config = scenario.get("path_validation", {})
    args.replanning_config = scenario.get("replanning", {})
    args.path_simplification_config = scenario.get("path_simplification", {})
    args.mission_entry_id = str(scenario.get("mission_entry_id", "MISSION_ENTRY"))
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
