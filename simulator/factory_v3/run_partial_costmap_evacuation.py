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
from mapping.fire_costmap import (
    load_factory_geometry, obstacles_for_initial_robot_map,
)
from mapping.grid_map import GridMap
from mapping.fire_localization import FireLocalizationConfig, FireLocalizer
from mapping.dynamic_obstacle_mapping import (
    DynamicObstacleMapper, DynamicObstacleMappingConfig,
)
from mapping.partial_costmap import (
    PartialCostmapConfig,
    PartialFireCostmap,
    path_has_new_block,
)
from mission.mission_manager import MissionEvent, MissionManager, MissionState
from navigation.return_path_planner import ReturnPathConfig, ReturnPathPlanner
from navigation.evacuation_strategy_selector import (
    EvacuationRouteSelectionConfig, EvacuationStrategy,
    EvacuationStrategySelector, HazardKnowledgeConfig, HazardKnowledgeState,
    HazardKnowledgeTracker, PathValidationConfig, ReplanningConfig,
)
from navigation.travel_history import TravelHistory, TravelHistoryConfig
from navigation.path_simplifier import PathSimplificationConfig, SafePathSimplifier
from navigation.exit_switching import (
    ExitSwitchingConfig, RouteCostTrendMonitor, current_direction_world,
    evaluate_path_cost,
)
from navigation.event_replanning import (
    EventReplanningConfig, EventReplanningPolicy, ReplanReason,
)
from navigation.exploration_manager import (
    ExplorationConfig, ExplorationManager, ExplorationPhase,
)
from navigation.initial_advance import InitialAdvanceConfig
from navigation.external_waypoint_motion import (
    ExternalWaypointFollower, ExternalWaypointMotionConfig,
    load_external_waypoints,
)
from navigation.victim_following import (
    FollowState, VictimFollowingConfig, VictimFollowingController,
    evacuation_success_ready,
)
from navigation.victim_scripted_motion import (
    ScriptedVictimMotionConfig, ScriptedVictimMotionController,
)
from navigation.exit_blockage import ExitBlockageConfig, ExitBlockageEvaluator
from planner.a_star import weighted_a_star_with_escape
from planner.evacuation_planner import EvacuationPlanner, ExitSelectionConfig
from planner.exit_evaluator import (
    ExitEvaluationConfig, ExitEvaluator, ExitRejectionReason,
)
from robot.path_follower import RobotState, ReplannablePathFollower
from sensors.mq135_sensor import MQ135Config, MQ135Sensor
from sensors.thermal_camera import ThermalCameraMLX90640
from simulation.ground_truth import FDSGroundTruthEnvironment
from visualization.partial_costmap_viewers import (
    MapOverlayConfig, MatplotlibThermalViewer, PerceptionMapDisplayConfig,
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
    actual_path_world: list[tuple[float, float]] = field(default_factory=list)


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
        temperature_power=args.temperature_power,
        co_weight=args.co_weight,
        co_power=args.co_power,
        unknown_penalty=args.unknown_penalty,
        replan_interval_seconds=args.replan_interval,
        sensor_update_interval_seconds=args.sensor_interval,
        selected_fds_start_time=args.fds_start_time,
        simulation_dt=args.dt,
        robot_speed=args.robot_speed,
        robot_angular_speed=math.radians(args.robot_angular_speed_deg),
        render_fps=args.render_fps,
        gas_update_radius=args.gas_update_radius,
        use_inflation=not args.no_inflation,
        inflation_radius=args.inflation_radius,
    )
    ground_truth = FDSGroundTruthEnvironment(
        args.fds_file, args.temperature_npz, args.fds_dir, args.co_npz
    )
    mission = MissionManager(
        victim_reached_distance_m=args.victim_approach_distance,
        exit_reached_distance_m=args.exit_reached_distance,
        victim_wait_timeout_s=args.victim_wait_timeout,
        max_replan_count=args.max_mission_replans,
    )
    print(f"Mission state: {mission.current_state.name}")
    mesh_xb, obstacles, holes = load_factory_geometry(args.fds_file)
    planner_obstacles = obstacles_for_initial_robot_map(
        obstacles, args.scenario
    )
    grid_map = GridMap(
        mesh_xb, planner_obstacles, holes, config.grid_resolution,
        config.inflation_radius if config.use_inflation else 0.0,
    )
    static_map = np.asarray(grid_map.occupancy, dtype=bool)
    display_grid_map = GridMap(
        mesh_xb, planner_obstacles, holes, config.grid_resolution, 0.0,
        include_lower_obstacle_boundary=False,
    )
    display_static_map = np.asarray(display_grid_map.occupancy, dtype=bool)
    belief = PartialFireCostmap(grid_map, static_map, config)
    world = WorldState.from_scenario(args.scenario, grid_map, config)
    world.set_known_occupancy_map(display_static_map)
    fire_localizer = FireLocalizer(
        world.map_metadata,
        world.static_obstacle_map,
        FireLocalizationConfig.from_mapping(args.fire_localization_config),
    )
    dynamic_mapping_config = DynamicObstacleMappingConfig.from_mapping(
        args.dynamic_obstacle_mapping_config
    )
    obstacle_by_id = {
        item.get("id"): item for item in obstacles if item.get("id")
    }
    missing_ignored_meshes = sorted(
        set(dynamic_mapping_config.ignored_fds_obstacle_ids) - obstacle_by_id.keys()
    )
    if missing_ignored_meshes:
        raise ValueError(
            "unknown ignored FDS obstacle IDs: "
            f"{missing_ignored_meshes}"
        )
    ignored_fds_bounds = tuple(
        obstacle_by_id[obstacle_id]["xb"]
        for obstacle_id in dynamic_mapping_config.ignored_fds_obstacle_ids
    )
    dynamic_obstacle_mapper = DynamicObstacleMapper(
        world.map_metadata, world.known_occupancy_map, dynamic_mapping_config,
        ignored_fds_bounds_world=ignored_fds_bounds,
    )
    if config.use_inflation and not math.isclose(
        dynamic_mapping_config.obstacle_inflation_radius_m,
        config.inflation_radius,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "dynamic obstacle inflation must match planner inflation radius"
        )
    exit_blockage_evaluator = ExitBlockageEvaluator(
        world.map_metadata,
        ExitBlockageConfig.from_mapping(args.exit_blockage_config),
    )
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
    exploration_config = ExplorationConfig.from_mapping(args.exploration_config)
    exploration_manager = ExplorationManager(evacuation_planner, exploration_config)
    hazard_knowledge_config = HazardKnowledgeConfig.from_mapping(
        args.hazard_knowledge_config
    )
    hazard_tracker = HazardKnowledgeTracker(
        temperature_elevated_c=(
            hazard_knowledge_config.temperature_elevated_c
        ),
        co_elevated_ppm=hazard_knowledge_config.co_elevated_ppm,
    )
    strategy_selector = EvacuationStrategySelector(
        world.map_metadata, hazard_tracker, return_planner, evacuation_planner,
        EvacuationRouteSelectionConfig.from_mapping(
            args.evacuation_route_selection_config
        ),
    )
    exit_switching_config = ExitSwitchingConfig.from_mapping(
        args.exit_switching_config
    )
    route_cost_monitor = RouteCostTrendMonitor(exit_switching_config)
    path_validation_config = PathValidationConfig.from_mapping(
        args.path_validation_config
    )
    event_replanning_config = EventReplanningConfig.from_mapping(
        args.replanning_config
    )
    legacy_replanning_keys = {
        key: value for key, value in args.replanning_config.items()
        if key in ReplanningConfig.__dataclass_fields__
    }
    replanning_config = ReplanningConfig.from_mapping(legacy_replanning_keys)
    event_replanning = EventReplanningPolicy(event_replanning_config)
    path_simplifier = SafePathSimplifier(
        world.map_metadata,
        PathSimplificationConfig.from_mapping(args.path_simplification_config),
    )
    following_config = VictimFollowingConfig.from_mapping(
        args.victim_following_config
    )
    victim_follower = VictimFollowingController(
        world.map_metadata, following_config
    )
    scripted_victim_motions = {}
    for human in args.scenario.get("humans", []):
        motion_config = ScriptedVictimMotionConfig.from_mapping(
            human.get("scripted_motion")
        )
        if not motion_config.enabled:
            continue
        controller = ScriptedVictimMotionController(
            world.map_metadata, human["id"], (human["x"], human["y"]),
            motion_config,
        )
        for index, waypoint in enumerate(motion_config.waypoints_world):
            _validate_free_point(
                f"human {human['id']} scripted waypoint {index}",
                waypoint, grid_map,
            )
        scripted_victim_motions[human["id"]] = controller
    world.attach_victim_following(victim_follower)
    world.attach_travel_history(travel_history)
    world.set_initial_robot_pose(args.start, math.radians(args.start_theta))
    world.configure_exploration_return(
        enabled=exploration_config.return_to_entrance_when_complete,
        entrance_exit_id=exploration_config.return_entrance_exit_id,
    )
    world.record_robot_position(args.start, sim_time=0.0)
    map_x = np.asarray([
        world.map_metadata.grid_to_world(col, 0)[0]
        for col in range(world.map_metadata.width)
    ])
    map_y = np.asarray([
        world.map_metadata.grid_to_world(0, row)[1]
        for row in range(world.map_metadata.height)
    ])
    for name, point in (("robot_start", args.start),):
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
    goal = tuple(args.start)
    follower = ReplannablePathFollower(grid_map, config)
    external_motion_config = ExternalWaypointMotionConfig.from_mapping(
        args.external_waypoint_motion_config
    )
    external_motion_follower = None
    if external_motion_config.enabled:
        external_waypoint_file = external_motion_config.resolve_waypoint_file(
            args.scenario_config.resolve().parent
        )
        external_world_path = load_external_waypoints(
            external_waypoint_file, external_motion_config
        )
        for index, point in enumerate(external_world_path):
            node = grid_map.world_to_grid(*point)
            if not grid_map.in_bounds(node):
                raise ValueError(
                    f"external waypoint {index} is outside factory_v3: {point}"
                )
        external_motion_follower = ExternalWaypointFollower(
            external_world_path,
            speed_mps=config.robot_speed,
            angular_speed_rad_s=config.robot_angular_speed,
            tolerance_m=config.waypoint_tolerance,
        )
        # Costmap/A* remains active for evaluation and display, but it does not
        # command this explicitly requested motion-only replay follower.
    thermal_camera = ThermalCameraMLX90640()
    # Stage 2 must be able to represent the specified 1600 ppm threshold.
    gas_sensor = MQ135Sensor(MQ135Config(
        max_ppm=2000.0, warning_ppm=1000.0,
        danger_ppm=1400.0, blocked_ppm=config.co_blocked,
    ))

    metrics = SimulationMetrics()
    human_detector = SimpleHumanDetector(
        args.human_detection_range, args.human_detection_fov_deg
    )
    active_victim = None
    victim_reached = False
    selected_exit = None
    returning_by_history = False
    returning_to_entrance = False
    blocked_return_grid = None
    detected_ids: set[str] = set()
    trajectory = metrics.actual_path_world
    trajectory.append((state.x, state.y))
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
    event_replanning.mark_reevaluation_complete(
        elapsed_time=sim_elapsed, robot_pose=(state.x, state.y),
        costmap_revision=belief.revision,
    )

    def begin_victim_following(victim) -> None:
        """Start from real poses only after a validated route is active."""
        victim_follower.start(
            victim.victim_id, victim.position_world,
            (state.x, state.y, state.theta), sim_time=sim_elapsed,
            costmap_revision=belief.revision,
        )
        world.start_victim_following(victim.victim_id)
        world.update_victim_status(
            victim.victim_id, VictimStatus.FOLLOWING, sim_time=sim_elapsed,
        )

    def remaining_route_is_safe() -> bool:
        remaining = follower.remaining_grid_path()
        if not remaining:
            return True
        validation = path_simplifier.validate_path(
            remaining, costmap=belief.final_cost_map,
            static_obstacle_map=belief.static_obstacle_map,
            dynamic_obstacle_map=world.dynamic_obstacle_mask(),
            estimated_fire_map=world.estimated_fire_map,
        )
        if validation.safe:
            world.record_route_validation(
                sim_time=sim_elapsed, costmap_revision=belief.revision
            )
        return validation.safe
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
        initial_cost = evaluate_path_cost(
            simplified.simplified_path_grid, belief.final_cost_map
        )
        route_cost_monitor.reset(
            None if initial_cost is None else initial_cost[1]
        )
        if selected_exit is not None:
            plan = replacement.evacuation_plan
            path_lengths = (
                {} if plan is None else {
                    item.exit_id: item.path_length_m
                    for item in plan.all_evaluations
                    if item.path_length_m is not None
                }
            )
            world.record_exit_selection(
                selected_exit, reason=replacement.reasons[0],
                costmap_revision=belief.revision,
                path_lengths_m=path_lengths,
            )
        return True

    def start_or_resume_exploration(phase: ExplorationPhase) -> bool:
        """Plan from the robot's actual pose; this function never teleports."""
        nonlocal goal, status, no_path_active, selected_exit
        follower.clear()
        world.clear_active_route()
        world.clear_active_exploration_plan()
        selected_exit = None
        plan = exploration_manager.plan_next_exit(
            world, (state.x, state.y), cost_map=belief.final_cost_map,
            costmap_revision=belief.revision, created_at=sim_elapsed,
            phase=phase,
        )
        if not plan.success:
            world.exploration_costmap_revision = belief.revision
            world.exploration_environment_revision = world.environment_revision
            if plan.failure_reason == "no_unchecked_exits":
                status = "EXPLORATION_COMPLETE"
                no_path_active = False
                if mission.current_state is MissionState.SEARCH_EXITS:
                    mission.handle_event(
                        MissionEvent.EXPLORATION_COMPLETED,
                        sim_time=sim_elapsed,
                    )
            else:
                status = f"EXPLORATION_STALLED: {plan.failure_reason}"
                no_path_active = True
                world.mark_exploration_stalled(plan.failure_reason)
                if mission.current_state is MissionState.SEARCH_EXITS:
                    mission.handle_event(
                        MissionEvent.EXPLORATION_STALLED,
                        sim_time=sim_elapsed, reason=plan.failure_reason,
                    )
            return False
        goal = plan.target_position_world
        simplified = activate_simplified_path(plan.path_grid, goal_world=goal)
        if simplified is None:
            status = "EXPLORATION_NO_PATH: simplification failed"
            no_path_active = True
            return False
        world.record_exploration_plan(plan, yaw_rad=state.theta)
        world.clear_exploration_stall()
        if mission.current_state is MissionState.EXPLORATION_STALLED:
            mission.handle_event(
                MissionEvent.RETRY_REQUESTED,
                sim_time=sim_elapsed,
                reason="costmap or obstacle revision changed",
            )
        if mission.current_state is MissionState.SEARCH_EXITS:
            mission.handle_event(
                MissionEvent.EXPLORATION_TARGET_SELECTED,
                exit_id=plan.target_exit_id,
                exit_position=plan.target_position_world,
                path=simplified.simplified_path_grid,
                sim_time=sim_elapsed,
            )
        status = f"EXPLORING {plan.target_exit_id}"
        no_path_active = False
        return True

    # Move physically from the configured pose before the first A* request.
    # This is a one-shot startup phase; resumed exploration always replans
    # directly from the robot's then-current pose without another advance.
    initial_advance = InitialAdvanceConfig.from_robot_motion(
        args.scenario.get("robot_motion")
    )
    initial_advance_pending = (
        initial_advance.distance_m > 0.0
        and external_motion_follower is None
    )
    initial_advance_goal = initial_advance.target_world(
        (state.x, state.y), state.theta
    )
    if initial_advance_pending:
        start_grid = grid_map.world_to_grid(state.x, state.y)
        advance_grid = grid_map.world_to_grid(*initial_advance_goal)
        advance_validation = path_simplifier.evaluate_segment(
            start_grid, advance_grid,
            costmap=belief.final_cost_map,
            static_obstacle_map=belief.static_obstacle_map,
            dynamic_obstacle_map=world.dynamic_obstacle_mask(),
            estimated_fire_map=world.estimated_fire_map,
        )
        if not advance_validation.safe:
            reasons = ",".join(
                item.value for item in advance_validation.rejection_reasons
            )
            raise ValueError(
                f"initial {initial_advance.distance_m:.3f} m forward motion "
                f"is unsafe: {reasons}"
            )
        follower.set_path(
            (start_grid, advance_grid), goal_world=initial_advance_goal,
            world_path=((state.x, state.y), initial_advance_goal),
        )
        goal = initial_advance_goal
        status = "INITIAL_FORWARD_ADVANCE"
    elif exploration_config.enabled:
        start_or_resume_exploration(ExplorationPhase.INITIAL)
    pygame_viewer = None
    thermal_viewer = None
    if not args.headless:
        pygame_viewer = PygameSimulationViewer(
            grid_map, config,
            display_static_obstacle_map=display_static_map,
            perception_display_config=PerceptionMapDisplayConfig.from_mapping(
                args.perception_map_display_config
            ),
            overlay_config=MapOverlayConfig.from_mapping(
                args.map_overlays_config
            ),
        )
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
        if (
            initial_advance_pending
            and follower.waypoint_index >= len(follower.world_path)
        ):
            initial_advance_pending = False
            follower.clear()
            goal = (state.x, state.y)
            status = "INITIAL_FORWARD_ADVANCE_COMPLETE"
            if exploration_config.enabled:
                start_or_resume_exploration(ExplorationPhase.INITIAL)
        newly_blocked = set()
        sensor_due = sim_elapsed - last_sensor_time >= config.sensor_update_interval_seconds - 1e-9
        if sensor_due:
            observed_before = belief.observed_mask.copy()
            latest_thermal, rays, _ = ground_truth.capture_thermal(
                thermal_camera, state, fds_time
            )
            if dynamic_mapping_config.enabled:
                dynamic_obstacle_mapper.process_thermal_rays(
                    f"obstacle:{sim_elapsed:.9f}", rays,
                    simulation_time=sim_elapsed, world_state=world,
                )
            dynamic_update = belief.update_dynamic_obstacles(
                world.dynamic_obstacle_mask(),
                inflation_radius_m=(
                    dynamic_mapping_config.obstacle_inflation_radius_m
                ),
            )
            thermal_update = belief.update_thermal_observations(
                latest_thermal, rays, fds_time
            )
            if fire_localizer.config.enabled:
                fire_localizer.add_thermal_observation(
                    f"thermal:{sim_elapsed:.9f}", sim_elapsed,
                    (state.x, state.y, state.theta), rays,
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
                if fire_localizer.config.enabled:
                    thermal_values = np.asarray(latest_thermal, dtype=float)
                    center_start = thermal_values.shape[1] // 3
                    center_end = thermal_values.shape[1] - center_start
                    finite_thermal = thermal_values[:, center_start:center_end]
                    finite_thermal = finite_thermal[np.isfinite(finite_thermal)]
                    local_temperature = (
                        float(finite_thermal.max()) if finite_thermal.size else 25.0
                    )
                    fire_localizer.add_co_observation(
                        f"co:{sim_elapsed:.9f}", sim_elapsed,
                        (state.x, state.y, state.theta),
                        gas.reading.measured_ppm, local_temperature,
                        thermal_direction_supported=any(
                            ray.valid
                            and center_start <= ray.col < center_end
                            and math.isfinite(float(ray.pixel_temperature))
                            and float(ray.pixel_temperature)
                            >= fire_localizer.config.thermal_warning_threshold_c
                            for ray in rays
                        ),
                    )
            else:
                from mapping.partial_costmap import BeliefUpdate
                co_update = BeliefUpdate(frozenset(), frozenset())
                latest_co_text = "invalid"
            if fire_localizer.config.enabled:
                localization_update = belief.update_estimated_fire_probability(
                    fire_localizer.fire_probability,
                    cost_weight=fire_localizer.config.estimated_fire_cost_weight,
                    minimum_probability=(
                        fire_localizer.config.possible_probability_threshold
                    ),
                )
            else:
                from mapping.partial_costmap import BeliefUpdate
                localization_update = BeliefUpdate(frozenset(), frozenset())
            _, newly_blocked = _combine_updates(
                thermal_update, co_update, localization_update, dynamic_update
            )
            world.estimated_fire_map.sync_from_belief(belief)
            world.estimated_fire_map.sync_fire_localization(fire_localizer)
            world.fire_localization_result = fire_localizer.latest_result
            world.update_costmap_revision(belief.revision)
            if (
                exit_blockage_evaluator.config.enabled
                and dynamic_update.changed_cells
            ):
                for exit_item in world.exits.values():
                    blockage = exit_blockage_evaluator.evaluate(
                        exit_item, (state.x, state.y),
                        cost_map=belief.final_cost_map,
                        static_obstacle_map=world.known_occupancy_map,
                        dynamic_inflated_map=(
                            belief.dynamic_inflated_obstacle_map
                        ),
                        active_obstacles=world.get_active_dynamic_obstacles(),
                        evaluated_at=sim_elapsed,
                        environment_revision=world.environment_revision,
                    )
                    world.record_exit_blockage_result(blockage)
                    if blockage.blocked_confirmed and exit_item.status is not ExitStatus.BLOCKED:
                        world.update_exit_status(
                            exit_item.exit_id, ExitStatus.BLOCKED,
                            sim_time=sim_elapsed, reason=blockage.reason,
                        )
                        if exit_item.exit_id in {
                            selected_exit, world.current_target_exit_id,
                            world.current_exploration_target_exit_id,
                        }:
                            world.last_perception_replan_reason = (
                                f"current exit {exit_item.exit_id} blocked by perceived obstacle"
                            )
            # Capture hazard knowledge at observation time so later normal
            # readings do not erase evidence seen earlier in the mission.
            world.hazard_knowledge_decision = hazard_tracker.evaluate(
                world.estimated_fire_map, evaluated_at=sim_elapsed
            )
            if (
                world.exploration_stalled
                and (
                    world.exploration_costmap_revision != belief.revision
                    or world.exploration_environment_revision
                    != world.environment_revision
                )
            ):
                start_or_resume_exploration(
                    ExplorationPhase.REPLAN_AFTER_MAP_CHANGE
                )
            # Gradual cost growth uses revision-based hysteresis. Hard blocks
            # are still handled immediately by the safety validation below.
            active_decision = world.active_route_decision
            if (
                active_decision is not None
                and not world.current_target_is_usable
                and world.active_route_valid
                and world.active_path_simplification is not None
            ):
                remaining_for_cost = (
                    world.active_path_simplification.simplified_path_grid[
                        max(0, follower.waypoint_index - 1):
                    ]
                )
                trend = route_cost_monitor.record(
                    remaining_for_cost, belief.final_cost_map,
                    revision=belief.revision, evaluated_at=sim_elapsed,
                )
                if route_cost_monitor.samples:
                    world.record_route_cost(
                        route_cost_monitor.samples[-1],
                        baseline=trend.baseline_average_cost,
                        consecutive=trend.consecutive_increases,
                    )
                if (
                    trend.switch_required
                    and not world.exit_switch_is_cooling_down(sim_elapsed)
                ):
                    reason = trend.reason
                    previous_exit = active_decision.target_exit_id
                    next_waypoint = (
                        follower.world_path[follower.waypoint_index]
                        if follower.waypoint_index < len(follower.world_path)
                        else None
                    )
                    direction = current_direction_world(
                        (state.x, state.y), next_waypoint,
                        travel_history.get_points_world()[-2:], state.theta,
                    )
                    follower.clear()
                    world.invalidate_active_route(reason)
                    world.clear_active_evacuation_plan()
                    mission.handle_event(
                        MissionEvent.ACTIVE_PATH_INVALIDATED,
                        sim_time=sim_elapsed, reason=reason,
                    )
                    no_path_active = True
                    status = f"PATH_INVALIDATED: {reason}"
                    if (
                        replanning_config.enabled
                        and world.route_replan_count
                        < replanning_config.max_replan_attempts
                    ):
                        mission.handle_event(
                            MissionEvent.REPLAN_REQUESTED,
                            sim_time=sim_elapsed, reason=reason,
                        )
                        world.route_replan_count += 1
                        replacement = strategy_selector.replan_to_opposite_exit(
                            world_state=world,
                            current_position_world=(state.x, state.y),
                            direction_world=direction,
                            cost_map=belief.final_cost_map,
                            costmap_revision=belief.revision,
                            created_at=sim_elapsed,
                            current_exit_id=previous_exit,
                            minimum_direction_difference_deg=(
                                exit_switching_config
                                .minimum_direction_difference_deg
                            ),
                        )
                        if not replacement.success:
                            replacement = strategy_selector.replan_to_safe_exit(
                                world_state=world,
                                current_position_world=(state.x, state.y),
                                cost_map=belief.final_cost_map,
                                costmap_revision=belief.revision,
                                created_at=sim_elapsed,
                                excluded_exit_ids=(previous_exit,),
                            )
                        replacement_cost = (
                            evaluate_path_cost(
                                replacement.path_grid, belief.final_cost_map
                            ) if replacement.success else None
                        )
                        improves = (
                            replacement_cost is not None
                            and trend.current_average_cost is not None
                            and replacement_cost[1]
                            < trend.current_average_cost - 1e-12
                        )
                        if improves and activate_replacement_route(replacement):
                            if previous_exit is not None:
                                world.update_exit_status(
                                    previous_exit,
                                    ExitStatus.DANGER_EXPECTED,
                                    sim_time=sim_elapsed,
                                    reason=reason,
                                )
                            world.record_exit_switch(
                                previous_exit_id=previous_exit,
                                new_exit_id=replacement.target_exit_id,
                                reason=reason, sim_time=sim_elapsed,
                                cooldown_seconds=(
                                    exit_switching_config.switch_cooldown_sec
                                ),
                                validation_result="validated",
                            )
                        else:
                            mission.handle_event(
                                MissionEvent.NO_SAFE_ROUTE_FOUND,
                                sim_time=sim_elapsed,
                                reason="no_better_opposite_exit",
                            )
                            world.final_route_failure_reason = "no_better_opposite_exit"
                            status = "NO_SAFE_ROUTE: no_better_opposite_exit"
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
                target_blocked = (
                    world.active_route_decision.target_exit_id is not None
                    and world.get_exit(
                        world.active_route_decision.target_exit_id
                    ).status in (
                        ExitStatus.BLOCKED, ExitStatus.DANGEROUS,
                        ExitStatus.DANGER_EXPECTED,
                    )
                )
                valid = route_validation.safe and not target_blocked
                blocked_grid = route_validation.first_rejected_cell
                invalid_reason = (
                    None if valid else (
                        "current_exit_blocked_or_dangerous" if target_blocked else
                        route_validation.rejection_reasons[0].value
                    )
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
                                risk_first=(
                                    invalid_reason
                                    == "current_exit_blocked_or_dangerous"
                                ),
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
                    exploration_target_blocked = (
                        world.active_exploration_plan is not None
                        and world.get_exit(
                            world.active_exploration_plan.target_exit_id
                        ).status in (
                            ExitStatus.BLOCKED, ExitStatus.DANGEROUS,
                            ExitStatus.DANGER_EXPECTED,
                        )
                    )
                    if not search_validation.safe or exploration_target_blocked:
                        was_exploring = world.active_exploration_plan is not None
                        follower.clear()
                        blocked_return_grid = search_validation.first_rejected_cell
                        status = (
                            "SEARCH_PATH_INVALIDATED: "
                            + (
                                "current_exit_blocked"
                                if exploration_target_blocked else
                                search_validation.rejection_reasons[0].value
                            )
                        )
                        no_path_active = True
                        world.clear_path_simplification()
                        if was_exploring:
                            start_or_resume_exploration(
                                ExplorationPhase.REPLAN_AFTER_MAP_CHANGE
                            )
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

        for victim_id, controller in scripted_victim_motions.items():
            victim = world.get_victim(victim_id)
            if (
                active_victim is not None
                and active_victim["id"] == victim_id
                and not victim_reached
                and victim.status in (
                    VictimStatus.DETECTED, VictimStatus.APPROACHING,
                )
            ):
                controller.move_toward(
                    (state.x, state.y), dt=config.simulation_dt,
                    stop_distance_m=args.victim_approach_distance,
                    static_obstacle_map=world.static_obstacle_map,
                    dynamic_obstacle_map=world.dynamic_obstacle_mask(),
                )
                world.update_victim_position(
                    victim_id, controller.position_world
                )
                continue
            if (
                controller.completed
                or (
                    controller.config.stop_when_detected
                    and victim.status is not VictimStatus.UNDETECTED
                )
            ):
                continue
            controller.update(
                dt=config.simulation_dt,
                static_obstacle_map=world.static_obstacle_map,
                dynamic_obstacle_map=world.dynamic_obstacle_mask(),
            )
            world.update_victim_position(victim_id, controller.position_world)
        args.humans = world.legacy_humans()

        if active_victim is not None and not victim_reached:
            current_victim = world.get_victim(active_victim["id"])
            active_victim["x"], active_victim["y"] = (
                current_victim.position_world
            )
            active_victim["distance"] = math.dist(
                (state.x, state.y), current_victim.position_world
            )

        detections = human_detector.detect(
            robot_position=(state.x, state.y),
            humans=args.humans,
            robot_heading_rad=state.theta,
            obstacle_map=static_map,
            map_origin=(grid_map.x_min, grid_map.y_min),
            map_resolution=grid_map.resolution,
            use_line_of_sight=True,
        )
        detected_ids.update(item["id"] for item in detections)
        if active_victim is None and detections:
            active_victim = detections[0]
            if world.active_exploration_plan is not None:
                world.record_exploration_interruption(
                    sim_time=sim_elapsed,
                    reason="victim_detected_during_exit_exploration",
                    victim_id=active_victim["id"],
                    robot_pose_world=(state.x, state.y, state.theta),
                    target_exit_id=(
                        world.active_exploration_plan.target_exit_id
                    ),
                    active_path_grid=follower.remaining_grid_path(),
                    costmap_revision=belief.revision,
                )
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
            world.clear_active_exploration_plan()
            world.clear_path_simplification()
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
                if route_decision.strategy in (
                    EvacuationStrategy.SAFE_EXIT_PLANNING,
                    EvacuationStrategy.NEAREST_REACHABLE_EXIT,
                ):
                    evacuation_plan = route_decision.evacuation_plan
                    selected_exit = route_decision.target_exit_id
                    metrics.selected_exit = selected_exit
                    goal = route_decision.target_position_world
                    mission.handle_event(
                        MissionEvent.SAFE_EXIT_SELECTED,
                        exit_id=selected_exit, exit_position=goal,
                        selection_reason=route_decision.reasons[0],
                        sim_time=sim_elapsed,
                    )
                    simplified = activate_simplified_path(
                        route_decision.path_grid, goal_world=goal
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
                        begin_victim_following(victim)
                        mission.handle_event(
                            MissionEvent.EVACUATION_PLAN_CREATED,
                            exit_id=selected_exit,
                            exit_position=goal,
                            path=simplified.simplified_path_grid,
                            selection_reason=route_decision.reasons[0],
                            sim_time=sim_elapsed,
                        )
                        initial_cost = evaluate_path_cost(
                            simplified.simplified_path_grid,
                            belief.final_cost_map,
                        )
                        route_cost_monitor.reset(
                            None if initial_cost is None else initial_cost[1]
                        )
                        path_lengths = {
                            item.exit_id: item.path_length_m
                            for item in evacuation_plan.all_evaluations
                            if item.path_length_m is not None
                        }
                        world.record_exit_selection(
                            selected_exit, reason=route_decision.reasons[0],
                            costmap_revision=belief.revision,
                            path_lengths_m=path_lengths,
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
                        begin_victim_following(victim)
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
            MissionState.EXPLORATION_STALLED,
            MissionState.EXPLORATION_COMPLETE,
        ) or returning_by_history:
            # TODO: A later sensor/costmap update may explicitly issue
            # RETRY_REQUESTED. Until then, do not plan back to the victim goal.
            replan_reason = None
        elif not remaining_path:
            replan_reason = "initial_or_missing_path"
        else:
            active_victim_item = (
                None if world.active_following_victim_id is None else
                world.get_victim(world.active_following_victim_id)
            )
            replan_decision = event_replanning.evaluate(
                current_path=remaining_path,
                current_costmap=belief.final_cost_map,
                costmap_revision=belief.revision,
                dynamic_obstacle_map=world.dynamic_obstacle_mask(),
                temperature_map=belief.temperature_belief_map,
                co_map=belief.co_belief_map,
                temperature_observed_mask=belief.temperature_observed_mask,
                co_observed_mask=belief.co_observed_mask,
                exit_statuses={
                    exit_id: item.status for exit_id, item in world.exits.items()
                },
                current_exit_id=selected_exit,
                robot_pose=(state.x, state.y), elapsed_time=sim_elapsed,
                victim_follow_active=victim_follower.active,
                victim_follow_distance_m=(
                    None if active_victim_item is None else
                    active_victim_item.follow_distance_m
                ),
                victim_progress_stalled=(
                    victim_follower.state is FollowState.FOLLOW_FAILED
                ),
            )
            if replan_decision.required:
                world.record_replan_decision(
                    replan_decision, costmap_revision=belief.revision,
                    sim_time=sim_elapsed, robot_pose_world=(state.x, state.y),
                    selected_exit_id=selected_exit,
                )
                if replan_decision.reason in (
                    ReplanReason.PERIODIC_REEVALUATION,
                    ReplanReason.DISTANCE_REEVALUATION,
                ):
                    # A general reevaluation first validates the current path.
                    # A* and controller replacement are unnecessary when it is
                    # still safe and the target exit remains valid.
                    if remaining_route_is_safe() and (
                        selected_exit is None
                        or world.get_exit(selected_exit).status not in (
                            ExitStatus.BLOCKED, ExitStatus.DANGEROUS,
                            ExitStatus.DANGER_EXPECTED,
                        )
                    ):
                        event_replanning.mark_reevaluation_complete(
                            elapsed_time=sim_elapsed,
                            robot_pose=(state.x, state.y),
                            costmap_revision=belief.revision,
                        )
                    else:
                        replan_reason = replan_decision.reason.value
                elif replan_decision.reason is ReplanReason.VICTIM_FOLLOW_FAILURE:
                    # FOLLOW_WAIT owns ordinary lag recovery. Only a confirmed
                    # following failure proceeds to route replanning.
                    if victim_follower.state is FollowState.FOLLOW_FAILED:
                        replan_reason = replan_decision.reason.value
                elif world.active_route_decision is None:
                    replan_reason = replan_decision.reason.value

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
            if replan_reason != "initial_or_missing_path":
                event_replanning.mark_processed(
                    replan_decision, costmap_revision=belief.revision,
                    elapsed_time=sim_elapsed, robot_pose=(state.x, state.y),
                    selected_exit_id=selected_exit,
                )
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

        follow_paused = victim_follower.state in (
            FollowState.FOLLOW_WAIT, FollowState.FOLLOW_FAILED,
        )
        if follow_paused:
            moved, motion_status = 0.0, "waiting for victim"
        elif external_motion_follower is not None:
            moved, motion_status = external_motion_follower.update(
                state, config.simulation_dt
            )
            motion_status = "external waypoint: " + motion_status
        else:
            moved, motion_status = follower.update(
                state, config.simulation_dt, belief.final_cost_map
            )
        metrics.travelled_distance += moved
        if moved > 0.0:
            trajectory.append((state.x, state.y))
            world.record_robot_position(
                (state.x, state.y), sim_time=sim_elapsed,
                is_returning=(returning_by_history or returning_to_entrance),
            )
            if victim_follower.active:
                victim_follower.record_robot_pose(
                    (state.x, state.y, state.theta), sim_time=sim_elapsed,
                    costmap_revision=belief.revision,
                )

        if (
            victim_follower.active
            and mission.current_state in (
                MissionState.ESCORT_VICTIM, MissionState.FOLLOW_WAIT,
            )
        ):
            if follower.goal_reached(state, goal):
                victim_follower.request_exit_catch_up()
            follow_update = victim_follower.update(
                (state.x, state.y), dt=config.simulation_dt,
                sim_time=sim_elapsed,
                static_obstacle_map=world.static_obstacle_map,
                dynamic_obstacle_map=world.dynamic_obstacle_mask(),
            )
            world.update_victim_following(follow_update, sim_time=sim_elapsed)
            args.humans = world.legacy_humans()
            victim = world.get_victim(victim_follower.victim_id)
            if follow_update.event == "victim_lagging":
                world.update_victim_status(
                    victim.victim_id, VictimStatus.FOLLOW_WAIT,
                    sim_time=sim_elapsed,
                )
                mission.handle_event(
                    MissionEvent.VICTIM_LAGGING, victim_id=victim.victim_id,
                    sim_time=sim_elapsed,
                )
                status = "FOLLOW_WAIT: victim is catching up"
                print("요구조자와 거리가 멀어졌습니다. 천천히 따라오십시오.")
            elif follow_update.event == "victim_caught_up":
                if remaining_route_is_safe():
                    world.update_victim_status(
                        victim.victim_id, VictimStatus.FOLLOWING,
                        sim_time=sim_elapsed,
                    )
                    mission.handle_event(
                        MissionEvent.VICTIM_CAUGHT_UP,
                        victim_id=victim.victim_id, sim_time=sim_elapsed,
                    )
                    status = "ESCORT_RESUMED: route revalidated"
                else:
                    follower.clear()
                    world.invalidate_active_route(
                        "route became unsafe while waiting for victim"
                    )
                    mission.handle_event(
                        MissionEvent.ACTIVE_PATH_INVALIDATED,
                        sim_time=sim_elapsed,
                        reason="route became unsafe while waiting for victim",
                    )
                    status = "REPLAN: route changed during follow wait"
            elif follow_update.event == "follow_reprompt":
                status = "FOLLOW_WAIT: guidance repeated"
                print("요구조자에게 다시 안내합니다. 로봇을 따라오십시오.")
            elif follow_update.event == "follow_failed":
                world.update_victim_status(
                    victim.victim_id, VictimStatus.FOLLOW_FAILED,
                    sim_time=sim_elapsed,
                )
                mission.handle_event(
                    MissionEvent.VICTIM_FOLLOW_FAILED,
                    victim_id=victim.victim_id, sim_time=sim_elapsed,
                    reason=victim_follower.failure_reason,
                )
                status = "FOLLOW_FAILED: robot stopped safely"

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
            active_victim is None
            and world.active_exploration_plan is not None
            and math.hypot(state.x - goal[0], state.y - goal[1])
            <= args.exit_reached_distance
        ):
            checked_exit_id = world.active_exploration_plan.target_exit_id
            continuing_phase = world.active_exploration_plan.phase
            dynamic = world.dynamic_obstacle_mask()
            effective = belief.final_cost_map.copy()
            effective[world.static_obstacle_map | dynamic] = np.inf
            evaluation = exit_evaluator.evaluate(
                world.get_exit(checked_exit_id), (state.x, state.y),
                cost_map=effective,
                static_obstacle_map=world.static_obstacle_map,
                dynamic_obstacle_map=dynamic,
                estimated_fire_map=world.estimated_fire_map,
                evaluated_at=sim_elapsed,
            )
            reasons = set(evaluation.rejection_reasons)
            if evaluation.accepted:
                checked_status = ExitStatus.USABLE
                checked_reason = "safe reachable exit confirmed within check radius"
            elif reasons & {
                ExitRejectionReason.TEMPERATURE_LIMIT_EXCEEDED,
                ExitRejectionReason.CO_LIMIT_EXCEEDED,
                ExitRejectionReason.PATH_RISK_COST_EXCEEDED,
            }:
                checked_status = ExitStatus.DANGEROUS
                checked_reason = ",".join(sorted(item.value for item in reasons))
            elif ExitRejectionReason.EXIT_BLOCKED in reasons:
                checked_status = ExitStatus.BLOCKED
                checked_reason = "exit_blocked"
            else:
                checked_status = ExitStatus.UNKNOWN
                checked_reason = (
                    ",".join(sorted(item.value for item in reasons))
                    or "visit completed; safety remains unknown"
                )
            world.record_exit_check(
                checked_exit_id, checked_status, sim_time=sim_elapsed,
                costmap_revision=belief.revision, reason=checked_reason,
            )
            mission.handle_event(
                MissionEvent.EXIT_CHECK_COMPLETED,
                exit_id=checked_exit_id, sim_time=sim_elapsed,
                reason=checked_reason,
            )
            world.clear_active_exploration_plan()
            world.clear_path_simplification()
            follower.clear()
            status = f"EXIT_CHECKED: {checked_exit_id}"
            if exploration_config.reselect_after_each_exit_check:
                start_or_resume_exploration(continuing_phase)

        robot_at_exit = follower.goal_reached(state, goal)
        victim_at_exit = (
            victim_follower.position_world is not None
            and math.dist(victim_follower.position_world, goal)
            <= args.exit_reached_distance
        )
        if robot_at_exit and selected_exit is not None:
            arrival_exit = world.get_exit(selected_exit)
            if arrival_exit.metadata.get("escort_arrival_revision") != belief.revision:
                dynamic = world.dynamic_obstacle_mask()
                effective = belief.final_cost_map.copy()
                effective[world.static_obstacle_map | dynamic] = np.inf
                arrival_evaluation = exit_evaluator.evaluate(
                    arrival_exit, (state.x, state.y), cost_map=effective,
                    static_obstacle_map=world.static_obstacle_map,
                    dynamic_obstacle_map=dynamic,
                    estimated_fire_map=world.estimated_fire_map,
                    evaluated_at=sim_elapsed,
                )
                arrival_exit.metadata["escort_arrival_revision"] = belief.revision
                arrival_exit.metadata["escort_arrival_evaluation"] = (
                    arrival_evaluation.to_dict()
                )
                arrival_reasons = set(arrival_evaluation.rejection_reasons)
                if (
                    arrival_evaluation.accepted
                    and arrival_evaluation.unknown_ratio is not None
                    and arrival_evaluation.unknown_ratio
                    <= exit_evaluator.config.usable_confirmation_max_unknown_ratio
                ):
                    world.update_exit_status(
                        selected_exit, ExitStatus.USABLE,
                        sim_time=sim_elapsed,
                        temperature_c=arrival_evaluation.exit_temperature_c
                        if arrival_evaluation.exit_temperature_c is not None
                        else np.nan,
                        co_ppm=arrival_evaluation.exit_co_ppm
                        if arrival_evaluation.exit_co_ppm is not None else np.nan,
                        path_cost=arrival_evaluation.accumulated_risk_cost,
                    )
                elif arrival_reasons & {
                    ExitRejectionReason.TEMPERATURE_LIMIT_EXCEEDED,
                    ExitRejectionReason.CO_LIMIT_EXCEEDED,
                    ExitRejectionReason.PATH_RISK_COST_EXCEEDED,
                }:
                    world.update_exit_status(
                        selected_exit, ExitStatus.DANGEROUS,
                        sim_time=sim_elapsed,
                        reason=",".join(
                            sorted(item.value for item in arrival_reasons)
                        ),
                    )
                elif ExitRejectionReason.EXIT_BLOCKED in arrival_reasons:
                    world.update_exit_status(
                        selected_exit, ExitStatus.BLOCKED,
                        sim_time=sim_elapsed, reason="exit_blocked_at_arrival",
                    )
        selected_exit_usable = (
            selected_exit is not None
            and world.get_exit(selected_exit).status is ExitStatus.USABLE
        )
        route_revision_valid = (
            world.active_route_valid
            and world.active_route_costmap_revision == belief.revision
        )
        if robot_at_exit and victim_follower.active and not victim_at_exit:
            status = "ROBOT_AT_EXIT: waiting for victim"

        evacuation_ready = (
            victim_reached and selected_exit is not None
            and victim_follower.position_world is not None
            and evacuation_success_ready(
                robot_position_world=(state.x, state.y),
                victim_position_world=victim_follower.position_world,
                exit_position_world=goal,
                exit_radius_m=args.exit_reached_distance,
                exit_usable=selected_exit_usable,
                route_valid=route_revision_valid,
                victim_moved_by_following=(
                    victim_follower.active
                    and victim_follower.total_victim_distance_m > 0.0
                ),
            )
        )
        if evacuation_ready:
            world.update_victim_status(
                active_victim["id"], VictimStatus.EVACUATED,
                sim_time=sim_elapsed,
            )
            victim_follower.mark_evacuated()
            world.complete_victim_following(
                active_victim["id"], sim_time=sim_elapsed,
                reason=(
                    "robot and victim reached a USABLE exit using a route "
                    f"validated at costmap revision {belief.revision}"
                ),
            )
            remaining_victims = bool(world.get_unrescued_victims())
            mission.handle_event(
                MissionEvent.EVACUATION_CONFIRMED,
                exit_id=selected_exit, sim_time=sim_elapsed,
                remaining_victims=remaining_victims,
                reason="robot and moving victim both reached a usable exit",
            )
            world.clear_active_evacuation_plan()
            world.clear_active_route()
            victim_follower.clear()
            world.clear_victim_following()
            status = "EVACUATION_SUCCESS"
            if (
                exploration_config.enabled
                and exploration_config.resume_after_victim_evacuation
                and exploration_config.start_after_evacuation
            ):
                if mission.current_state is MissionState.EVACUATION_COMPLETE:
                    mission.handle_event(
                        MissionEvent.SEARCH_RESUMED,
                        sim_time=sim_elapsed,
                        reason="resume exploration from evacuation completion pose",
                    )
                active_victim = None
                victim_reached = False
                selected_exit = None
                returning_by_history = False
                returning_to_entrance = False
                blocked_return_grid = None
                resumed = start_or_resume_exploration(
                    ExplorationPhase.RESUMED_AFTER_EVACUATION
                )
                if resumed:
                    continue
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
                    victim_follower,
                    world.fire_localization_result,
                    exit_states={
                        key: item.status for key, item in world.exits.items()
                    },
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
                victim_follower,
                world.fire_localization_result,
                exit_states={
                    key: item.status for key, item in world.exits.items()
                },
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


def export_actual_path(metrics, yaml_path, image_path, *, elapsed):
    """Export only poses reached through physical translation, in world (x,y)."""
    points = tuple(metrics.actual_path_world)
    if not points:
        raise ValueError("actual robot path is empty")
    yaml_path = Path(yaml_path)
    image_path = Path(image_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coordinate_frame": "factory_v3_world_xy_m",
        "point_order": "x_y",
        "simulation_time_s": round(float(elapsed), 6),
        "travelled_distance_m": round(float(metrics.travelled_distance), 6),
        "points": [
            {"x": round(float(x), 6), "y": round(float(y), 6)}
            for x, y in points
        ],
    }
    yaml_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    import pygame

    width, height, margin = 1000, 760, 55
    surface = pygame.Surface((width, height))
    surface.fill((250, 250, 250))
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min(
        (width - 2 * margin) / span_x,
        (height - 2 * margin) / span_y,
    )

    def screen_point(point):
        return (
            int(margin + (point[0] - min_x) * scale),
            int(height - margin - (point[1] - min_y) * scale),
        )

    pygame.draw.rect(
        surface, (190, 190, 190),
        (margin, margin, width - 2 * margin, height - 2 * margin), 2,
    )
    rendered = [screen_point(item) for item in points]
    if len(rendered) >= 2:
        pygame.draw.lines(surface, (35, 105, 210), False, rendered, 4)
    pygame.draw.circle(surface, (35, 175, 75), rendered[0], 9)
    pygame.draw.circle(surface, (220, 55, 55), rendered[-1], 9)
    pygame.image.save(surface, str(image_path))


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
    parser.add_argument(
        "--co-npz", type=Path,
        default=base / "processed/fds_co_2d_timeseries.npz",
    )
    parser.add_argument("--start", type=float, nargs=2, default=None)
    parser.add_argument("--start-theta", type=float, default=None)
    parser.add_argument("--grid-resolution", type=float, default=None)
    parser.add_argument("--fds-start-time", type=float, default=0.0)
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--sensor-interval", type=float, default=0.25)
    parser.add_argument("--replan-interval", type=float, default=1.0)
    parser.add_argument("--robot-speed", type=float, default=1.3)
    parser.add_argument("--robot-angular-speed-deg", type=float, default=None)
    parser.add_argument("--render-fps", type=int, default=None)
    parser.add_argument("--unknown-penalty", type=float, default=2.0)
    parser.add_argument("--temperature-weight", type=float, default=None)
    parser.add_argument("--temperature-power", type=float, default=None)
    parser.add_argument("--co-weight", type=float, default=None)
    parser.add_argument("--co-power", type=float, default=None)
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
    parser.add_argument(
        "--actual-path-yaml", type=Path, default=None,
        help="write the physically traversed world-coordinate path as YAML",
    )
    parser.add_argument(
        "--actual-path-image", type=Path, default=None,
        help="write a simple PNG visualization of the physically traversed path",
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
    args.co_npz = (
        args.co_npz
        if args.co_npz != Path(__file__).resolve().parent / "processed/fds_co_2d_timeseries.npz"
        else base / scenario["co_npz"]
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
    motion_config = scenario.get("robot_motion", {})
    if args.robot_angular_speed_deg is None:
        args.robot_angular_speed_deg = float(
            motion_config.get("angular_speed_deg_s", 120.0)
        )
    if args.robot_speed == 1.3:
        args.robot_speed = float(motion_config.get("linear_speed_mps", 1.3))
    display_config = scenario.get("display", {})
    if args.render_fps is None:
        args.render_fps = int(display_config.get("render_fps", 30))
    sensor_costmap = scenario.get("sensor_costmap", {})
    if not isinstance(sensor_costmap, dict):
        raise ValueError("sensor_costmap must be a mapping")
    cost_defaults = {
        "temperature_weight": 8.0,
        "temperature_power": 2.0,
        "co_weight": 8.0,
        "co_power": 2.0,
    }
    for name, default in cost_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, float(sensor_costmap.get(name, default)))
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if args.robot_angular_speed_deg <= 0.0:
        raise ValueError("robot angular speed must be positive")
    if args.render_fps < 1:
        raise ValueError("render FPS must be a positive integer")
    args.humans = list(scenario["humans"])
    args.exits = list(scenario["exits"])
    args.human_detection_range = float(scenario["human_detection_range_m"])
    args.human_detection_fov_deg = float(
        scenario["human_detection_horizontal_fov_deg"]
    )
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
    args.hazard_knowledge_config = scenario.get("hazard_knowledge", {})
    args.exit_switching_config = scenario.get("exit_switching", {})
    args.exploration_config = scenario.get("exploration", {})
    args.path_validation_config = scenario.get("path_validation", {})
    args.replanning_config = scenario.get("replanning", {})
    args.path_simplification_config = scenario.get("path_simplification", {})
    args.victim_following_config = scenario.get("victim_following", {})
    args.fire_localization_config = scenario.get("fire_localization", {})
    args.dynamic_obstacle_mapping_config = scenario.get(
        "dynamic_obstacle_mapping", {}
    )
    args.exit_blockage_config = scenario.get("exit_blockage", {})
    args.perception_map_display_config = scenario.get(
        "perception_map_display", {}
    )
    args.map_overlays_config = scenario.get("map_overlays", {})
    args.external_waypoint_motion_config = scenario.get(
        "external_waypoint_motion", {}
    )
    return args


def main() -> int:
    args = apply_scenario_config(parse_args())
    success, metrics, belief, elapsed = run_simulation(args)
    print_summary(success, metrics, belief, elapsed)
    if (args.actual_path_yaml is None) != (args.actual_path_image is None):
        raise ValueError(
            "--actual-path-yaml and --actual-path-image must be provided together"
        )
    if args.actual_path_yaml is not None:
        export_actual_path(
            metrics, args.actual_path_yaml, args.actual_path_image,
            elapsed=elapsed,
        )
    if args.debug_costmap_plots:
        show_debug_costmaps(belief)
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
