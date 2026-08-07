"""Pygame main viewer and Matplotlib thermal-only viewer.

These classes consume simulation state and arrays for display. They never
produce sensor observations, costs, paths, or other simulation inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from world.entities import ExitStatus


def _rgb(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three RGB values")
    result = tuple(int(item) for item in value)
    if any(item < 0 or item > 255 for item in result):
        raise ValueError(f"{name} RGB values must be in [0,255]")
    return result


@dataclass(frozen=True)
class PerceptionMapDisplayConfig:
    unknown_color: tuple[int, int, int] = (110, 110, 110)
    safe_color: tuple[int, int, int] = (40, 170, 70)
    caution_color: tuple[int, int, int] = (230, 190, 40)
    danger_color: tuple[int, int, int] = (210, 50, 50)
    blocked_color: tuple[int, int, int] = (25, 25, 25)
    dynamic_obstacle_color: tuple[int, int, int] = (120, 40, 150)
    safe_cost_max: float = 0.25
    caution_cost_max: float = 0.60
    danger_cost_max: float = 0.99

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown perception_map_display settings: {sorted(unknown)}")
        for name in (
            "unknown_color", "safe_color", "caution_color", "danger_color",
            "blocked_color", "dynamic_obstacle_color",
        ):
            if name in values:
                values[name] = _rgb(values[name], name)
        return cls(**values)

    def __post_init__(self):
        for name in (
            "unknown_color", "safe_color", "caution_color", "danger_color",
            "blocked_color", "dynamic_obstacle_color",
        ):
            object.__setattr__(self, name, _rgb(getattr(self, name), name))
        if not 0 <= self.safe_cost_max < self.caution_cost_max < self.danger_cost_max <= 1:
            raise ValueError("display cost thresholds must increase within [0,1]")

    def color_for(self, *, observed, blocked, normalized_cost):
        if blocked:
            return self.blocked_color
        if not observed:
            return self.unknown_color
        if normalized_cost <= self.safe_cost_max:
            return self.safe_color
        if normalized_cost <= self.caution_cost_max:
            return self.caution_color
        return self.danger_color


@dataclass(frozen=True)
class MapOverlayConfig:
    show_dynamic_obstacles: bool = True
    show_fire_candidates: bool = True
    show_estimated_fire_center: bool = True
    show_detected_humans: bool = True
    show_exit_states: bool = True
    show_current_path: bool = True

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown map_overlays settings: {sorted(unknown)}")
        return cls(**values)

    def __post_init__(self):
        if any(type(getattr(self, name)) is not bool for name in self.__dataclass_fields__):
            raise TypeError("map overlay settings must be boolean")


class WorldTransform:
    """Convert FDS world metres to Pygame coordinates with explicit y flip."""

    def __init__(self, grid_map, rect) -> None:
        self.grid_map = grid_map
        self.rect = rect
        map_width = grid_map.x_max - grid_map.x_min
        map_height = grid_map.y_max - grid_map.y_min
        self.scale = min(rect.width / map_width, rect.height / map_height)
        draw_width = map_width * self.scale
        draw_height = map_height * self.scale
        self.offset_x = rect.x + (rect.width - draw_width) / 2.0
        self.offset_y = rect.y + (rect.height - draw_height) / 2.0

    def world_to_screen(self, x, y):
        px = self.offset_x + (x - self.grid_map.x_min) * self.scale
        py = self.offset_y + (self.grid_map.y_max - y) * self.scale
        return int(round(px)), int(round(py))

    def grid_rect(self, gx, gy):
        import pygame

        x, y = self.grid_map.grid_to_world(gx, gy)
        half = self.grid_map.resolution / 2.0
        left, bottom = self.world_to_screen(x - half, y - half)
        right, top = self.world_to_screen(x + half, y + half)
        return pygame.Rect(
            min(left, right), min(top, bottom),
            max(1, abs(right - left)), max(1, abs(bottom - top)),
        )


class PygameSimulationViewer:
    """Main real-time map display; all simulation logic remains outside."""

    BACKGROUND = (28, 31, 36)
    UNKNOWN = (82, 86, 94)
    OBSERVED = (65, 125, 82)
    STATIC = (20, 20, 22)
    BLOCKED = (220, 45, 50)
    PATH = (35, 225, 235)
    TRAJECTORY = (75, 135, 255)
    TRAVEL_HISTORY = (245, 170, 45)
    RETURN_PATH = (220, 80, 245)
    NEW_OBSERVATION = (255, 235, 80)
    UNDETECTED_HUMAN = (155, 160, 170)
    DETECTED_HUMAN = (255, 100, 100)

    def __init__(
        self, grid_map, config, title="Partial Costmap Evacuation",
        *, display_static_obstacle_map=None, perception_display_config=None,
        overlay_config=None,
    ) -> None:
        import pygame

        pygame.init()
        self.pygame = pygame
        self.screen = pygame.display.set_mode((1280, 900))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 17)
        self.small_font = pygame.font.SysFont("Arial", 14)
        self.title_font = pygame.font.SysFont("Arial", 23, bold=True)
        self.map_rect = pygame.Rect(15, 15, 730, 870)
        self.transform = WorldTransform(grid_map, self.map_rect)
        self.grid_map = grid_map
        display_static = (
            np.asarray(grid_map.occupancy, dtype=bool)
            if display_static_obstacle_map is None
            else np.asarray(display_static_obstacle_map, dtype=bool)
        )
        expected = (grid_map.height, grid_map.width)
        if display_static.shape != expected:
            raise ValueError(
                f"display static map shape={display_static.shape}, expected={expected}"
            )
        self.display_static_obstacle_map = display_static.copy()
        self.config = config
        self.perception_display_config = (
            perception_display_config or PerceptionMapDisplayConfig()
        )
        self.overlay_config = overlay_config or MapOverlayConfig()
        self.running = True
        self.paused = False

    def process_events(self) -> tuple[bool, bool]:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_ESCAPE:
                    self.running = False
                elif event.key == self.pygame.K_SPACE:
                    self.paused = not self.paused
        if self.paused:
            self.clock.tick(30)
        return self.running, self.paused

    @staticmethod
    def _risk_color(value, maximum, alpha=105):
        ratio = 0.0 if maximum <= 0.0 else float(np.clip(value / maximum, 0.0, 1.0))
        return (int(255 * ratio), int(150 * (1.0 - ratio)), 35, alpha)

    @classmethod
    def human_marker_color(cls, detected):
        """Return the display color without changing victim belief state."""
        return cls.DETECTED_HUMAN if bool(detected) else cls.UNDETECTED_HUMAN

    @staticmethod
    def exit_status_label(status):
        """Return the user-facing Enum value used beside an exit marker."""
        if not isinstance(status, ExitStatus):
            raise TypeError("exit display status must be ExitStatus")
        return status.value

    def _draw_belief_cells(self, belief):
        pygame = self.pygame
        cfg = self.perception_display_config
        finite = belief.final_cost_map[
            np.isfinite(belief.final_cost_map) & ~belief.blocked_mask
        ]
        base = float(self.config.base_cost)
        cost_max = max(float(finite.max()) if finite.size else base + 1.0, base + 1e-9)
        for gy in range(self.grid_map.height):
            for gx in range(self.grid_map.width):
                rect = self.transform.grid_rect(gx, gy)
                value = belief.final_cost_map[gy, gx]
                normalized = 1.0 if not np.isfinite(value) else float(np.clip(
                    (value - base) / max(cost_max - base, 1e-9), 0.0, 1.0
                ))
                color = cfg.color_for(
                    observed=bool(belief.observed_mask[gy, gx]),
                    blocked=bool(belief.blocked_mask[gy, gx]),
                    normalized_cost=normalized,
                )
                pygame.draw.rect(self.screen, color, rect)

    def _draw_blocked(self, belief):
        pygame = self.pygame
        for gy in range(self.grid_map.height):
            for gx in range(self.grid_map.width):
                rect = self.transform.grid_rect(gx, gy)
                if self.display_static_obstacle_map[gy, gx]:
                    pygame.draw.rect(self.screen, self.STATIC, rect)
                elif (
                    belief.blocked_mask[gy, gx]
                    and not belief.static_obstacle_map[gy, gx]
                ):
                    pygame.draw.rect(self.screen, self.BLOCKED, rect)
                if (
                    self.overlay_config.show_dynamic_obstacles
                    and belief.dynamic_inflated_obstacle_map[gy, gx]
                ):
                    pygame.draw.rect(
                        self.screen,
                        self.perception_display_config.dynamic_obstacle_color,
                        rect, 2,
                    )

    def _draw_sensor_area(self, state, camera, newly_observed_cells, gas_radius):
        pygame = self.pygame
        cam_x = state.x + camera.front_offset * math.cos(state.theta)
        cam_y = state.y + camera.front_offset * math.sin(state.theta)
        camera_screen = self.transform.world_to_screen(cam_x, cam_y)
        left_angle = state.theta + camera.fov_h / 2.0
        right_angle = state.theta - camera.fov_h / 2.0
        left = self.transform.world_to_screen(
            cam_x + camera.max_range * math.cos(left_angle),
            cam_y + camera.max_range * math.sin(left_angle),
        )
        right = self.transform.world_to_screen(
            cam_x + camera.max_range * math.cos(right_angle),
            cam_y + camera.max_range * math.sin(right_angle),
        )
        fov_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(
            fov_surface, (40, 210, 120, 35), [camera_screen, left, right]
        )
        pygame.draw.lines(
            fov_surface, (70, 245, 150, 150), False,
            [left, camera_screen, right], 2,
        )
        self.screen.blit(fov_surface, (0, 0))
        for gx, gy in newly_observed_cells:
            if self.grid_map.in_bounds((gx, gy)):
                pygame.draw.rect(
                    self.screen, self.NEW_OBSERVATION,
                    self.transform.grid_rect(gx, gy), 2,
                )
        center = self.transform.world_to_screen(state.x, state.y)
        radius_px = max(5, int(gas_radius * self.transform.scale))
        pygame.draw.circle(self.screen, (170, 80, 240), center, radius_px, 2)

    def _draw_paths(self, follower, trajectory):
        pygame = self.pygame
        if len(follower.remaining_grid_path()) >= 2:
            points = [
                self.transform.world_to_screen(*self.grid_map.grid_to_world(gx, gy))
                for gx, gy in follower.remaining_grid_path()
            ]
            pygame.draw.lines(self.screen, self.PATH, False, points, 4)
        if len(trajectory) >= 2:
            points = [self.transform.world_to_screen(x, y) for x, y in trajectory]
            pygame.draw.lines(self.screen, self.TRAJECTORY, False, points, 3)

    def _draw_travel_and_return(self, travel_history, return_path, blocked_grid):
        if len(travel_history) >= 2:
            points = [self.transform.world_to_screen(*item) for item in travel_history]
            self.pygame.draw.lines(self.screen, self.TRAVEL_HISTORY, False, points, 3)
        if len(return_path) >= 2:
            points = [self.transform.world_to_screen(*item) for item in return_path]
            self.pygame.draw.lines(self.screen, self.RETURN_PATH, False, points, 5)
            self.pygame.draw.circle(self.screen, (255, 255, 255), points[1], 6)
        if blocked_grid is not None and self.grid_map.in_bounds(blocked_grid):
            point = self.transform.world_to_screen(
                *self.grid_map.grid_to_world(*blocked_grid)
            )
            self.pygame.draw.line(self.screen, (255, 30, 30), (point[0] - 8, point[1] - 8), (point[0] + 8, point[1] + 8), 4)
            self.pygame.draw.line(self.screen, (255, 30, 30), (point[0] - 8, point[1] + 8), (point[0] + 8, point[1] - 8), 4)

    def _draw_markers(
        self, state, start, goal, humans=(), exits=(), detected_ids=(),
        exit_evaluations=(), selected_exit_id=None,
    ):
        pygame = self.pygame
        start_p = self.transform.world_to_screen(*start)
        goal_p = self.transform.world_to_screen(*goal)
        robot_p = self.transform.world_to_screen(state.x, state.y)
        pygame.draw.circle(self.screen, (80, 190, 255), start_p, 8)
        pygame.draw.circle(self.screen, (255, 225, 20), goal_p, 11)
        pygame.draw.circle(self.screen, (255, 255, 255), goal_p, 11, 2)
        pygame.draw.circle(self.screen, (30, 125, 255), robot_p, 10)
        heading = (
            robot_p[0] + int(25 * math.cos(state.theta)),
            robot_p[1] - int(25 * math.sin(state.theta)),
        )
        pygame.draw.line(self.screen, (255, 255, 255), robot_p, heading, 4)
        detected = set(detected_ids)
        for human in humans:
            if not self.overlay_config.show_detected_humans:
                continue
            point = self.transform.world_to_screen(human["x"], human["y"])
            is_detected = human["id"] in detected
            color = self.human_marker_color(is_detected)
            pygame.draw.circle(self.screen, color, point, 9)
            pygame.draw.circle(self.screen, (255, 255, 255), point, 9, 2)
            label = human["id"] if is_detected else f"{human['id']} (undetected)"
            self.screen.blit(
                self.small_font.render(label, True, color),
                (point[0] + 11, point[1] - 7),
            )
        evaluations = {item.exit_id: item for item in exit_evaluations}
        for exit_item in exits:
            approach = exit_item["approach"]
            point = self.transform.world_to_screen(approach["x"], approach["y"])
            evaluation = evaluations.get(exit_item["id"])
            if exit_item["id"] == selected_exit_id:
                color = (40, 245, 100)
            elif evaluation is not None and not evaluation.accepted:
                color = (235, 70, 70)
            elif evaluation is not None:
                color = (80, 210, 255)
            else:
                color = (210, 180, 75)
            pygame.draw.polygon(
                self.screen, color,
                [(point[0], point[1] - 9), (point[0] - 8, point[1] + 7),
                 (point[0] + 8, point[1] + 7)],
            )
            label = exit_item["id"]
            if evaluation is not None and evaluation.rejection_reasons:
                label += f": {evaluation.rejection_reasons[0].value}"
            self.screen.blit(
                self.small_font.render(label, True, color),
                (point[0] + 10, point[1] - 8),
            )

    def _draw_mini_layer(self, array, rect, title, blocked=None):
        pygame = self.pygame
        pygame.draw.rect(self.screen, (48, 52, 58), rect)
        values = np.asarray(array, dtype=float)
        finite = values[np.isfinite(values)]
        maximum = max(float(finite.max()) if finite.size else 1.0, 1e-9)
        cell_w = rect.width / values.shape[1]
        cell_h = rect.height / values.shape[0]
        for gy in range(values.shape[0]):
            for gx in range(values.shape[1]):
                if blocked is not None and blocked[gy, gx]:
                    color = self.BLOCKED
                elif not np.isfinite(values[gy, gx]):
                    color = self.STATIC
                else:
                    ratio = float(np.clip(values[gy, gx] / maximum, 0.0, 1.0))
                    color = (int(255 * ratio), int(170 * (1 - ratio)), 45)
                x = rect.x + int(gx * cell_w)
                y = rect.bottom - int((gy + 1) * cell_h)
                pygame.draw.rect(
                    self.screen, color,
                    pygame.Rect(x, y, max(1, int(cell_w + 1)), max(1, int(cell_h + 1))),
                )
        self.screen.blit(self.small_font.render(title, True, (240, 240, 240)),
                         (rect.x, rect.y - 18))

    def _draw_status(self, snapshot):
        x = 760
        y = 565
        lines = [
            f"FDS time: {snapshot['fds_time']:.2f} s",
            f"Thermal min/max: {snapshot['thermal_min']:.1f} / {snapshot['thermal_max']:.1f} °C",
            f"CO: {snapshot['co_text']}",
            f"Replans: {snapshot['replan_count']}",
            f"Last reason: {snapshot['last_replan_reason']}",
            f"Observed: {snapshot['observed_ratio']:.2f}%",
            f"Path cost: {snapshot['path_cost']}",
            f"Status: {snapshot['status']}",
            f"Mission: {snapshot['mission_state']}",
            f"Navigation: {snapshot['navigation_mode']}",
            f"Hazard knowledge: {snapshot.get('hazard_knowledge', 'UNDECIDED')}",
            f"Fire estimate: {snapshot.get('fire_estimate', 'UNOBSERVED')}",
            f"Strategy: {snapshot.get('evacuation_strategy', 'UNDECIDED')}",
            f"Route failure: {snapshot.get('route_failure', 'none')}",
            f"Path O/C/F: {snapshot.get('path_simplification', 'N/A')}",
            f"Exit plan: {snapshot['exit_plan']}",
            f"Victim state: {snapshot.get('victim_state', 'N/A')}",
            f"Following: {snapshot.get('following_active', False)}",
            f"Robot-victim: {snapshot.get('follow_distance', 'N/A')}",
            f"Follow target: {snapshot.get('target_follow_distance', 'N/A')}",
            f"Follow wait: {snapshot.get('follow_wait', False)}",
            "SPACE: pause/resume   ESC: quit",
        ]
        self.screen.blit(self.title_font.render("Simulation status", True, (245, 245, 245)), (x, y))
        y += 30
        for line in lines:
            self.screen.blit(self.small_font.render(line, True, (225, 225, 225)), (x, y))
            y += 18

    def draw(
        self, belief, state, start, goal, follower, trajectory, camera,
        newly_observed_cells, snapshot, humans=(), exits=(), detected_ids=(),
        travel_history=(), return_path=(), blocked_return_grid=None,
        exit_evaluations=(), selected_exit_id=None, path_simplification=None,
        victim_following=None, fire_localization=None, exit_states=None,
    ) -> None:
        snapshot = dict(snapshot)
        snapshot["observed_ratio"] = float(belief.observed_mask.mean() * 100.0)
        self.screen.fill(self.BACKGROUND)
        self._draw_belief_cells(belief)
        self._draw_blocked(belief)
        fire_visible = (
            fire_localization is not None
            and fire_localization.state.name in {
                "POSSIBLE_FIRE", "LIKELY_FIRE", "CONFIRMED_FIRE_REGION"
            }
        )
        if fire_localization is not None:
            snapshot["fire_estimate"] = (
                f"{fire_localization.state.name} "
                f"p={fire_localization.highest_probability:.3f}, "
                f"obs={fire_localization.valid_observation_count}"
            )
        if fire_visible and self.overlay_config.show_fire_candidates:
            for col, row in fire_localization.candidate_cells_grid:
                overlay = self.pygame.Surface(
                    self.transform.grid_rect(col, row).size,
                    self.pygame.SRCALPHA,
                )
                overlay.fill((255, 70, 20, 75))
                self.screen.blit(overlay, self.transform.grid_rect(col, row).topleft)
            if (
                self.overlay_config.show_estimated_fire_center
                and fire_localization.state.name in {
                    "LIKELY_FIRE", "CONFIRMED_FIRE_REGION"
                }
                and fire_localization.weighted_center_world is not None
            ):
                peak = self.transform.world_to_screen(
                    *fire_localization.weighted_center_world
                )
                self.pygame.draw.line(self.screen, (255, 30, 20),
                                      (peak[0] - 9, peak[1]), (peak[0] + 9, peak[1]), 3)
                self.pygame.draw.line(self.screen, (255, 30, 20),
                                      (peak[0], peak[1] - 9), (peak[0], peak[1] + 9), 3)
        self._draw_sensor_area(
            state, camera, newly_observed_cells, self.config.gas_update_radius
        )
        if self.overlay_config.show_current_path:
            self._draw_paths(follower, trajectory)
        if path_simplification is not None:
            original = [
                self.transform.world_to_screen(*self.grid_map.grid_to_world(*cell))
                for cell in path_simplification.original_path_grid
            ]
            final = [
                self.transform.world_to_screen(*point)
                for point in path_simplification.waypoints_world
            ]
            if len(original) >= 2:
                self.pygame.draw.lines(
                    self.screen, (145, 150, 160), False, original, 2
                )
            if len(final) >= 2:
                self.pygame.draw.lines(
                    self.screen, (40, 220, 90), False, final, 5
                )
            for cell in path_simplification.corner_path_grid:
                point = self.transform.world_to_screen(
                    *self.grid_map.grid_to_world(*cell)
                )
                self.pygame.draw.circle(self.screen, (255, 220, 35), point, 5)
            for point in final:
                self.pygame.draw.circle(self.screen, (40, 245, 110), point, 6, 2)
            for rejected in path_simplification.rejected_shortcuts[:12]:
                a = self.transform.world_to_screen(
                    *self.grid_map.grid_to_world(*rejected.start_grid)
                )
                b = self.transform.world_to_screen(
                    *self.grid_map.grid_to_world(*rejected.end_grid)
                )
                self.pygame.draw.line(self.screen, (235, 55, 55), a, b, 1)
        self._draw_travel_and_return(
            travel_history, return_path, blocked_return_grid
        )
        if victim_following is not None and victim_following.position_world is not None:
            history = [
                self.transform.world_to_screen(item.x, item.y)
                for item in victim_following.pose_history
            ]
            if len(history) >= 2:
                self.pygame.draw.lines(
                    self.screen, (100, 170, 255), False, history, 2
                )
            victim_point = self.transform.world_to_screen(
                *victim_following.position_world
            )
            robot_point = self.transform.world_to_screen(state.x, state.y)
            self.pygame.draw.line(
                self.screen, (255, 180, 70), robot_point, victim_point, 2
            )
            if victim_following.target_world is not None:
                target = self.transform.world_to_screen(
                    *victim_following.target_world
                )
                self.pygame.draw.circle(self.screen, (255, 220, 40), target, 6, 2)
            snapshot["victim_state"] = victim_following.state.name
            snapshot["following_active"] = victim_following.active
            snapshot["follow_distance"] = f"{math.dist((state.x, state.y), victim_following.position_world):.2f} m"
            snapshot["target_follow_distance"] = f"{victim_following.config.target_follow_distance_m:.2f} m"
            snapshot["follow_wait"] = victim_following.state.name == "FOLLOW_WAIT"
        self._draw_markers(
            state, start, goal, humans, exits, detected_ids,
            exit_evaluations, selected_exit_id,
        )
        if self.overlay_config.show_exit_states and exit_states:
            colors = {
                ExitStatus.UNKNOWN: (150, 150, 150),
                ExitStatus.USABLE: (40, 220, 230),
                ExitStatus.BLOCKED: (10, 10, 10),
                ExitStatus.DANGEROUS: (240, 45, 45),
            }
            for exit_item in exits:
                state_value = exit_states.get(exit_item["id"], ExitStatus.UNKNOWN)
                approach = exit_item["approach"]
                point = self.transform.world_to_screen(approach["x"], approach["y"])
                label = self.exit_status_label(state_value)
                self.screen.blit(
                    self.small_font.render(label, True, colors[state_value]),
                    (point[0] + 10, point[1] + 9),
                )
        self.pygame.draw.rect(self.screen, (210, 210, 210), self.map_rect, 2)
        self._draw_mini_layer(
            belief.temperature_cost_map,
            self.pygame.Rect(770, 55, 220, 205), "Temperature cost",
        )
        self._draw_mini_layer(
            belief.co_cost_map,
            self.pygame.Rect(1035, 55, 220, 205), "CO cost",
        )
        self._draw_mini_layer(
            belief.estimated_fire_cost_map,
            self.pygame.Rect(770, 340, 220, 205), "Estimated fire cost",
        )
        self._draw_mini_layer(
            belief.final_cost_map,
            self.pygame.Rect(1035, 340, 220, 205), "Final costmap",
            blocked=belief.blocked_mask,
        )
        self._draw_status(snapshot)
        self.pygame.display.flip()
        target_fps = max(1, int(round(1.0 / self.config.simulation_dt)))
        self.clock.tick(target_fps)

    def close(self):
        self.pygame.quit()


class MatplotlibThermalViewer:
    """Thermal image viewer that consumes raw Celsius arrays only."""

    def __init__(self) -> None:
        # Binary-extension import failures (for example Matplotlib compiled
        # against NumPy 1.x under NumPy 2.x) can print a full traceback before
        # raising the ImportError handled by the simulator.  Capture only that
        # noisy import output; the caller still reports a concise warning and
        # continues with Pygame.
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(6, 4.5))
        self.image = self.axis.imshow(
            np.full((24, 32), 25.0), origin="upper",
            cmap="inferno", vmin=20.0, vmax=120.0, aspect="auto",
        )
        self.axis.set_xlabel("pixel x")
        self.axis.set_ylabel("pixel y")
        self.figure.colorbar(self.image, ax=self.axis, label="Temperature [°C]")

    def update(self, temperature_celsius, fds_time):
        values = np.asarray(temperature_celsius, dtype=float)
        self.image.set_data(values)
        self.image.set_clim(20.0, max(60.0, float(np.nanmax(values))))
        self.axis.set_title(f"Thermal Camera | FDS t={fds_time:.2f}s")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self):
        self.plt.close(self.figure)


def show_debug_costmaps(belief):
    """Optional static Matplotlib analysis; never used as simulation input."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    layers = (
        (belief.temperature_cost_map, "Temperature cost"),
        (belief.co_cost_map, "CO cost"),
        (belief.final_cost_map, "Final costmap"),
    )
    for axis, (layer, title) in zip(axes, layers):
        image = axis.imshow(layer, origin="lower", cmap="magma")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    plt.show()
