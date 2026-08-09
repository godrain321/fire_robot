"""Pygame main viewer and Matplotlib thermal-only viewer.

These classes consume simulation state and arrays for display. They never
produce sensor observations, costs, paths, or other simulation inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

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
    temperature_caution_c: float = 40.0
    temperature_danger_c: float = 60.0
    co_caution_ppm: float = 100.0
    co_danger_ppm: float = 1600.0

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
        thresholds = (
            self.temperature_caution_c, self.temperature_danger_c,
            self.co_caution_ppm, self.co_danger_ppm,
        )
        if not all(math.isfinite(value) for value in thresholds):
            raise ValueError("display hazard thresholds must be finite")
        if self.temperature_caution_c >= self.temperature_danger_c:
            raise ValueError("temperature caution threshold must be below danger")
        if self.co_caution_ppm >= self.co_danger_ppm:
            raise ValueError("CO caution threshold must be below danger")

    def color_for(
        self, *, observed, obstacle_blocked=False,
        temperature_c=None, co_ppm=None,
    ):
        """Classify by absolute observed hazards, never relative map cost."""
        if obstacle_blocked:
            return self.blocked_color
        if not observed:
            return self.unknown_color
        temperature_valid = temperature_c is not None and math.isfinite(temperature_c)
        co_valid = co_ppm is not None and math.isfinite(co_ppm)
        if (
            temperature_valid and temperature_c >= self.temperature_danger_c
        ) or (co_valid and co_ppm >= self.co_danger_ppm):
            return self.danger_color
        if (
            temperature_valid and temperature_c >= self.temperature_caution_c
        ) or (co_valid and co_ppm >= self.co_caution_ppm):
            return self.caution_color
        return self.safe_color


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
        self._grid_surface_rect = self._make_grid_surface_rect()
        self._static_slam_surface = self._make_static_slam_surface()
        self._costmap_surface = None
        self._costmap_surface_revision = None
        self._costmap_surface_build_count = 0
        self._mini_surface_cache = {}
        self._fov_surface = pygame.Surface(
            self.screen.get_size(), pygame.SRCALPHA
        )
        self._fire_overlay_surface = pygame.Surface(
            self.screen.get_size(), pygame.SRCALPHA
        )
        self._fire_overlay_key = None
        self._screen_point_cache = {}
        self._text_surface_cache = {}
        self._last_render_pose = None
        self.render_fps = int(getattr(config, "render_fps", 30))
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

    @classmethod
    def exit_status_display_label(cls, exit_id, status):
        """Format the sole text label shown for an exit."""
        return f"{exit_id} status: {cls.exit_status_label(status).upper()}"

    @staticmethod
    def perception_blocked_overlay(blocked, planner_static):
        """Keep inflated planner-static cells out of the SLAM wall display."""
        return bool(blocked and not planner_static)

    def _make_grid_surface_rect(self):
        half = self.grid_map.resolution / 2.0
        left, top = self.transform.world_to_screen(
            self.grid_map.x_min - half, self.grid_map.y_max + half
        )
        right, bottom = self.transform.world_to_screen(
            self.grid_map.x_max + half, self.grid_map.y_min - half
        )
        return self.pygame.Rect(
            min(left, right), min(top, bottom),
            max(1, abs(right - left)), max(1, abs(bottom - top)),
        )

    def _cached_screen_points(self, namespace, world_points):
        frozen = tuple((float(x), float(y)) for x, y in world_points)
        cached = self._screen_point_cache.get(namespace)
        if cached is None or cached[0] != frozen:
            points = tuple(self.transform.world_to_screen(x, y) for x, y in frozen)
            self._screen_point_cache[namespace] = (frozen, points)
            return points
        return cached[1]

    def _cached_text(self, text, font, color):
        key = (str(text), id(font), tuple(color))
        surface = self._text_surface_cache.get(key)
        if surface is None:
            surface = font.render(str(text), True, color)
            if len(self._text_surface_cache) >= 512:
                self._text_surface_cache.clear()
            self._text_surface_cache[key] = surface
        return surface

    def _surface_from_rgb_yx(self, rgb_yx, size):
        """Convert map[y,x,RGB] to a y-flipped, scaled Pygame Surface."""
        values = np.asarray(rgb_yx, dtype=np.uint8)
        expected = (self.grid_map.height, self.grid_map.width, 3)
        if values.shape != expected:
            raise ValueError(f"RGB map shape={values.shape}, expected={expected}")
        pixels_xy = np.ascontiguousarray(np.flipud(values).transpose(1, 0, 2))
        surface = self.pygame.surfarray.make_surface(pixels_xy)
        return self.pygame.transform.scale(surface, size)

    def _make_static_slam_surface(self):
        """Build the immutable non-inflated SLAM wall layer once."""
        transparent_key = (1, 2, 3)
        rgb = np.empty(
            (self.grid_map.height, self.grid_map.width, 3), dtype=np.uint8
        )
        rgb[:] = transparent_key
        rgb[self.display_static_obstacle_map] = self.STATIC
        surface = self._surface_from_rgb_yx(rgb, self._grid_surface_rect.size)
        surface.set_colorkey(transparent_key)
        return surface

    def _belief_rgb_array(self, belief):
        """Classify absolute observed temperature/CO without relative scaling."""
        cfg = self.perception_display_config
        observed = np.asarray(belief.observed_mask, dtype=bool)
        temperature_observed = np.asarray(
            belief.temperature_observed_mask, dtype=bool
        )
        co_observed = np.asarray(belief.co_observed_mask, dtype=bool)
        temperatures = np.asarray(belief.temperature_belief_map, dtype=float)
        co_values = np.asarray(belief.co_belief_map, dtype=float)
        planner_static = np.asarray(belief.static_obstacle_map, dtype=bool)
        expected = (self.grid_map.height, self.grid_map.width)
        if any(item.shape != expected for item in (
            observed, temperature_observed, co_observed, temperatures,
            co_values, planner_static,
        )):
            raise ValueError("belief display layers must match the grid shape")

        rgb = np.empty(expected + (3,), dtype=np.uint8)
        rgb[:] = cfg.unknown_color
        temperature_danger = temperature_observed & (
            temperatures >= cfg.temperature_danger_c
        )
        co_danger = co_observed & (co_values >= cfg.co_danger_ppm)
        danger = temperature_danger | co_danger
        caution = ~danger & (
            (temperature_observed & (
                temperatures >= cfg.temperature_caution_c
            ))
            | (co_observed & (co_values >= cfg.co_caution_ppm))
        )
        safe = observed & ~caution & ~danger
        rgb[safe] = cfg.safe_color
        rgb[caution] = cfg.caution_color
        rgb[danger] = cfg.danger_color
        # Planner static occupancy includes robot-clearance inflation.  It is
        # intentionally not painted here: the exact, non-inflated SLAM walls
        # are composited by ``_draw_blocked`` after this belief surface.
        if self.overlay_config.show_dynamic_obstacles:
            rgb[np.asarray(belief.dynamic_obstacle_map, dtype=bool)] = (
                cfg.dynamic_obstacle_color
            )
        return rgb

    def _draw_belief_cells(self, belief):
        if self._costmap_surface_revision != belief.revision:
            self._costmap_surface = self._surface_from_rgb_yx(
                self._belief_rgb_array(belief), self._grid_surface_rect.size
            )
            self._costmap_surface_revision = belief.revision
            self._costmap_surface_build_count += 1
            self._mini_surface_cache.clear()
        self.screen.blit(self._costmap_surface, self._grid_surface_rect)

    def _draw_blocked(self, belief):
        # Exact SLAM geometry is immutable and was rasterized once in __init__.
        self.screen.blit(self._static_slam_surface, self._grid_surface_rect)

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
        fov_surface = self._fov_surface
        fov_surface.fill((0, 0, 0, 0))
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
        remaining = tuple(follower.remaining_grid_path())
        if len(remaining) >= 2:
            points = self._cached_screen_points(
                "active_grid_path",
                (self.grid_map.grid_to_world(gx, gy) for gx, gy in remaining),
            )
            pygame.draw.lines(self.screen, self.PATH, False, points, 4)
        if len(trajectory) >= 2:
            points = self._cached_screen_points("trajectory", trajectory)
            pygame.draw.lines(self.screen, self.TRAJECTORY, False, points, 3)

    def _draw_travel_and_return(self, travel_history, return_path, blocked_grid):
        if len(travel_history) >= 2:
            points = self._cached_screen_points("travel_history", travel_history)
            self.pygame.draw.lines(self.screen, self.TRAVEL_HISTORY, False, points, 3)
        if len(return_path) >= 2:
            points = self._cached_screen_points("return_path", return_path)
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
                self._cached_text(label, self.small_font, color),
                (point[0] + 11, point[1] - 7),
            )
        for exit_item in exits:
            approach = exit_item["approach"]
            point = self.transform.world_to_screen(approach["x"], approach["y"])
            if exit_item["id"] == selected_exit_id:
                color = (40, 245, 100)
            else:
                color = (210, 180, 75)
            pygame.draw.polygon(
                self.screen, color,
                [(point[0], point[1] - 9), (point[0] - 8, point[1] + 7),
                 (point[0] + 8, point[1] + 7)],
            )

    def _draw_mini_layer(
        self, array, rect, title, blocked=None, *, revision=None,
    ):
        pygame = self.pygame
        pygame.draw.rect(self.screen, (48, 52, 58), rect)
        values = np.asarray(array, dtype=float)
        cache_key = (title, revision)
        surface = self._mini_surface_cache.get(cache_key)
        if surface is None:
            finite = values[np.isfinite(values)]
            maximum = max(float(finite.max()) if finite.size else 1.0, 1e-9)
            ratio = np.zeros(values.shape, dtype=float)
            finite_mask = np.isfinite(values)
            ratio[finite_mask] = np.clip(
                values[finite_mask] / maximum, 0.0, 1.0
            )
            rgb = np.empty(values.shape + (3,), dtype=np.uint8)
            rgb[..., 0] = (255 * ratio).astype(np.uint8)
            rgb[..., 1] = (170 * (1.0 - ratio)).astype(np.uint8)
            rgb[..., 2] = 45
            rgb[~finite_mask] = self.STATIC
            if blocked is not None:
                rgb[np.asarray(blocked, dtype=bool)] = self.BLOCKED
            surface = self._surface_from_rgb_yx(rgb, rect.size)
            self._mini_surface_cache[cache_key] = surface
        self.screen.blit(surface, rect)
        self.screen.blit(self._cached_text(title, self.small_font, (240, 240, 240)),
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
        self.screen.blit(
            self._cached_text(
                "Simulation status", self.title_font, (245, 245, 245)
            ),
            (x, y),
        )
        y += 30
        for line in lines:
            self.screen.blit(
                self._cached_text(line, self.small_font, (225, 225, 225)),
                (x, y),
            )
            y += 18

    def _draw_once(
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
            fire_key = (
                fire_localization.state.name,
                tuple(fire_localization.candidate_cells_grid),
            )
            if fire_key != self._fire_overlay_key:
                self._fire_overlay_surface.fill((0, 0, 0, 0))
                for col, row in fire_localization.candidate_cells_grid:
                    self.pygame.draw.rect(
                        self._fire_overlay_surface, (255, 70, 20, 75),
                        self.transform.grid_rect(col, row),
                    )
                self._fire_overlay_key = fire_key
            self.screen.blit(self._fire_overlay_surface, (0, 0))
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
            original = self._cached_screen_points(
                "simplification_original",
                (self.grid_map.grid_to_world(*cell)
                 for cell in path_simplification.original_path_grid),
            )
            final = self._cached_screen_points(
                "simplification_final", path_simplification.waypoints_world
            )
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
            history = self._cached_screen_points(
                "victim_pose_history",
                ((item.x, item.y) for item in victim_following.pose_history),
            )
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
            status_text_color = (240, 45, 45)
            for exit_item in exits:
                state_value = exit_states.get(exit_item["id"], ExitStatus.UNKNOWN)
                approach = exit_item["approach"]
                point = self.transform.world_to_screen(approach["x"], approach["y"])
                label = self.exit_status_display_label(
                    exit_item["id"], state_value
                )
                self.screen.blit(
                    self._cached_text(
                        label, self.small_font, status_text_color
                    ),
                    (point[0] + 10, point[1] - 8),
                )
        self.pygame.draw.rect(self.screen, (210, 210, 210), self.map_rect, 2)
        self._draw_mini_layer(
            belief.temperature_cost_map,
            self.pygame.Rect(770, 55, 220, 205), "Temperature cost",
            revision=belief.revision,
        )
        self._draw_mini_layer(
            belief.co_cost_map,
            self.pygame.Rect(1035, 55, 220, 205), "CO cost",
            revision=belief.revision,
        )
        self._draw_mini_layer(
            belief.estimated_fire_cost_map,
            self.pygame.Rect(770, 340, 220, 205), "Estimated fire cost",
            revision=belief.revision,
        )
        self._draw_mini_layer(
            belief.final_cost_map,
            self.pygame.Rect(1035, 340, 220, 205), "Final costmap",
            blocked=belief.blocked_mask, revision=belief.revision,
        )
        self._draw_status(snapshot)

    def draw(self, *args, **kwargs) -> None:
        """Run render-only interpolation between two simulation poses.

        This loop never advances simulation time, sensors, Costmap, or A*.
        The simulation supplies one pose per fixed 10 Hz update; this renderer
        independently emits the configured number of display-only frames.
        """
        if len(args) >= 2:
            state = args[1]
        elif "state" in kwargs:
            state = kwargs["state"]
        else:
            raise TypeError("draw requires belief and robot state")
        current = (float(state.x), float(state.y), float(state.theta))
        previous = self._last_render_pose or current
        frame_count = max(
            1, int(round(float(self.config.simulation_dt) * self.render_fps))
        )
        angle_delta = math.atan2(
            math.sin(current[2] - previous[2]),
            math.cos(current[2] - previous[2]),
        )
        for frame_index in range(frame_count):
            ratio = (frame_index + 1) / frame_count
            render_state = SimpleNamespace(
                x=previous[0] + (current[0] - previous[0]) * ratio,
                y=previous[1] + (current[1] - previous[1]) * ratio,
                theta=previous[2] + angle_delta * ratio,
            )
            if len(args) >= 2:
                frame_args = list(args)
                frame_args[1] = render_state
                self._draw_once(*frame_args, **kwargs)
            else:
                frame_kwargs = dict(kwargs)
                frame_kwargs["state"] = render_state
                self._draw_once(**frame_kwargs)
            self.pygame.display.flip()
            self.clock.tick(self.render_fps)
        self._last_render_pose = current

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
