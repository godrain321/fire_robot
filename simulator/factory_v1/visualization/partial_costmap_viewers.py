"""Pygame main viewer and Matplotlib thermal-only viewer.

These classes consume simulation state and arrays for display. They never
produce sensor observations, costs, paths, or other simulation inputs.
"""

from __future__ import annotations

import math

import numpy as np


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
    NEW_OBSERVATION = (255, 235, 80)

    def __init__(self, grid_map, config, title="Partial Costmap Evacuation") -> None:
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
        self.config = config
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

    def _draw_belief_cells(self, belief):
        pygame = self.pygame
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        finite = belief.final_cost_map[np.isfinite(belief.final_cost_map)]
        cost_max = max(float(finite.max()) if finite.size else 1.0, 1.0)
        for gy in range(self.grid_map.height):
            for gx in range(self.grid_map.width):
                rect = self.transform.grid_rect(gx, gy)
                base = self.OBSERVED if belief.observed_mask[gy, gx] else self.UNKNOWN
                pygame.draw.rect(self.screen, base, rect)
                if np.isfinite(belief.final_cost_map[gy, gx]):
                    # Display-only normalization; the underlying cost is untouched.
                    pygame.draw.rect(
                        overlay,
                        self._risk_color(belief.final_cost_map[gy, gx], cost_max),
                        rect,
                    )
        self.screen.blit(overlay, (0, 0))

    def _draw_blocked(self, belief):
        pygame = self.pygame
        for gy in range(self.grid_map.height):
            for gx in range(self.grid_map.width):
                rect = self.transform.grid_rect(gx, gy)
                if belief.static_obstacle_map[gy, gx]:
                    pygame.draw.rect(self.screen, self.STATIC, rect)
                elif belief.blocked_mask[gy, gx]:
                    pygame.draw.rect(self.screen, self.BLOCKED, rect)

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

    def _draw_markers(self, state, start, goal):
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
        y = 600
        lines = [
            f"FDS time: {snapshot['fds_time']:.2f} s",
            f"Thermal min/max: {snapshot['thermal_min']:.1f} / {snapshot['thermal_max']:.1f} °C",
            f"CO: {snapshot['co_text']}",
            f"Replans: {snapshot['replan_count']}",
            f"Last reason: {snapshot['last_replan_reason']}",
            f"Observed: {snapshot['observed_ratio']:.2f}%",
            f"Path cost: {snapshot['path_cost']}",
            f"Status: {snapshot['status']}",
            "SPACE: pause/resume   ESC: quit",
        ]
        self.screen.blit(self.title_font.render("Simulation status", True, (245, 245, 245)), (x, y))
        y += 34
        for line in lines:
            self.screen.blit(self.font.render(line, True, (225, 225, 225)), (x, y))
            y += 27

    def draw(
        self, belief, state, start, goal, follower, trajectory, camera,
        newly_observed_cells, snapshot,
    ) -> None:
        snapshot = dict(snapshot)
        snapshot["observed_ratio"] = float(belief.observed_mask.mean() * 100.0)
        self.screen.fill(self.BACKGROUND)
        self._draw_belief_cells(belief)
        self._draw_blocked(belief)
        self._draw_sensor_area(
            state, camera, newly_observed_cells, self.config.gas_update_radius
        )
        self._draw_paths(follower, trajectory)
        self._draw_markers(state, start, goal)
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
            belief.final_cost_map,
            self.pygame.Rect(900, 340, 220, 205), "Final costmap",
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
