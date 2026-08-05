import math


class GridMap:
    """A 2-D lattice map built from the FDS mesh and OBST/HOLE records."""

    def __init__(self, mesh_xb, obstacles, holes, resolution=0.5, clearance=0.45):
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")

        self.x_min, self.x_max = mesh_xb[0], mesh_xb[1]
        self.y_min, self.y_max = mesh_xb[2], mesh_xb[3]
        self.resolution = float(resolution)
        self.clearance = max(0.0, float(clearance))
        self.width = int(round((self.x_max - self.x_min) / self.resolution)) + 1
        self.height = int(round((self.y_max - self.y_min) / self.resolution)) + 1

        self.occupancy = [
            [False for _ in range(self.width)] for _ in range(self.height)
        ]
        self._rasterize(obstacles, holes)

    def world_to_grid(self, x, y):
        gx = int(round((x - self.x_min) / self.resolution))
        gy = int(round((y - self.y_min) / self.resolution))
        return gx, gy

    def grid_to_world(self, gx, gy):
        return (
            self.x_min + gx * self.resolution,
            self.y_min + gy * self.resolution,
        )

    def in_bounds(self, node):
        gx, gy = node
        return 0 <= gx < self.width and 0 <= gy < self.height

    def is_blocked(self, node):
        if not self.in_bounds(node):
            return True
        gx, gy = node
        return self.occupancy[gy][gx]

    def _rasterize(self, obstacles, holes):
        for gy in range(self.height):
            for gx in range(self.width):
                x, y = self.grid_to_world(gx, gy)
                blocked = any(self._inside_expanded_xb(x, y, item["xb"])
                              for item in obstacles)

                # A HOLE removes part of an OBST. Keep enough lateral clearance
                # for the robot while allowing its centre to cross the wall.
                if blocked and any(self._inside_safe_hole(x, y, item["xb"])
                                   for item in holes):
                    blocked = False

                self.occupancy[gy][gx] = blocked

    def _inside_expanded_xb(self, x, y, xb):
        x1, x2, y1, y2 = xb[:4]
        return (
            min(x1, x2) - self.clearance <= x <= max(x1, x2) + self.clearance
            and min(y1, y2) - self.clearance <= y <= max(y1, y2) + self.clearance
        )

    def _inside_safe_hole(self, x, y, xb):
        x1, x2, y1, y2 = xb[:4]
        left, right = min(x1, x2), max(x1, x2)
        bottom, top = min(y1, y2), max(y1, y2)
        return (
            left - self.clearance <= x <= right + self.clearance
            and bottom + self.clearance <= y <= top - self.clearance
        )


def robot_rotation_clearance(length, width, safety_margin=0.03):
    """Radius needed by a rectangular robot while rotating in place."""
    return math.hypot(length / 2.0, width / 2.0) + safety_margin
