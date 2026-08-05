"""Validate ideal static fire costmap planning with complete FDS fields."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from mapping.fire_costmap import (
    FireCostmapConfig,
    FireCostmapLayers,
    build_fire_costmap,
)
from planner.a_star import AStarResult, weighted_a_star_with_escape


DEFAULT_VICTIM_POSITION = (17.0, 20.0)
DEFAULT_EXIT_POSITION = (20.0, 15.0)


def _path_length_metres(path: list[tuple[int, int]], resolution: float) -> float:
    return resolution * sum(
        math.hypot(x1 - x0, y1 - y0)
        for (x0, y0), (x1, y1) in zip(path, path[1:])
    )


def print_evaluation(
    layers: FireCostmapLayers,
    result: AStarResult,
    victim_world: tuple[float, float],
    exit_world: tuple[float, float],
    elapsed_seconds: float,
) -> None:
    """Print the required route and risk metrics."""

    grid_map = layers.grid_map
    victim_grid = grid_map.world_to_grid(*victim_world)
    exit_grid = grid_map.world_to_grid(*exit_world)
    blocked_count = int(layers.blocked_mask.sum())
    cell_count = int(layers.blocked_mask.size)

    print("\n=== Perfect costmap evacuation evaluation ===")
    print(
        "Selected FDS time: "
        f"temperature={layers.selected_temperature_time:.3f}s, "
        f"CO={layers.selected_co_time:.3f}s"
    )
    print(
        "Sampling height: "
        f"temperature={layers.selected_temperature_height:.2f}m, "
        f"CO={layers.selected_co_height:.2f}m"
    )
    print(f"Victim: world={victim_world} m, grid={victim_grid}")
    print(f"Exit: world={exit_world} m, grid={exit_grid}")
    print(f"Path found: {bool(result.path)} ({result.reason})")
    if result.escape_path:
        print(f"Escape path cells: {len(result.escape_path)}")
        print(f"Costmap replan start: {result.replan_start}")
    print(f"Path cells: {len(result.path)}")
    print(
        f"Path length: {_path_length_metres(result.path, grid_map.resolution):.3f} m"
    )
    print(f"Accumulated path cost: {result.total_cost:.6f}")

    if result.path:
        indices = tuple(np.asarray(result.path, dtype=int).T[::-1])
        path_temperatures = layers.temperature_map[indices]
        path_co = layers.co_map[indices]
        finite_temperatures = path_temperatures[np.isfinite(path_temperatures)]
        finite_co = path_co[np.isfinite(path_co)]
        print(f"Path maximum temperature: {np.max(finite_temperatures):.3f} °C")
        print(f"Path average temperature: {np.mean(finite_temperatures):.3f} °C")
        print(f"Path maximum CO: {np.max(finite_co):.3f} ppm")
        print(f"Path average CO: {np.mean(finite_co):.3f} ppm")
        invalid_samples = int(
            np.count_nonzero(~np.isfinite(path_temperatures))
            + np.count_nonzero(~np.isfinite(path_co))
        )
        if invalid_samples:
            print(
                "Invalid environmental samples on escape segment: "
                f"{invalid_samples} (excluded from risk statistics)"
            )
    else:
        print("Path maximum temperature: N/A")
        print("Path average temperature: N/A")
        print("Path maximum CO: N/A")
        print("Path average CO: N/A")

    print(f"Blocked cells: {blocked_count} / {cell_count}")
    print(f"Blocked ratio: {100.0 * blocked_count / cell_count:.3f}%")
    print(f"Execution time: {elapsed_seconds:.3f} s")


def visualize(
    layers: FireCostmapLayers,
    result: AStarResult,
    victim_world: tuple[float, float],
    exit_world: tuple[float, float],
    config: FireCostmapConfig,
    save_path: Path | None,
    show: bool,
) -> None:
    """Draw source fields, individual blocking layers, costmap and path."""

    grid = layers.grid_map
    extent = [grid.x_min, grid.x_max, grid.y_min, grid.y_max]
    figure, axes = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)

    static_image = axes[0, 0].imshow(
        layers.static_obstacle_map,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(["white", "black"]),
        vmin=0,
        vmax=1,
    )
    axes[0, 0].set_title("Static obstacles")
    figure.colorbar(static_image, ax=axes[0, 0], ticks=[0, 1])

    temperature_image = axes[0, 1].imshow(
        layers.temperature_map,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="inferno",
    )
    axes[0, 1].contour(
        layers.temperature_map,
        levels=[config.temperature_blocked],
        colors=["cyan"],
        origin="lower",
        extent=extent,
    )
    axes[0, 1].set_title(
        f"Temperature [°C], z={layers.selected_temperature_height:.2f} m"
    )
    figure.colorbar(temperature_image, ax=axes[0, 1])

    co_image = axes[0, 2].imshow(
        layers.co_map,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="viridis",
    )
    if np.nanmax(layers.co_map) >= config.co_blocked:
        axes[0, 2].contour(
            layers.co_map,
            levels=[config.co_blocked],
            colors=["magenta"],
            origin="lower",
            extent=extent,
        )
    axes[0, 2].set_title(f"CO [ppm], FDS slice z={layers.selected_co_height:.2f} m")
    figure.colorbar(co_image, ax=axes[0, 2])

    combined_risk = layers.temperature_cost_map + layers.co_cost_map
    risk_image = axes[1, 0].imshow(
        combined_risk,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="magma",
    )
    axes[1, 0].set_title("Temperature cost + CO cost")
    figure.colorbar(risk_image, ax=axes[1, 0])

    reason_map = np.zeros(layers.blocked_mask.shape, dtype=int)
    reason_map[layers.static_obstacle_map] = 1
    reason_map[layers.temperature_map >= config.temperature_blocked] = 2
    reason_map[layers.co_map >= config.co_blocked] = 3
    invalid = ~np.isfinite(layers.temperature_map) | ~np.isfinite(layers.co_map)
    reason_map[invalid & ~layers.static_obstacle_map] = 4
    blocked_image = axes[1, 1].imshow(
        reason_map,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(["white", "black", "red", "blue", "gray"]),
        vmin=0,
        vmax=4,
    )
    axes[1, 1].set_title("Blocked: static / temperature / CO / invalid")
    colorbar = figure.colorbar(blocked_image, ax=axes[1, 1], ticks=range(5))
    colorbar.ax.set_yticklabels(["free", "static", "temp", "CO", "invalid"])

    finite_cost = np.ma.masked_invalid(layers.final_cost_map)
    final_image = axes[1, 2].imshow(
        finite_cost,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="plasma",
    )
    axes[1, 2].imshow(
        np.ma.masked_where(~layers.blocked_mask, layers.blocked_mask),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(["black"]),
        alpha=0.75,
    )
    final_overlays = (
        (layers.static_obstacle_map, "black"),
        ((layers.temperature_map >= config.temperature_blocked)
         & ~layers.static_obstacle_map, "red"),
        ((layers.co_map >= config.co_blocked)
         & ~layers.static_obstacle_map
         & (layers.temperature_map < config.temperature_blocked), "blue"),
        (invalid & ~layers.static_obstacle_map, "gray"),
    )
    for mask, color in final_overlays:
        axes[1, 2].imshow(
            np.ma.masked_where(~mask, mask),
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=ListedColormap([color]),
            alpha=0.85,
        )
    figure.colorbar(final_image, ax=axes[1, 2], label="traversal cost")
    if result.path:
        path_world = np.asarray(
            [grid.grid_to_world(gx, gy) for gx, gy in result.path]
        )
        axes[1, 2].plot(
            path_world[:, 0], path_world[:, 1], color="lime", linewidth=2.5,
            label="A* path",
        )
    if len(result.escape_path) >= 2:
        escape_world = np.asarray(
            [grid.grid_to_world(gx, gy) for gx, gy in result.escape_path]
        )
        axes[1, 2].plot(
            escape_world[:, 0], escape_world[:, 1], color="cyan",
            linewidth=3.5, label="infinite-cost escape",
        )
    axes[1, 2].scatter(*victim_world, marker="o", s=90, color="deepskyblue",
                       edgecolor="white", label="victim", zorder=5)
    axes[1, 2].scatter(*exit_world, marker="*", s=180, color="yellow",
                       edgecolor="black", label="exit", zorder=5)
    axes[1, 2].set_title("Final costmap and evacuation path")
    handles, _ = axes[1, 2].get_legend_handles_labels()
    handles.extend([
        Patch(color="black", label="static blocked"),
        Patch(color="red", label="temperature blocked"),
        Patch(color="blue", label="CO blocked"),
        Patch(color="gray", label="invalid data"),
    ])
    axes[1, 2].legend(handles=handles, loc="best", fontsize="small")

    for axis in axes.flat:
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")
        axis.set_xlim(grid.x_min, grid.x_max)
        axis.set_ylim(grid.y_min, grid.y_max)
        axis.set_aspect("equal")

    figure.suptitle(
        "Perfect static costmap evacuation | "
        f"T={layers.selected_temperature_time:.1f}s, "
        f"temperature weight={config.temperature_weight:g}, "
        f"CO weight={config.co_weight:g}"
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150)
        print(f"Visualization saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--fds-file", type=Path, default=base_dir / "factory_v1.fds")
    parser.add_argument("--fds-dir", type=Path, default=base_dir)
    parser.add_argument(
        "--temperature-npz", type=Path,
        default=base_dir / "processed/fds_temperature_3d_timeseries.npz",
    )
    parser.add_argument("--time", type=float, default=60.0)
    parser.add_argument("--robot-height", type=float, default=1.3)
    parser.add_argument("--victim", type=float, nargs=2, default=DEFAULT_VICTIM_POSITION)
    parser.add_argument("--exit", dest="exit_position", type=float, nargs=2,
                        default=DEFAULT_EXIT_POSITION)
    parser.add_argument("--grid-resolution", type=float, default=0.5)
    parser.add_argument("--temperature-weight", type=float, default=8.0)
    parser.add_argument("--co-weight", type=float, default=8.0)
    parser.add_argument("--no-inflation", action="store_true")
    parser.add_argument("--inflation-radius", type=float, default=0.45)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = FireCostmapConfig(
        fds_time=args.time,
        robot_height=args.robot_height,
        grid_resolution=args.grid_resolution,
        temperature_weight=args.temperature_weight,
        co_weight=args.co_weight,
        use_inflation=not args.no_inflation,
        inflation_radius=args.inflation_radius,
    )
    started = time.perf_counter()
    layers = build_fire_costmap(
        args.fds_file, args.temperature_npz, args.fds_dir, config
    )
    victim_world = tuple(args.victim)
    exit_world = tuple(args.exit_position)
    result = weighted_a_star_with_escape(
        layers.final_cost_map,
        layers.grid_map.world_to_grid(*victim_world),
        layers.grid_map.world_to_grid(*exit_world),
        layers.static_obstacle_map,
    )
    elapsed = time.perf_counter() - started
    print_evaluation(layers, result, victim_world, exit_world, elapsed)
    visualize(
        layers, result, victim_world, exit_world, config, args.save, not args.no_show
    )
    return 0 if result.path else 2


if __name__ == "__main__":
    raise SystemExit(main())
