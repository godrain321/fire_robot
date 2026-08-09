#!/usr/bin/env python3
"""Record factory_v3 once, then replay its GUI without loading FDS/NPZ data.

Recording reuses the existing Pygame renderer and writes two artifacts:

* an MP4 containing the exact GUI frames at a fixed presentation frame rate;
* a gzip-compressed pickle stream containing each 0.1 s simulation snapshot,
  including robot pose, active path and robot-belief Costmap layers.

The pickle sidecar is local trusted data.  Do not open an untrusted replay file.
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


BASE = Path(__file__).resolve().parent
DEFAULT_VIDEO = BASE / "output/factory_v3_replay.mp4"
DEFAULT_DATA = BASE / "output/factory_v3_replay.pkl.gz"


def _finite_float(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _point(value):
    return tuple(_finite_float(item, "point coordinate") for item in value[:2])


def _enum_value(value):
    return getattr(value, "value", str(value))


class ReplayDataWriter:
    """Append independently compressed-friendly simulation-tick records."""

    def __init__(self, path: Path, *, grid_map, fps: int, simulation_dt: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = gzip.open(path, "wb", compresslevel=5)
        self.count = 0
        self._last_belief_revision = None
        pickle.dump({
            "format": "factory_v3_gui_replay",
            "version": 1,
            "render_fps": int(fps),
            "simulation_dt_s": float(simulation_dt),
            "grid": {
                "width": int(grid_map.width),
                "height": int(grid_map.height),
                "resolution_m": float(grid_map.resolution),
                "x_min": float(grid_map.x_min),
                "x_max": float(grid_map.x_max),
                "y_min": float(grid_map.y_min),
                "y_max": float(grid_map.y_max),
            },
        }, self.stream, protocol=pickle.HIGHEST_PROTOCOL)

    def write(self, args, kwargs) -> None:
        belief, state, start, goal, follower = args[:5]
        snapshot = dict(args[8])
        humans = args[9] if len(args) > 9 else ()
        exits = args[10] if len(args) > 10 else ()
        detected_ids = args[11] if len(args) > 11 else ()
        travel_history = args[12] if len(args) > 12 else ()
        return_path = args[13] if len(args) > 13 else ()
        selected_exit = args[16] if len(args) > 16 else None
        exit_states = kwargs.get("exit_states", {})
        revision = int(belief.revision)
        costmap_layers = None
        if revision != self._last_belief_revision:
            costmap_layers = {
                "observed_mask": np.asarray(
                    belief.observed_mask, dtype=bool
                ).copy(),
                "temperature_cost_map": np.asarray(
                    belief.temperature_cost_map, dtype=np.float32
                ).copy(),
                "co_cost_map": np.asarray(
                    belief.co_cost_map, dtype=np.float32
                ).copy(),
                "estimated_fire_cost_map": np.asarray(
                    belief.estimated_fire_cost_map, dtype=np.float32
                ).copy(),
                "final_cost_map": np.asarray(
                    belief.final_cost_map, dtype=np.float32
                ).copy(),
                "blocked_mask": np.asarray(
                    belief.blocked_mask, dtype=bool
                ).copy(),
            }
            self._last_belief_revision = revision
        record = {
            "index": self.count,
            "fds_time_s": _finite_float(snapshot["fds_time"], "FDS time"),
            "robot_pose_world": (
                float(state.x), float(state.y), float(state.theta)
            ),
            "start_world": _point(start),
            "goal_world": _point(goal),
            "remaining_path_grid": tuple(
                (int(col), int(row)) for col, row in follower.remaining_grid_path()
            ),
            # The ordered robot poses across all records are the actual path.
            # Only the latest history sample is repeated here to avoid O(n^2)
            # growth while retaining the displayed history update boundary.
            "travel_history_latest_world": (
                None if not travel_history else _point(travel_history[-1])
            ),
            "return_path_world": tuple(_point(item) for item in return_path),
            "belief_revision": revision,
            # None means reuse the exact arrays previously stored for this
            # revision. Thus every 0.1 s tick still has an exact Costmap.
            "costmap_layers": costmap_layers,
            "snapshot": snapshot,
            "humans": tuple(dict(item) for item in humans),
            "exits": tuple(dict(item) for item in exits),
            "detected_ids": tuple(sorted(str(item) for item in detected_ids)),
            "selected_exit_id": selected_exit,
            "exit_states": {
                str(key): _enum_value(value) for key, value in exit_states.items()
            },
        }
        pickle.dump(record, self.stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.count += 1

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class VideoEncoder:
    def __init__(self, path: Path, *, width: int, height: int, fps: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to record the replay MP4")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-pix_fmt", "yuv420p", str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write_surface(self, pygame, surface) -> None:
        if self.process.stdin is None:
            raise RuntimeError("video encoder is already closed")
        try:
            self.process.stdin.write(pygame.image.tostring(surface, "RGB"))
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg stopped while recording GUI frames") from exc

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        self.process = None
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")


def make_recording_viewer(base_viewer, *, video_path, data_path, fps, show):
    class RecordingViewer(base_viewer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.render_fps = int(fps)
            self._show_recording = bool(show)
            width, height = self.screen.get_size()
            self._encoder = VideoEncoder(
                video_path, width=width, height=height, fps=self.render_fps
            )
            self._data_writer = ReplayDataWriter(
                data_path, grid_map=self.grid_map, fps=self.render_fps,
                simulation_dt=float(self.config.simulation_dt),
            )
            self._recording_closed = False

        def process_events(self):
            if self._show_recording:
                return super().process_events()
            return True, False

        def draw(self, *args, **kwargs):
            self._data_writer.write(args, kwargs)
            state = args[1]
            current = (float(state.x), float(state.y), float(state.theta))
            previous = self._last_render_pose or current
            frame_count = max(1, int(round(
                float(self.config.simulation_dt) * self.render_fps
            )))
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
                frame_args = list(args)
                frame_args[1] = render_state
                self._draw_once(*frame_args, **kwargs)
                self._encoder.write_surface(self.pygame, self.screen)
                if self._show_recording:
                    self.pygame.display.flip()
            self._last_render_pose = current

        def close(self):
            if self._recording_closed:
                return
            self._recording_closed = True
            try:
                self._data_writer.close()
                self._encoder.close()
            finally:
                super().close()

    return RecordingViewer


def read_replay_info(path: Path) -> tuple[dict, int, float | None, float | None]:
    count = 0
    first_time = None
    last_time = None
    with gzip.open(path, "rb") as stream:
        header = pickle.load(stream)
        if header.get("format") != "factory_v3_gui_replay":
            raise ValueError("not a factory_v3 replay sidecar")
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            fds_time = float(record["fds_time_s"])
            first_time = fds_time if first_time is None else first_time
            last_time = fds_time
            count += 1
    return header, count, first_time, last_time


def record(args) -> int:
    if args.fps < 1:
        raise ValueError("fps must be at least 1")
    if args.output.resolve() == args.data.resolve():
        raise ValueError("video and sidecar paths must be different")
    if not args.show:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import run_partial_costmap_evacuation as simulation
    from visualization.partial_costmap_viewers import PygameSimulationViewer

    simulation.PygameSimulationViewer = make_recording_viewer(
        PygameSimulationViewer,
        video_path=args.output.resolve(), data_path=args.data.resolve(),
        fps=args.fps, show=args.show,
    )
    simulation_arguments = list(args.simulation_args)
    if simulation_arguments[:1] == ["--"]:
        simulation_arguments.pop(0)
    simulation_arguments = [
        item for item in simulation_arguments if item != "--headless"
    ]
    if "--no-thermal-window" not in simulation_arguments:
        simulation_arguments.append("--no-thermal-window")
    saved_argv = sys.argv
    try:
        sys.argv = ["run_partial_costmap_evacuation.py"] + simulation_arguments
        simulation_args = simulation.apply_scenario_config(simulation.parse_args())
        simulation_args.headless = False
        success, _, _, elapsed = simulation.run_simulation(simulation_args)
    finally:
        sys.argv = saved_argv
    header, count, first_time, last_time = read_replay_info(args.data.resolve())
    print(f"Video: {args.output.resolve()}")
    print(f"Replay data: {args.data.resolve()}")
    print(f"Simulation ticks: {count}")
    print(f"FDS time range: {first_time} .. {last_time} s")
    print(f"Render FPS: {header['render_fps']}")
    print(f"Simulation elapsed: {elapsed:.3f} s; mission success={success}")
    return 0


def play(args) -> int:
    if not args.input.is_file():
        raise ValueError(f"replay video does not exist: {args.input}")
    ffplay = shutil.which("ffplay")
    if ffplay is None:
        raise RuntimeError("ffplay is required to play the replay MP4")
    return subprocess.call([
        ffplay, "-autoexit", "-loglevel", "warning", str(args.input.resolve())
    ])


def info(args) -> int:
    header, count, first_time, last_time = read_replay_info(args.input)
    print(f"Format: {header['format']} v{header['version']}")
    print(f"Simulation ticks: {count}")
    print(f"Simulation dt: {header['simulation_dt_s']} s")
    print(f"Render FPS: {header['render_fps']}")
    print(f"FDS time range: {first_time} .. {last_time} s")
    print(f"Grid: {header['grid']}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--output", type=Path, default=DEFAULT_VIDEO)
    record_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    record_parser.add_argument("--fps", type=int, default=30)
    record_parser.add_argument("--show", action="store_true")
    record_parser.add_argument(
        "simulation_args", nargs=argparse.REMAINDER,
        help="arguments for run_partial_costmap_evacuation.py after --",
    )
    record_parser.set_defaults(handler=record)

    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--input", type=Path, default=DEFAULT_VIDEO)
    play_parser.set_defaults(handler=play)

    info_parser = subparsers.add_parser("info")
    info_parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    info_parser.set_defaults(handler=info)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
