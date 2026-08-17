import gzip
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

from record_simulation_replay import ReplayDataWriter, read_replay_info


class Follower:
    def remaining_grid_path(self):
        return ((1, 2), (2, 2))


def test_replay_sidecar_keeps_tick_poses_and_revision_costmaps(tmp_path):
    grid = SimpleNamespace(
        width=3, height=2, resolution=0.2,
        x_min=0.0, x_max=0.4, y_min=0.0, y_max=0.2,
    )
    shape = (2, 3)
    belief = SimpleNamespace(
        revision=4,
        observed_mask=np.ones(shape, dtype=bool),
        temperature_cost_map=np.full(shape, 1.0),
        co_cost_map=np.full(shape, 2.0),
        estimated_fire_cost_map=np.full(shape, 3.0),
        final_cost_map=np.full(shape, 4.0),
        blocked_mask=np.zeros(shape, dtype=bool),
    )
    state = SimpleNamespace(x=1.0, y=2.0, theta=0.5)
    snapshot = {"fds_time": 5.0, "status": "moving"}
    arguments = [
        belief, state, (1.0, 2.0), (2.0, 2.0), Follower(), (), None, (),
        snapshot, (), (), (), ((1.0, 2.0),), (), None, (), "EXIT2",
    ]
    path = tmp_path / "replay.pkl.gz"
    writer = ReplayDataWriter(
        path, grid_map=grid, fps=30, simulation_dt=0.1
    )
    writer.write(arguments, {"exit_states": {}})
    state.x = 1.1
    snapshot["fds_time"] = 5.1
    writer.write(arguments, {"exit_states": {}})
    writer.close()

    header, count, first_time, last_time = read_replay_info(path)
    assert header["render_fps"] == 30
    assert (count, first_time, last_time) == (2, 5.0, 5.1)
    with gzip.open(path, "rb") as stream:
        pickle.load(stream)
        first = pickle.load(stream)
        second = pickle.load(stream)
    assert first["robot_pose_world"] == (1.0, 2.0, 0.5)
    assert second["robot_pose_world"] == (1.1, 2.0, 0.5)
    assert np.array_equal(
        first["costmap_layers"]["final_cost_map"],
        np.full(shape, 4.0, dtype=np.float32),
    )
    assert second["costmap_layers"] is None


def test_default_replay_outputs_stay_inside_factory_v5():
    from record_simulation_replay import DEFAULT_DATA, DEFAULT_VIDEO

    expected = Path(__file__).resolve().parents[1] / "output"
    assert DEFAULT_VIDEO.parent == expected
    assert DEFAULT_DATA.parent == expected
