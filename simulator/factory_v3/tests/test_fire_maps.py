import numpy as np
import pytest

from world.fire_maps import EstimatedFireMap, GroundTruthFireMap, MapMetadata


def metadata():
    return MapMetadata(1.0, 3.0, 5.0, 7.0, 1.0, 3, 3, (1.0, 5.0))


def test_world_grid_roundtrip_and_boundaries():
    item = metadata()
    for point in ((1, 5), (3, 7), (2.999, 6.999)):
        cell = item.world_to_grid(*point)
        assert item.is_grid_position_in_bounds(*cell)
        world = item.grid_to_world(*cell)
        assert abs(world[0] - point[0]) <= 0.501
        assert abs(world[1] - point[1]) <= 0.501
    assert not item.is_world_position_in_bounds(3.001, 7)
    with pytest.raises(ValueError):
        item.world_to_grid(3.001, 7)
    with pytest.raises(ValueError):
        item.grid_to_world(-1, 0)


def test_maps_are_independent_and_unobserved_is_explicit():
    gt = GroundTruthFireMap(metadata(), temperature_blocked_c=60, co_blocked_ppm=1600)
    estimated = EstimatedFireMap(metadata(), temperature_blocked_c=60, co_blocked_ppm=1600)
    gt.update_cell(1, 1, temperature_c=100, co_ppm=10, sim_time=1)
    assert np.isnan(estimated.temperature_c[1, 1])
    assert not estimated.observed_mask[1, 1]
    assert gt.temperature_c is not estimated.temperature_c


def test_partial_updates_and_block_thresholds_use_yx_order():
    fire = EstimatedFireMap(metadata(), temperature_blocked_c=60, co_blocked_ppm=1600)
    fire.update_cell(2, 0, temperature_c=59.9, co_ppm=1600, sim_time=2)
    assert fire.observed_mask[0, 2]
    assert fire.blocked_mask[0, 2]
    fire.update_cell(0, 2, temperature_c=60, co_ppm=0, sim_time=3)
    assert fire.blocked_mask[2, 0]
    assert fire.axis_order == "[row,col]=[y,x]"


def test_layer_shape_validation_and_copying():
    fire = EstimatedFireMap(metadata(), temperature_blocked_c=60, co_blocked_ppm=1600)
    source = np.ones((3, 3))
    fire.replace_layers(source, source, source.astype(bool), source)
    source[0, 0] = 99
    assert fire.temperature_c[0, 0] == 1
    with pytest.raises(ValueError):
        fire.replace_layers(np.ones((2, 2)), source, source, source)
