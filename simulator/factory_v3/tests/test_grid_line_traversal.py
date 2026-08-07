import pytest

from navigation.path_simplifier import cells_touched_by_segment


@pytest.mark.parametrize("start,end,required", [
    ((0, 0), (3, 0), {(0, 0), (1, 0), (2, 0), (3, 0)}),
    ((1, 0), (1, 3), {(1, 0), (1, 1), (1, 2), (1, 3)}),
    ((0, 0), (2, 2), {(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)}),
    ((0, 0), (3, 1), {(0, 0), (1, 0), (2, 1), (3, 1)}),
    ((3, 2), (0, 0), {(3, 2), (0, 0)}),
])
def test_supercover_contains_required_cells(start, end, required):
    cells = cells_touched_by_segment(start, end)
    assert cells[0] == start
    assert cells[-1] == end
    assert required.issubset(set(cells))
    assert len(cells) == len(set(cells))


def test_single_cell_segment():
    assert cells_touched_by_segment((2, 3), (2, 3)) == ((2, 3),)


def test_supercover_is_deterministic():
    first = cells_touched_by_segment((0, 0), (5, 2))
    assert first == cells_touched_by_segment((0, 0), (5, 2))
