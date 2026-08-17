import pytest

from navigation.path_simplifier import extract_direction_change_points


@pytest.mark.parametrize("path,expected", [
    ([], ()),
    ([(1, 1)], ((1, 1),)),
    ([(1, 1), (2, 1)], ((1, 1), (2, 1))),
    ([(1, 1), (1, 1), (2, 1)], ((1, 1), (2, 1))),
    ([(0, 0), (1, 0), (2, 0), (3, 0)], ((0, 0), (3, 0))),
    ([(0, 0), (0, 1), (0, 2)], ((0, 0), (0, 2))),
    ([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
     ((0, 0), (2, 0), (2, 2))),
    ([(0, 0), (1, 1), (2, 2), (2, 3), (2, 4)],
     ((0, 0), (2, 2), (2, 4))),
    ([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
     ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2))),
])
def test_direction_change_extraction(path, expected):
    before = list(path)
    assert extract_direction_change_points(path) == expected
    assert path == before


def test_direction_is_normalized_for_long_segments():
    path = [(0, 0), (2, 0), (5, 0), (5, 3)]
    assert extract_direction_change_points(path) == ((0, 0), (5, 0), (5, 3))
