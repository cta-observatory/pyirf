import numpy as np
import pytest


@pytest.fixture
def grid_1d():
    return np.array([[1], [2], [3], [4]])


@pytest.fixture
def grid_2d():
    return np.array([[1, 1], [1, 2], [2, 1], [2, 2]])


@pytest.fixture
def contents(grid_1d):
    return np.array(
        [
            [[np.full(10, x), np.full(10, x / 2)], [np.full(10, 2 * x), np.zeros(10)]]
            for x in grid_1d
        ]
    )


def test_NearestNeighborSearcher_1DGrid(grid_1d, contents):
    from pyirf.interpolation import NearestNeighborSearcher

    searcher = NearestNeighborSearcher(grid_1d, contents, norm_ord=2)

    target = np.array([[0]])
    assert np.array_equal(searcher(target), contents[0, :])

    target = np.array([[1.9]])
    assert np.array_equal(searcher(target), contents[1, :])


def test_NearestNeighborSearcher_2DGrid(grid_2d, contents):
    from pyirf.interpolation import NearestNeighborSearcher

    searcher = NearestNeighborSearcher(grid_2d, contents, norm_ord=2)

    target = np.array([[0, 1]])
    assert np.array_equal(searcher(target), contents[0, :])

    target = np.array([[3, 3]])
    assert np.array_equal(searcher(target), contents[-1, :])


def test_NearestNeighborSearcher_manhatten_norm(grid_2d, contents):
    from pyirf.interpolation import NearestNeighborSearcher

    searcher = NearestNeighborSearcher(grid_2d, contents, norm_ord=1)

    target = np.array([[0, 1]])
    assert np.array_equal(searcher(target), contents[0, :])

    target = np.array([[3, 3]])
    assert np.array_equal(searcher(target), contents[-1, :])


def test_NearestNeighborSearcher_wrong_norm(grid_1d, contents):
    from pyirf.interpolation import NearestNeighborSearcher

    with pytest.raises(ValueError, match="Only positiv integers allowed for norm_ord"):
        searcher = NearestNeighborSearcher(grid_1d, contents, norm_ord=-2)

    with pytest.raises(ValueError, match="Only positiv integers allowed for norm_ord"):
        searcher = NearestNeighborSearcher(grid_1d, contents, norm_ord=1.5)

    with pytest.raises(ValueError, match="Only positiv integers allowed for norm_ord"):
        searcher = NearestNeighborSearcher(grid_1d, contents, norm_ord=np.inf)

    with pytest.raises(ValueError, match="Only positiv integers allowed for norm_ord"):
        searcher = NearestNeighborSearcher(grid_1d, contents, norm_ord="nuc")
