import numpy as np
import pytest


@pytest.fixture
def grid_1d():
    return np.array([[1], [2], [3], [4]])


@pytest.fixture
def grid_2d():
    return np.array([[1, 1], [1, 2], [2, 1], [2, 2]])


@pytest.fixture
def binned_pdf(grid_1d):
    return np.array(
        [
            [[np.full(10, x), np.full(10, x / 2)], [np.full(10, 2 * x), np.zeros(10)]]
            for x in grid_1d
        ]
    )


def test_BaseNearestNeighborSearcher_1DGrid(grid_1d, binned_pdf):
    from pyirf.interpolation import BaseNearestNeighborSearcher

    searcher = BaseNearestNeighborSearcher(grid_1d, binned_pdf, norm_ord=2)

    target = np.array([0])
    assert np.array_equal(searcher(target), binned_pdf[0, :])

    target = np.array([[1.9]])
    assert np.array_equal(searcher(target), binned_pdf[1, :])


def test_BaseNearestNeighborSearcher_2DGrid(grid_2d, binned_pdf):
    from pyirf.interpolation import BaseNearestNeighborSearcher

    searcher = BaseNearestNeighborSearcher(grid_2d, binned_pdf, norm_ord=2)

    target = np.array([[0, 1]])
    assert np.array_equal(searcher(target), binned_pdf[0, :])

    target = np.array([3, 3])
    assert np.array_equal(searcher(target), binned_pdf[-1, :])


def test_BaseNearestNeighborSearcher_manhatten_norm(grid_2d, binned_pdf):
    from pyirf.interpolation import BaseNearestNeighborSearcher

    searcher = BaseNearestNeighborSearcher(grid_2d, binned_pdf, norm_ord=1)

    target = np.array([[0, 1]])
    assert np.array_equal(searcher(target), binned_pdf[0, :])

    target = np.array([[3, 3]])
    assert np.array_equal(searcher(target), binned_pdf[-1, :])


def test_BaseNearestNeighborSearcher_wrong_norm(grid_1d, binned_pdf):
    from pyirf.interpolation import BaseNearestNeighborSearcher

    with pytest.raises(ValueError, match="Only positiv integers allowed for norm_ord"):
        BaseNearestNeighborSearcher(grid_1d, binned_pdf, norm_ord=-2)

    with pytest.raises(ValueError, match="Only positiv integers allowed for norm_ord"):
        BaseNearestNeighborSearcher(grid_1d, binned_pdf, norm_ord=1.5)

    with pytest.raises(ValueError, match="Only positiv integers allowed for norm_ord"):
        BaseNearestNeighborSearcher(grid_1d, binned_pdf, norm_ord=np.inf)

    with pytest.raises(ValueError, match="Only positiv integers allowed for norm_ord"):
        BaseNearestNeighborSearcher(grid_1d, binned_pdf, norm_ord="nuc")


def test_DiscretePDFNearestNeighborSearcher(grid_2d, binned_pdf):
    from pyirf.interpolation import DiscretePDFNearestNeighborSearcher

    bin_edges = np.linspace(0, 1, binned_pdf.shape[-1] + 1)

    searcher = DiscretePDFNearestNeighborSearcher(
        grid_points=grid_2d, bin_edges=bin_edges, binned_pdf=binned_pdf, norm_ord=1
    )

    target = np.array([[0, 1]])
    assert np.array_equal(searcher(target), binned_pdf[0, :])

    target = np.array([[3, 3]])
    assert np.array_equal(searcher(target), binned_pdf[-1, :])


def test_ParametrizedNearestNeighborSearcher(grid_2d, binned_pdf):
    from pyirf.interpolation import ParametrizedNearestNeighborSearcher

    searcher = ParametrizedNearestNeighborSearcher(
        grid_points=grid_2d, params=binned_pdf, norm_ord=1
    )

    target = np.array([[0, 1]])
    assert np.array_equal(searcher(target), binned_pdf[0, :])

    target = np.array([[3, 3]])
    assert np.array_equal(searcher(target), binned_pdf[-1, :])
