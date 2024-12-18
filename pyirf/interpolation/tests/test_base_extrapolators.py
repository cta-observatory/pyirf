"""Tests for base extrapolator classes"""
import numpy as np
import pytest


def test_BaseExtrapolator_instantiation():
    """Test ParametrizedExtrapolator initialization"""
    from pyirf.interpolation.base_extrapolators import BaseExtrapolator

    grid_points1D = np.array([1, 2, 3])
    target1D = np.array([[0]])

    grid_points2D = np.array([[1, 1], [2, 1], [1.5, 1.5]])
    target2D = np.array([[0.25, 0.25]])

    with pytest.raises(TypeError):  # Abstract class, cannot be instantiated
        BaseExtrapolator(grid_points1D)

    class DummyBaseExtrapolator(BaseExtrapolator):
        def extrapolate(self, target_point):
            return 42

    interp1D = DummyBaseExtrapolator(grid_points1D)
    assert interp1D(target1D) == 42

    interp2D = DummyBaseExtrapolator(grid_points2D)
    assert interp2D(target2D) == 42


def test_ParametrizedExtrapolator_instantiation():
    """Test ParametrizedExtrapolator initialization"""
    from pyirf.interpolation.base_extrapolators import ParametrizedExtrapolator

    grid_points1D = np.array([1, 2, 3])
    grid_points2D = np.array([[1, 1], [2, 1], [1.5, 1.5]])
    target1D = np.array([[0]])
    target2D = np.array([[0.25, 0.25]])

    params = np.array([[1], [2], [3]])

    with pytest.raises(TypeError):  # Abstract class, cannot be instantiated
        ParametrizedExtrapolator(grid_points1D, params)

    class DummyParametrizedExtrapolator(ParametrizedExtrapolator):
        def extrapolate(self, target_point):
            return 42

    interp1D = DummyParametrizedExtrapolator(grid_points1D, params)
    assert interp1D(target1D) == 42

    interp2D = DummyParametrizedExtrapolator(grid_points2D, params)
    assert interp2D(target2D) == 42

    # If only one param per point exists and param.shape is not (n_points, 1)
    # they should be broadcasted internally
    interp1D = DummyParametrizedExtrapolator(grid_points1D, params.squeeze())
    assert interp1D(target1D) == 42


def test_DiscretePDFExtrapolator_instantiation():
    """Test DiscretePDFExtrapolator initialization and sanity checks"""
    from pyirf.interpolation.base_extrapolators import DiscretePDFExtrapolator

    grid_points1D = np.array([1, 2, 3])
    grid_points2D = np.array([[1, 1], [2, 1], [1.5, 1.5]])
    target1D = np.array([[0]])
    target2D = np.array([[0.25, 0.25]])

    bin_edges = np.linspace(-1, 1, 11)
    binned_pdf = np.ones(shape=(len(grid_points1D), len(bin_edges) - 1))

    with pytest.raises(TypeError):  # Abstract class, cannot be instantiated
        DiscretePDFExtrapolator(grid_points1D, bin_edges, binned_pdf)

    class DummyBinnedExtrapolator(DiscretePDFExtrapolator):
        def extrapolate(self, target_point):
            return 42

    interp1D = DummyBinnedExtrapolator(grid_points1D, bin_edges, binned_pdf)
    assert interp1D(target1D) == 42

    interp2D = DummyBinnedExtrapolator(grid_points2D, bin_edges, binned_pdf)
    assert interp2D(target2D) == 42


def test_virtual_subclasses():
    """Tests that corresponding nearest neighbor seacher are virtual sublasses of extrapolators"""
    from pyirf.interpolation import (
        DiscretePDFExtrapolator,
        DiscretePDFNearestNeighborSearcher,
        ParametrizedExtrapolator,
        ParametrizedNearestNeighborSearcher,
    )

    assert issubclass(DiscretePDFNearestNeighborSearcher, DiscretePDFExtrapolator)
    assert issubclass(ParametrizedNearestNeighborSearcher, ParametrizedExtrapolator)
    assert not issubclass(ParametrizedNearestNeighborSearcher, DiscretePDFExtrapolator)
    assert not issubclass(DiscretePDFNearestNeighborSearcher, ParametrizedExtrapolator)
