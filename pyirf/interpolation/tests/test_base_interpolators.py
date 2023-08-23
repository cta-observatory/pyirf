"""Tests for base interpolator classes"""
import numpy as np
import pytest


def test_BaseInterpolator_instantiation():
    """Test ParametrizedInterpolator initialization"""
    from pyirf.interpolation.base_interpolators import BaseInterpolator

    grid_points1D = np.array([1, 2, 3])
    target1D = np.array([[1]])

    grid_points2D = np.array([[1, 1], [2, 1], [1.5, 1.5]])
    target2D = np.array([[1.25, 1.25]])

    with pytest.raises(TypeError):  # Abstract class, cannot be instantiated
        BaseInterpolator(grid_points1D)

    class DummyBaseInterpolator(BaseInterpolator):
        def interpolate(self, target_point, **kwargs):
            return 42

    interp1D = DummyBaseInterpolator(grid_points1D)
    assert interp1D(target1D) == 42

    interp2D = DummyBaseInterpolator(grid_points2D)
    assert interp2D(target2D) == 42


def test_ParametrizedInterpolator_instantiation():
    """Test ParametrizedInterpolator initialization"""
    from pyirf.interpolation.base_interpolators import ParametrizedInterpolator

    grid_points1D = np.array([1, 2, 3])
    grid_points2D = np.array([[1, 1], [2, 1], [1.5, 1.5]])
    target1D = np.array([[1]])
    target2D = np.array([[1.25, 1.25]])

    params = np.array([[1], [2], [3]])

    with pytest.raises(TypeError):  # Abstract class, cannot be instantiated
        ParametrizedInterpolator(grid_points1D, params)

    class DummyParametrizedInterpolator(ParametrizedInterpolator):
        def interpolate(self, target_point, **kwargs):
            return 42

    interp1D = DummyParametrizedInterpolator(grid_points1D, params)
    assert interp1D(target1D) == 42

    interp2D = DummyParametrizedInterpolator(grid_points2D, params)
    assert interp2D(target2D) == 42


def test_DiscretePDFInterpolator_instantiation():
    """Test DiscretePDFInterpolator initialization and sanity checks"""
    from pyirf.interpolation.base_interpolators import DiscretePDFInterpolator

    grid_points1D = np.array([1, 2, 3])
    grid_points2D = np.array([[1, 1], [2, 1], [1.5, 1.5]])
    target1D = np.array([[1]])
    target2D = np.array([[1.25, 1.25]])

    bin_edges = np.linspace(-1, 1, 11)
    binned_pdf = np.ones(shape=(len(grid_points1D), len(bin_edges) - 1))

    with pytest.raises(TypeError):  # Abstract class, cannot be instantiated
        DiscretePDFInterpolator(grid_points1D, bin_edges, binned_pdf)

    class DummyBinnedInterpolator(DiscretePDFInterpolator):
        def interpolate(self, target_point, **kwargs):
            return 42

    interp1D = DummyBinnedInterpolator(grid_points1D, bin_edges, binned_pdf)
    assert interp1D(target1D) == 42

    interp2D = DummyBinnedInterpolator(grid_points2D, bin_edges, binned_pdf)
    assert interp2D(target2D) == 42


def test_virtual_subclasses():
    """Tests that corresponding nearest neighbor seacher are virtual sublasses of interpolators"""
    from pyirf.interpolation import (
        DiscretePDFInterpolator,
        DiscretePDFNearestNeighborSearcher,
        ParametrizedInterpolator,
        ParametrizedNearestNeighborSearcher,
    )

    assert issubclass(DiscretePDFNearestNeighborSearcher, DiscretePDFInterpolator)
    assert issubclass(ParametrizedNearestNeighborSearcher, ParametrizedInterpolator)
    assert not issubclass(ParametrizedNearestNeighborSearcher, DiscretePDFInterpolator)
    assert not issubclass(DiscretePDFNearestNeighborSearcher, ParametrizedInterpolator)
