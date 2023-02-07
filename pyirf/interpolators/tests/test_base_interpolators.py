"""Tests for base interpolator classes"""
import numpy as np
import pytest


def test_BaseInterpolator_datastructure_checks():
    """Test checks performed to enforce correct data structure"""
    from pyirf.interpolators.base_interpolators import BaseInterpolator

    grid_points1D_good = np.array([1, 2, 3])
    grid_points1D_bad = [1, 2, 3]
    target1D_bad = 1

    grid_points2D_good = np.array([[1, 1], [2, 1], [1.5, 1.5]])
    grid_points2D_bad_str = np.array([["1", "1"], ["2", "1"], ["1.5", "1.5"]])
    grid_points2D_bad_obj = np.array([[1, 1], [2, 1], [1.5]], dtype="object")
    target2D_bad = [1.25, 1.25]

    with pytest.raises(TypeError):  # Abstract class, cannot be instantiated
        BaseInterpolator(grid_points1D_good)

    class DummyBaseInterpolator(BaseInterpolator):
        def interpolate(self, target_point, **kwargs):
            raise NotImplementedError

    with pytest.raises(TypeError):  # Grid is a list
        DummyBaseInterpolator(grid_points1D_bad)

    with pytest.raises(TypeError):  # Grid is an array with dtype object
        DummyBaseInterpolator(grid_points2D_bad_obj)

    with pytest.raises(TypeError):  # Grid is an array of strings
        DummyBaseInterpolator(grid_points2D_bad_str)

    with pytest.raises(TypeError):  # Target is an integer
        Interp = DummyBaseInterpolator(grid_points1D_good)
        Interp(target1D_bad)

    with pytest.raises(TypeError):  # Target is a list
        Interp = DummyBaseInterpolator(grid_points2D_good)
        Interp(target2D_bad)


def test_BaseInterpolator_sanity_checks():
    """Test checks performed to enforce correct data values"""
    from pyirf.interpolators.base_interpolators import BaseInterpolator

    grid_points1D_good = np.array([1, 2, 3])
    target1D_inGrid = np.array([1.5])
    target1D_outofGrid = np.array([0.5])

    grid_points2D_good = np.array([[1, 1], [2, 1], [1.5, 1.5]])
    grid_points2D_toofew = np.array([[1, 1], [2, 1]])
    target2D_inGrid = np.array([1.25, 1.25])
    target2D_outofGrid = np.array([0.5, 0.5])
    target2D_twopoints = np.array([[1.25, 1.25], [1.25, 1.25]])

    class DummyBaseInterpolator(BaseInterpolator):
        def interpolate(self, target_point, **kwargs):
            raise NotImplementedError

    with pytest.raises(ValueError):  # Target out of grid
        interp = DummyBaseInterpolator(grid_points1D_good)
        interp(target1D_outofGrid)

    with pytest.raises(ValueError):  # To few grid points
        DummyBaseInterpolator(grid_points2D_toofew)

    with pytest.raises(ValueError):  # Two target points
        interp = DummyBaseInterpolator(grid_points2D_good)
        interp(target2D_twopoints)

    with pytest.raises(ValueError):  # Target out of grid
        interp = DummyBaseInterpolator(grid_points2D_good)
        interp(target2D_outofGrid)

    with pytest.raises(ValueError):  # 1D target in 2D grid
        interp = DummyBaseInterpolator(grid_points2D_good)
        interp(target1D_inGrid)

    with pytest.raises(NotImplementedError):
        # Everything ok but _interpolate not implemented
        interp = DummyBaseInterpolator(grid_points1D_good)
        interp(target1D_inGrid)

    with pytest.raises(NotImplementedError):
        # Everything ok but _interpolate not implemented
        interp = DummyBaseInterpolator(grid_points2D_good)
        interp(target2D_inGrid)


def test_BaseInterpolator_extrapolation():
    """Test extrapolators are correctly passed"""
    from pyirf.interpolators.base_interpolators import BaseInterpolator

    def dummy_extrapolator(target_point):
        return np.sum(target_point)

    class DummyBaseInterpolator(BaseInterpolator):
        def interpolate(self, target_point, **kwargs):
            raise NotImplementedError

    grid_points1D_good = np.array([1, 2, 3])
    target1D_outofGrid = np.array([0.5])

    grid_points2D_good = np.array([[1, 1], [2, 1], [1.5, 1.5]])
    target2D_outofGrid = np.array([0.5, 0.5])

    # 1D target out of grid but extrapolator given
    interp1D = DummyBaseInterpolator(grid_points1D_good)
    assert interp1D(target1D_outofGrid, extrapolator=dummy_extrapolator) == np.sum(
        target1D_outofGrid
    )

    # 2D target out of grid but extrapolator given
    interp2D = DummyBaseInterpolator(grid_points2D_good)
    assert interp2D(target2D_outofGrid, extrapolator=dummy_extrapolator) == np.sum(
        target2D_outofGrid
    )


def test_ParametrizedInterpolator():
    """Test ParametrizedInterpolator initialization and sanity checks"""
    from pyirf.interpolators.base_interpolators import ParametrizedInterpolator

    grid_points = np.array([1, 2, 3])
    params_good = np.array([[1], [2], [3]])
    params_shape_missmatch = np.array([[1], [2]])

    with pytest.raises(TypeError):  # Abstract class, cannot be instantiated
        ParametrizedInterpolator(grid_points, params_good)

    class DummyParametrizedInterpolator(ParametrizedInterpolator):
        def interpolate(self, target_point, **kwargs):
            raise NotImplementedError

    with pytest.raises(TypeError):  # parameters not a np.ndarray
        DummyParametrizedInterpolator(grid_points, params_good.tolist())

    with pytest.raises(ValueError):  # Fewer parameters then grid_points
        DummyParametrizedInterpolator(grid_points, params_shape_missmatch)

    interp = DummyParametrizedInterpolator(grid_points, params_good)

    with pytest.raises(NotImplementedError):
        # Everything ok but _interpolate not implemented
        interp(np.array([1]))


def test_BinnedInterpolator():
    """Test BinnedInterpolator initialization and sanity checks"""
    from pyirf.interpolators.base_interpolators import BinnedInterpolator

    grid_points = np.array([1, 2, 3])
    bin_edges = np.linspace(-1, 1, 11)
    bin_content_good_shape = np.ones(shape=(len(grid_points), len(bin_edges) - 1))
    bin_content_bad_nhist = np.ones(shape=(len(grid_points) - 1, len(bin_edges) - 1))
    bin_content_bad_nbins = np.ones(shape=(len(grid_points), len(bin_edges)))

    with pytest.raises(TypeError):  # Abstract class, cannot be instantiated
        BinnedInterpolator(grid_points, bin_edges, bin_content_good_shape)

    class DummyBinnedInterpolator(BinnedInterpolator):
        def interpolate(self, target_point, **kwargs):
            raise NotImplementedError

    with pytest.raises(TypeError):  # bin_edges are not a np.ndarray
        DummyBinnedInterpolator(grid_points, bin_edges.tolist(), bin_content_good_shape)

    with pytest.raises(TypeError):  # bin_contents are not a np.ndarray
        DummyBinnedInterpolator(grid_points, bin_edges, bin_content_good_shape.tolist())

    with pytest.raises(ValueError):  # Fewer entries in bin_contents then grid_points
        DummyBinnedInterpolator(grid_points, bin_edges, bin_content_bad_nhist)

    with pytest.raises(ValueError):
        # Fewer entries per bin-content then indicated by bin_edges
        DummyBinnedInterpolator(grid_points, bin_edges, bin_content_bad_nbins)

    interp = DummyBinnedInterpolator(grid_points, bin_edges, bin_content_good_shape)

    with pytest.raises(NotImplementedError):
        # Everything ok but _interpolate not implemented
        interp(np.array([1]))
