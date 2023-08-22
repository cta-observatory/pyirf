"""Tests for estimator base classes """
import re

import numpy as np
import pytest


def test_BaseComponentEstimator_grid_checks():
    """Test checks for the grid perfomed by the base estimator class"""
    from pyirf.interpolation.component_estimators import BaseComponentEstimator

    grid_points1D_bad = [1, 2, 3]
    grid_points1D_toofew = np.array([1])

    grid_points2D_bad_str = np.array([["1", "1"], ["2", "1"], ["1.5", "1.5"]])
    grid_points2D_bad_obj = np.array([[1, 1], [2, 1], [1.5]], dtype="object")
    grid_points2D_toofew = np.array([[1, 1], [2, 1]])

    with pytest.raises(TypeError, match="Input grid_points is not a numpy array."):
        BaseComponentEstimator(grid_points1D_bad)

    with pytest.raises(
        TypeError, match="Input grid_points array cannot be of dtype object."
    ):
        BaseComponentEstimator(grid_points2D_bad_obj)

    with pytest.raises(
        TypeError, match="Input grid_points dtype incompatible with float."
    ):
        BaseComponentEstimator(grid_points2D_bad_str)

    with pytest.raises(ValueError, match="To few points for grid dimension"):
        BaseComponentEstimator(grid_points1D_toofew)

    with pytest.raises(ValueError, match="To few points for grid dimension"):
        BaseComponentEstimator(grid_points2D_toofew)


def test_BaseComponentEstimator_target_point_checks():
    """Test checks for the target point perfomed by the base estimator class"""
    from pyirf.interpolation.component_estimators import BaseComponentEstimator

    # Create a DummyEstimator as BaseComponentEstimator does not set an
    # Interpolator or Extrapolator class which is needed for __call__
    def dummy_interp(target_point):
        return 42

    class DummyEstimator(BaseComponentEstimator):
        def __init__(self, grid_points):
            super().__init__(grid_points)
            self.interpolator = dummy_interp
            self.extrapolator = None

    grid_points1D_good = np.array([1, 2, 3])
    target1D_bad_type = 1
    target1D_outofGrid = np.array([0.5])
    target1D_inGrid = np.array([1.5])

    grid_points2D_good = np.array([[1, 1], [2, 1], [1.5, 1.5]])
    target2D_bad_type = [1.25, 1.25]
    target2D_outofGrid = np.array([0.5, 0.5])
    target2D_twopoints = np.array([[1.25, 1.25], [1.25, 1.25]])

    with pytest.raises(TypeError, match="Target point is not a numpy array."):
        interp = DummyEstimator(grid_points1D_good)
        interp(target1D_bad_type)

    with pytest.raises(
        ValueError,
        match="Target point outside grids convex hull and no extrapolator given.",
    ):
        interp = DummyEstimator(grid_points1D_good)
        interp(target1D_outofGrid)

    with pytest.raises(TypeError, match="Target point is not a numpy array."):
        interp = DummyEstimator(grid_points2D_good)
        interp(target2D_bad_type)

    with pytest.raises(
        ValueError,
        match="Target point outside grids convex hull and no extrapolator given.",
    ):
        interp = DummyEstimator(grid_points2D_good)
        interp(target2D_outofGrid)

    with pytest.raises(ValueError, match="Only one target_point per call supported."):
        interp = DummyEstimator(grid_points2D_good)
        interp(target2D_twopoints)

    with pytest.raises(
        ValueError, match="Missmatch between target-point and grid dimension."
    ):
        interp = DummyEstimator(grid_points2D_good)
        interp(target1D_inGrid)


def test_BaseComponentEstimator_call():
    """Test base_estimator's __call__"""
    from pyirf.interpolation.component_estimators import BaseComponentEstimator

    grid_points1D_good = np.array([1, 2, 3])
    target1D_outofGrid = np.array([0.5])
    target1D_inGrid = np.array([1.5])

    grid_points2D_good = np.array([[1, 1], [2, 1], [1.5, 1.5]])
    target2D_outofGrid = np.array([0.5, 0.5])
    target2D_inGrid = np.array([1.25, 1.25])

    # Create a DummyEstimators as BaseComponentEstimator does not set an
    # Interpolator or Extrapolator class which is needed for __call__
    def dummy_interp(target_point):
        return 42

    def dummy_extrap(target_point):
        return 43

    class DummyEstimator(BaseComponentEstimator):
        def __init__(self, grid_points):
            super().__init__(grid_points)
            self.interpolator = dummy_interp
            self.extrapolator = dummy_extrap

    estim1D = DummyEstimator(grid_points1D_good)
    assert estim1D(target1D_inGrid) == 42

    estim2D = DummyEstimator(grid_points2D_good)
    assert estim2D(target2D_inGrid) == 42

    estim1D = DummyEstimator(grid_points1D_good)
    with pytest.warns(UserWarning, match="has to be extrapolated"):
        assert estim1D(target1D_outofGrid) == 43

    estim2D = DummyEstimator(grid_points2D_good)
    with pytest.warns(UserWarning, match="has to be extrapolated"):
        assert estim2D(target2D_outofGrid) == 43


def test_ParametrizedComponentEstimator_checks():
    """Test checks for inputs perfomed by the base estimator class"""
    from pyirf.interpolation import (
        DiscretePDFExtrapolator,
        DiscretePDFInterpolator,
        ParametrizedExtrapolator,
        ParametrizedInterpolator,
    )
    from pyirf.interpolation.component_estimators import ParametrizedComponentEstimator

    class DummyInterpolator(ParametrizedInterpolator):
        def interpolate(self, target_point):
            return 42

    class DummyExtrapolator(ParametrizedExtrapolator):
        def extrapolate(self, target_point):
            return 43

    class WrongInterpolator(DiscretePDFInterpolator):
        def interpolate(self, target_point):
            return 41

    class WrongExtrapolator(DiscretePDFExtrapolator):
        def extrapolate(self, target_point):
            return 40

    grid_points = np.array([1, 2, 3])
    params_good = np.array([[1], [2], [3]])
    params_shape_missmatch = np.array([[1], [2]])

    with pytest.raises(
        TypeError,
        match="interpolator_cls must be a ParametrizedInterpolator subclass, got",
    ):
        ParametrizedComponentEstimator(
            grid_points=grid_points,
            params=params_good,
            interpolator_cls=WrongInterpolator,
        )

    with pytest.raises(TypeError, match="Input params is not a numpy array."):
        ParametrizedComponentEstimator(
            grid_points=grid_points,
            params=params_good.tolist(),
            interpolator_cls=DummyInterpolator,
        )

    with pytest.raises(
        ValueError,
        match="Shape missmatch, number of grid_points and rows in params not matching.",
    ):
        ParametrizedComponentEstimator(
            grid_points=grid_points,
            params=params_shape_missmatch,
            interpolator_cls=DummyInterpolator,
        )

    with pytest.raises(
        TypeError,
        match="extrapolator_cls must be a ParametrizedExtrapolator subclass, got",
    ):
        ParametrizedComponentEstimator(
            grid_points=grid_points,
            params=params_good,
            interpolator_cls=DummyInterpolator,
            extrapolator_cls=WrongExtrapolator,
        )

    estim = ParametrizedComponentEstimator(
        grid_points=grid_points,
        params=params_good,
        interpolator_cls=DummyInterpolator,
        extrapolator_cls=DummyExtrapolator,
    )
    assert estim(np.array([[1.5]])) == 42
    with pytest.warns(UserWarning, match="has to be extrapolated"):
        assert estim(np.array([[0]])) == 43


def test_DiscretePDFComponentEstimator_checks():
    """Test checks for inputs perfomed by the base estimator class"""
    from pyirf.interpolation import (
        DiscretePDFExtrapolator,
        DiscretePDFInterpolator,
        ParametrizedExtrapolator,
        ParametrizedInterpolator,
    )
    from pyirf.interpolation.component_estimators import DiscretePDFComponentEstimator

    class DummyInterpolator(DiscretePDFInterpolator):
        def interpolate(self, target_point):
            return 42

    class DummyExtrapolator(DiscretePDFExtrapolator):
        def extrapolate(self, target_point):
            return 43

    class WrongInterpolator(ParametrizedInterpolator):
        def interpolate(self, target_point):
            return 41

    class WrongExtrapolator(ParametrizedExtrapolator):
        def extrapolate(self, target_point):
            return 40

    grid_points = np.array([1, 2, 3])
    bin_edges = np.linspace(-1, 1, 11)
    binned_pdf_good_shape = np.ones(shape=(len(grid_points), len(bin_edges) - 1))
    binned_pdf_bad_nhist = np.ones(shape=(len(grid_points) - 1, len(bin_edges) - 1))
    binned_pdf_bad_nbins = np.ones(shape=(len(grid_points), len(bin_edges)))

    with pytest.raises(
        TypeError,
        match="interpolator_cls must be a DiscretePDFInterpolator subclass, got",
    ):
        DiscretePDFComponentEstimator(
            grid_points=grid_points,
            binned_pdf=binned_pdf_good_shape,
            bin_edges=bin_edges,
            interpolator_cls=WrongInterpolator,
        )

    with pytest.raises(TypeError, match="Input bin_edges is not a numpy array."):
        DiscretePDFComponentEstimator(
            grid_points=grid_points,
            binned_pdf=binned_pdf_good_shape,
            bin_edges=bin_edges.tolist(),
            interpolator_cls=DummyInterpolator,
        )

    with pytest.raises(TypeError, match="Input binned_pdf is not a numpy array."):
        DiscretePDFComponentEstimator(
            grid_points=grid_points,
            binned_pdf=binned_pdf_good_shape.tolist(),
            bin_edges=bin_edges,
            interpolator_cls=DummyInterpolator,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Shape missmatch, number of grid_points (3) and "
            "number of histograms in binned_pdf (2) not matching."
        ),
    ):
        DiscretePDFComponentEstimator(
            grid_points=grid_points,
            binned_pdf=binned_pdf_bad_nhist,
            bin_edges=bin_edges,
            interpolator_cls=DummyInterpolator,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Shape missmatch, bin_edges (10 bins) "
            "and binned_pdf (11 bins) not matching."
        ),
    ):
        DiscretePDFComponentEstimator(
            grid_points=grid_points,
            binned_pdf=binned_pdf_bad_nbins,
            bin_edges=bin_edges,
            interpolator_cls=DummyInterpolator,
        )

    with pytest.raises(
        TypeError,
        match="extrapolator_cls must be a DiscretePDFExtrapolator subclass, got",
    ):
        DiscretePDFComponentEstimator(
            grid_points=grid_points,
            bin_edges=bin_edges,
            binned_pdf=binned_pdf_good_shape,
            interpolator_cls=DummyInterpolator,
            extrapolator_cls=WrongExtrapolator,
        )

    estim = DiscretePDFComponentEstimator(
        grid_points=grid_points,
        bin_edges=bin_edges,
        binned_pdf=binned_pdf_good_shape,
        interpolator_cls=DummyInterpolator,
        extrapolator_cls=DummyExtrapolator,
    )
    assert estim(np.array([[1.5]])) == 42
    with pytest.warns(UserWarning, match="has to be extrapolated"):
        assert estim(np.array([[0]])) == 43


def test_DiscretePDFComponentEstimator_NearestNeighbors():
    """Test DiscretePDFComponentEstimator to be usable with NearestNeighborSearch."""
    from pyirf.interpolation.component_estimators import DiscretePDFComponentEstimator
    from pyirf.interpolation.nearest_neighbor_searcher import (
        DiscretePDFNearestNeighborSearcher,
        ParametrizedNearestNeighborSearcher,
    )

    grid_points = np.array([1, 2, 3])
    bin_edges = np.linspace(0, 1, 11)
    binned_pdf = np.array([np.full(10, x) for x in grid_points])

    estim = DiscretePDFComponentEstimator(
        grid_points=grid_points,
        binned_pdf=binned_pdf,
        bin_edges=bin_edges,
        interpolator_cls=DiscretePDFNearestNeighborSearcher,
        interpolator_kwargs={"norm_ord": 2},
        extrapolator_cls=DiscretePDFNearestNeighborSearcher,
        extrapolator_kwargs={"norm_ord": 1},
    )

    assert np.allclose(estim(target_point=np.array([1.1])), binned_pdf[0, :])
    with pytest.warns(UserWarning, match="has to be extrapolated"):
        assert np.allclose(estim(target_point=np.array([4.1])), binned_pdf[2, :])

    with pytest.raises(
        TypeError,
        match="interpolator_cls must be a DiscretePDFInterpolator subclass, got",
    ):
        DiscretePDFComponentEstimator(
            grid_points=grid_points,
            binned_pdf=binned_pdf,
            bin_edges=bin_edges,
            interpolator_cls=ParametrizedNearestNeighborSearcher,
            interpolator_kwargs={"norm_ord": 3},
            extrapolator_cls=DiscretePDFNearestNeighborSearcher,
            extrapolator_kwargs=None,
        )


def test_ParametrizedComponentEstimator_NearestNeighbors():
    """Test ParametrizedComponentEstimator to be usable with NearestNeighborSearch."""
    from pyirf.interpolation.component_estimators import ParametrizedComponentEstimator
    from pyirf.interpolation.nearest_neighbor_searcher import (
        DiscretePDFNearestNeighborSearcher,
        ParametrizedNearestNeighborSearcher,
    )

    grid_points = np.array([1, 2, 3])
    params = np.array([np.full(10, x) for x in grid_points])

    estim = ParametrizedComponentEstimator(
        grid_points=grid_points,
        params=params,
        interpolator_cls=ParametrizedNearestNeighborSearcher,
        interpolator_kwargs={"norm_ord": 2},
        extrapolator_cls=ParametrizedNearestNeighborSearcher,
        extrapolator_kwargs={"norm_ord": 1},
    )

    assert np.allclose(estim(target_point=np.array([1.1])), params[0, :])
    with pytest.warns(UserWarning, match="has to be extrapolated"):
        assert np.allclose(estim(target_point=np.array([4.1])), params[2, :])

    with pytest.raises(
        TypeError,
        match="interpolator_cls must be a ParametrizedInterpolator subclass, got",
    ):
        ParametrizedComponentEstimator(
            grid_points=grid_points,
            params=params,
            interpolator_cls=DiscretePDFNearestNeighborSearcher,
            interpolator_kwargs={"norm_ord": 3},
            extrapolator_cls=ParametrizedNearestNeighborSearcher,
            extrapolator_kwargs=None,
        )


def test_DiscretePDFComponentEstimator_1Dsorting():
    """Test DiscretePDFComponentEstimator sorts 1D-grid input in increasing order."""
    from pyirf.interpolation.base_extrapolators import DiscretePDFExtrapolator
    from pyirf.interpolation.base_interpolators import DiscretePDFInterpolator
    from pyirf.interpolation.component_estimators import DiscretePDFComponentEstimator

    grid_points = np.array([[3], [1], [2]])
    bin_edges = np.linspace(0, 1, 11)
    binned_pdf = np.array([np.full(10, x) for x in grid_points])

    class DummyInterpolator(DiscretePDFInterpolator):
        def interpolate(self, target_point):
            return 42

    class DummyExtrapolator(DiscretePDFExtrapolator):
        def extrapolate(self, target_point):
            if target_point < self.grid_points.min():
                return self.binned_pdf[0]
            elif target_point > self.grid_points.max():
                return self.binned_pdf[-1]

    estim = DiscretePDFComponentEstimator(
        grid_points=grid_points,
        binned_pdf=binned_pdf,
        bin_edges=bin_edges,
        interpolator_cls=DummyInterpolator,
        extrapolator_cls=DummyExtrapolator,
    )

    # Nearest neighbor is grid_point 1 at the index 1 of the original binned_pdf
    with pytest.warns(UserWarning, match="has to be extrapolated"):
        assert np.allclose(estim(target_point=np.array([0])), binned_pdf[1, :])
    # Nearest neighbor is grid_point 3 at the index 0 of the original binned_pdf
    with pytest.warns(UserWarning, match="has to be extrapolated"):
        assert np.allclose(estim(target_point=np.array([4])), binned_pdf[0, :])


def test_ParametrizedComponentEstimator_1Dsorting():
    """Test ParametrizedComponentEstimator sorts 1D-grid input in increasing order."""
    from pyirf.interpolation.base_extrapolators import ParametrizedExtrapolator
    from pyirf.interpolation.base_interpolators import ParametrizedInterpolator
    from pyirf.interpolation.component_estimators import ParametrizedComponentEstimator

    grid_points = np.array([[3], [1], [2]])
    params = np.array([np.full(10, x) for x in grid_points])

    class DummyInterpolator(ParametrizedInterpolator):
        def interpolate(self, target_point):
            return 42

    class DummyExtrapolator(ParametrizedExtrapolator):
        def extrapolate(self, target_point):
            if target_point < self.grid_points.min():
                return self.params[0]
            elif target_point > self.grid_points.max():
                return self.params[-1]

    estim = ParametrizedComponentEstimator(
        grid_points=grid_points,
        params=params,
        interpolator_cls=DummyInterpolator,
        extrapolator_cls=DummyExtrapolator,
    )

    # Nearest neighbor is grid_point 1 at the index 1 of the original binned_pdf
    with pytest.warns(UserWarning, match="has to be extrapolated"):
        assert np.allclose(estim(target_point=np.array([0])), params[1, :])
    # Nearest neighbor is grid_point 3 at the index 0 of the original binned_pdf
    with pytest.warns(UserWarning, match="has to be extrapolated"):
        assert np.allclose(estim(target_point=np.array([4])), params[0, :])
