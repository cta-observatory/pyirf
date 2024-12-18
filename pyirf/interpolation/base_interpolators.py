"""Base classes for interpolators"""
from abc import ABCMeta, abstractmethod
import enum

import numpy as np

from ..binning import bin_center

__all__ = [
    "BaseInterpolator",
    "ParametrizedInterpolator",
    "DiscretePDFInterpolator",
    "PDFNormalization",
]


class PDFNormalization(enum.Enum):
    """How a discrete PDF is normalized"""

    #: PDF is normalized to a "normal" area integral of 1
    AREA = enum.auto()
    #: PDF is normalized to 1 over the solid angle integral where the bin
    #: edges represent the opening angles of cones in radian.
    CONE_SOLID_ANGLE = enum.auto()


class BaseInterpolator(metaclass=ABCMeta):
    """
    Base class for all interpolators, only knowing grid-points,
    providing a common __call__-interface.
    """

    def __init__(self, grid_points):
        """BaseInterpolator

        Parameters
        ----------
        grid_points: np.ndarray, shape=(n_points, n_dims):
            Grid points at which interpolation templates exist

        """
        self.grid_points = grid_points
        if self.grid_points.ndim == 1:
            self.grid_points = self.grid_points.reshape(*self.grid_points.shape, 1)
        self.n_points = self.grid_points.shape[0]
        self.grid_dim = self.grid_points.shape[1]

    @abstractmethod
    def interpolate(self, target_point):
        """Overridable function for the actual interpolation code"""

    def __call__(self, target_point):
        """Providing a common __call__ interface

        Parameters
        ----------
        target_point: np.ndarray, shape=(1, n_dims)
            Target for inter-/extrapolation
            When target_point is outside of the grids convex hull but extrapolator is None

        Returns
        -------
        Interpolated result.
        """
        return self.interpolate(target_point=target_point)


class ParametrizedInterpolator(BaseInterpolator):
    """
    Base class for all interpolators used with IRF components that can be
    independently interpolated, e.g. parametrized ones like 3Gauss
    but also AEff. Derived from pyirf.interpolation.BaseInterpolator
    """

    def __init__(self, grid_points, params):
        """ParametrizedInterpolator

        Parameters
        ----------
        grid_points, np.ndarray, shape=(n_points, n_dims)
            Grid points at which interpolation templates exist
        params: np.ndarray, shape=(n_points, ..., n_params)
            Corresponding parameter values at each point in grid_points.
            First dimesion has to correspond to number of grid_points

        Note
        ----
            Also calls pyirf.interpolation.BaseInterpolators.__init__
        """
        super().__init__(grid_points)

        self.params = params

        if self.params.ndim == 1:
            self.params = self.params[..., np.newaxis]


class DiscretePDFInterpolator(BaseInterpolator):
    """
    Base class for all interpolators used with binned IRF components like EDisp.
    Derived from pyirf.interpolation.BaseInterpolator
    """

    def __init__(
        self, grid_points, bin_edges, binned_pdf, normalization=PDFNormalization.AREA
    ):
        """DiscretePDFInterpolator

        Parameters
        ----------
        grid_points : np.ndarray, shape=(n_points, n_dims)
            Grid points at which interpolation templates exist
        bin_edges : np.ndarray, shape=(n_bins+1)
            Edges of the data binning
        binned_pdf : np.ndarray, shape=(n_points, ..., n_bins)
            Content of each bin in bin_edges for
            each point in grid_points. First dimesion has to correspond to number
            of grid_points, last dimension has to correspond to number of bins for
            the quantity that should be interpolated (e.g. the Migra axis for EDisp)
        normalization : PDFNormalization
            How the PDF is normalized

        Note
        ----
            Also calls pyirf.interpolation.BaseInterpolators.__init__
        """
        super().__init__(grid_points)

        self.bin_edges = bin_edges
        self.bin_mids = bin_center(self.bin_edges)
        self.binned_pdf = binned_pdf
        self.normalization = normalization
