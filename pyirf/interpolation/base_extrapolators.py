"""Base classes for extrapolators"""
from abc import ABCMeta, abstractmethod

import numpy as np
from pyirf.binning import bin_center
from pyirf.interpolation.base_interpolators import PDFNormalization

__all__ = ["BaseExtrapolator", "ParametrizedExtrapolator", "DiscretePDFExtrapolator"]


class BaseExtrapolator(metaclass=ABCMeta):
    """
    Base class for all extrapolators, only knowing grid-points,
    providing a common __call__-interface.
    """

    def __init__(self, grid_points):
        """BaseExtrapolator

        Parameters
        ----------
        grid_points: np.ndarray, shape=(n_points, n_dims):
            Grid points at which templates exist

        """
        self.grid_points = grid_points
        if self.grid_points.ndim == 1:
            self.grid_points = self.grid_points.reshape(*self.grid_points.shape, 1)
        self.n_points = self.grid_points.shape[0]
        self.grid_dim = self.grid_points.shape[1]

    @abstractmethod
    def extrapolate(self, target_point):
        """Overridable function for the actual extrapolation code"""

    def __call__(self, target_point):
        """Providing a common __call__ interface

        Parameters
        ----------
        target_point: np.ndarray, shape=(1, n_dims)
            Target for extrapolation

        Returns
        -------
        Extrapolated result.
        """
        return self.extrapolate(target_point=target_point)


class ParametrizedExtrapolator(BaseExtrapolator):
    """
    Base class for all extrapolators used with IRF components that can be
    treated independently, e.g. parametrized ones like 3Gauss
    but also AEff. Derived from pyirf.interpolation.BaseExtrapolator
    """

    def __init__(self, grid_points, params):
        """ParametrizedExtrapolator

        Parameters
        ----------
        grid_points, np.ndarray, shape=(n_points, n_dims)
            Grid points at which templates exist
        params: np.ndarray, shape=(n_points, ..., n_params)
            Corresponding parameter values at each point in grid_points.
            First dimesion has to correspond to number of grid_points

        Note
        ----
            Also calls pyirf.interpolation.BaseExtrapolators.__init__
        """
        super().__init__(grid_points)

        self.params = params

        if self.params.ndim == 1:
            self.params = self.params[..., np.newaxis]


class DiscretePDFExtrapolator(BaseExtrapolator):
    """
    Base class for all extrapolators used with binned IRF components like EDisp.
    Derived from pyirf.interpolation.BaseExtrapolator
    """

    def __init__(self, grid_points, bin_edges, binned_pdf, normalization=PDFNormalization.AREA):
        """DiscretePDFExtrapolator

        Parameters
        ----------
        grid_points : np.ndarray, shape=(n_points, n_dims)
            Grid points at which templates exist
        bin_edges : np.ndarray, shape=(n_bins+1)
            Edges of the data binning
        binned_pdf : np.ndarray, shape=(n_points, ..., n_bins)
            Content of each bin in bin_edges for
            each point in grid_points. First dimesion has to correspond to number
            of grid_points, last dimension has to correspond to number of bins for
            the quantity that should be extrapolated (e.g. the Migra axis for EDisp)
        normalization : PDFNormalization
            How the PDF is normalized

        Note
        ----
            Also calls pyirf.interpolation.BaseExtrapolators.__init__
        """
        super().__init__(grid_points)

        self.normalization = normalization
        self.bin_edges = bin_edges
        self.bin_mids = bin_center(self.bin_edges)
        self.binned_pdf = binned_pdf
