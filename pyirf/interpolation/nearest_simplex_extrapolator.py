"""
Extrapolators for Parametrized and DiscretePDF components extrapolating from the 
nearest simplex
"""
import warnings

import numpy as np
from scipy.spatial import Delaunay


from .base_extrapolators import DiscretePDFExtrapolator, ParametrizedExtrapolator, PDFNormalization
from .moment_morph_interpolator import (
    barycentric_2D_interpolation_coefficients,
    linesegment_1D_interpolation_coefficients,
    moment_morph_estimation,
)
from .utils import find_nearest_simplex, get_bin_width

__all__ = [
    "MomentMorphNearestSimplexExtrapolator",
    "ParametrizedNearestSimplexExtrapolator",
]


class ParametrizedNearestSimplexExtrapolator(ParametrizedExtrapolator):
    def __init__(self, grid_points, params):
        """
        Extrapolator class using linear extrapolation in one ore two
        grid-dimensions.

        Parameters
        ----------
        grid_points: np.ndarray, shape=(N, ...)
            Grid points at which templates exist. May be one ot two dimensional.
            Have to be sorted in accending order for 1D.
        params: np.ndarray, shape=(N, ...)
            Array of corresponding parameter values at each point in grid_points.
            First dimesion has to correspond to number of grid_points

        Note
        ----
            Also calls pyirf.interpolation.ParametrizedInterpolator.__init__.
        """
        super().__init__(grid_points, params)

        if self.grid_dim == 2:
            self.triangulation = Delaunay(self.grid_points)
        elif self.grid_dim > 2:
            raise NotImplementedError(
                "Extrapolation in more then two dimension not impemented."
            )

    def _extrapolate1D(self, segment_inds, target_point):
        """
        Function to compute extrapolation coefficients for a target_point on a
        specified grid segment and extrapolate from this subset
        """
        coefficients = linesegment_1D_interpolation_coefficients(
            grid_points=self.grid_points[segment_inds],
            target_point=target_point.squeeze(),
        )

        # reshape to be broadcastable
        coefficients = coefficients.reshape(
            coefficients.shape[0], *np.ones(self.params.ndim - 1, "int")
        )

        return np.sum(coefficients * self.params[segment_inds, :], axis=0)[
            np.newaxis, :
        ]

    def _extrapolate2D(self, simplex_inds, target_point):
        """
        Function to compute extrapolation coeficcients and extrapolate from the nearest
        simplex by extending barycentric interpolation
        """
        coefficients = barycentric_2D_interpolation_coefficients(
            grid_points=self.grid_points[simplex_inds],
            target_point=target_point,
        )

        # reshape to be broadcastable
        coefficients = coefficients.reshape(
            coefficients.shape[0], *np.ones(self.params.ndim - 1, "int")
        )

        return np.sum(coefficients * self.params[simplex_inds, :], axis=0)[
            np.newaxis, :
        ]

    def extrapolate(self, target_point):
        """
        Takes a grid of scalar values for a bunch of different parameters
        and extrapolates it to given value of those parameters.

        Parameters
        ----------
        target_point: numpy.ndarray
            Value for which the extrapolation is performed (target point)

        Returns
        -------
        values: numpy.ndarray, shape=(1,...)
            Extrapolated values

        """
        if self.grid_dim == 1:
            if target_point < self.grid_points.min():
                segment_inds = np.array([0, 1], "int")
            else:
                segment_inds = np.array([-2, -1], "int")

            extrapolant = self._extrapolate1D(segment_inds, target_point)
        elif self.grid_dim == 2:
            nearest_simplex_ind = find_nearest_simplex(
                self.triangulation, target_point.squeeze()
            )
            simplex_indices = self.triangulation.simplices[nearest_simplex_ind]

            extrapolant = self._extrapolate2D(simplex_indices, target_point)

        return extrapolant


class MomentMorphNearestSimplexExtrapolator(DiscretePDFExtrapolator):
    def __init__(self, grid_points, bin_edges, binned_pdf, normalization=PDFNormalization.AREA):
        """
        Extrapolator class extending/reusing parts of Moment Morphing
        by allowing for negative extrapolation coefficients computed
        by from the nearest simplex in 1D or 2D (so either a line-segement
        or a triangle).

        Parameters
        ----------
        grid_points: np.ndarray, shape=(N, ...)
            Grid points at which templates exist. May be one ot two dimensional.
            Have to be sorted in accending order for 1D.
        bin_edges: np.ndarray, shape=(M+1)
            Edges of the data binning
        binned_pdf: np.ndarray, shape=(N, ..., M)
            Content of each bin in bin_edges for
            each point in grid_points. First dimesion has to correspond to number
            of grid_points. Extrapolation dimension, meaning the
            the quantity that should be interpolated (e.g. the Migra axis for EDisp)
            has to be at the last axis as well as having entries
            corresponding to the number of bins given through bin_edges keyword.
        normalization : PDFNormalization
            How the PDF is normalized

        Note
        ----
            Also calls pyirf.interpolation.DiscretePDFExtrapolator.__init__.
        """
        super().__init__(grid_points, bin_edges, binned_pdf, normalization)

        if self.grid_dim == 2:
            self.triangulation = Delaunay(self.grid_points)
        elif self.grid_dim > 2:
            raise NotImplementedError(
                "Extrapolation in more then two dimension not impemented."
            )

    def _extrapolate1D(self, segment_inds, target_point):
        """
        Function to compute extrapolation coefficients for a target_point on a
        specified grid segment and extrapolate from this subset
        """
        coefficients = linesegment_1D_interpolation_coefficients(
            grid_points=self.grid_points[segment_inds],
            target_point=target_point.squeeze(),
        )

        return moment_morph_estimation(
            bin_edges=self.bin_edges,
            binned_pdf=self.binned_pdf[segment_inds],
            coefficients=coefficients,
            normalization=self.normalization,
        )

    def _extrapolate2D(self, simplex_inds, target_point):
        """
        Function to compute extrapolation coeficcients and extrapolate from the nearest
        simplex by extending barycentric interpolation
        """
        coefficients = barycentric_2D_interpolation_coefficients(
            grid_points=self.grid_points[simplex_inds],
            target_point=target_point,
        )

        return moment_morph_estimation(
            bin_edges=self.bin_edges,
            binned_pdf=self.binned_pdf[simplex_inds],
            coefficients=coefficients,
            normalization=self.normalization,
        )

    def extrapolate(self, target_point):
        """
        Takes a grid of discretized PDFs for a bunch of different parameters
        and extrapolates it to given value of those parameters.

        Parameters
        ----------
        target_point: numpy.ndarray
            Value for which the extrapolation is performed (target point)

        Returns
        -------
        values: numpy.ndarray, shape=(1, ..., M)
            Extrapolated discretized PDFs

        """
        if self.grid_dim == 1:
            if target_point < self.grid_points.min():
                segment_inds = np.array([0, 1], "int")
            else:
                segment_inds = np.array([-2, -1], "int")

            extrapolant = self._extrapolate1D(segment_inds, target_point)
        elif self.grid_dim == 2:
            nearest_simplex_ind = find_nearest_simplex(
                self.triangulation, target_point.squeeze()
            )
            simplex_indices = self.triangulation.simplices[nearest_simplex_ind]

            extrapolant = self._extrapolate2D(simplex_indices, target_point)

        if not np.all(extrapolant >= 0):
            warnings.warn(
                "Some bin-entries where below zero after extrapolation and "
                "thus cut off. Check result carefully."
            )

            # cut off values below 0 and re-normalize
            extrapolant[extrapolant < 0] = 0
            bin_width = get_bin_width(self.bin_edges, self.normalization)
            norm = np.expand_dims(np.sum(extrapolant * bin_width, axis=-1), -1)
            return np.divide(
                extrapolant, norm, out=np.zeros_like(extrapolant), where=norm != 0
            ).reshape(1, *self.binned_pdf.shape[1:])
        else:
            return extrapolant
