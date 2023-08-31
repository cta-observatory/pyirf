"""
Extrapolators for Parametrized and DiscretePDF components blending all extrapolating from 
all visible simplices
"""
import numpy as np
from scipy.spatial import Delaunay

from .base_extrapolators import PDFNormalization
from .nearest_simplex_extrapolator import ParametrizedNearestSimplexExtrapolator, MomentMorphNearestSimplexExtrapolator
from .utils import find_simplex_to_facet, point_facet_angle

__all__ = ["ParametrizedVisibleEdgesExtrapolator", "MomentMorphVisibleEdgesExtrapolator"]


def find_visible_facets(grid_points, target_point):
    """
    Utility function to find all facets of a convex hull that are visible from
    a point outside. To do so, this function constructs a trinangulation containing
    the target point and returns all facets that span a trinangulation simplex with
    the it.

    Parameters
    ----------
    grid_points: np.ndarray, shape=(N, M)
        Grid points at which templates exist. May be one ot two dimensional.
        Have to be sorted in accending order for 1D.
    target_point: numpy.ndarray, shape=(1, M)
        Value for which the extrapolation is performed (target point)

    Returns
    -------
    visible_facets: np.ndarray, shape=(L, M)
        L visible facets, spanned by a simplex in M-1 dimensions
        (thus a line for M=2)
    """
    # Build a triangulation including the target point
    full_set = np.vstack((grid_points, target_point))
    triag = Delaunay(full_set)

    # The target point is included in a simplex with all facets
    # visible by it
    simplices = triag.points[triag.simplices]
    matches_target = np.all(simplices == target_point, axis=-1)
    target_in_simplex = np.any(matches_target, axis=-1)

    # The visible facets are spanned by those points in the matched
    # simplices that are not the target
    facet_point_mask = ~matches_target[target_in_simplex]
    visible_facets = np.array(
        [
            triag.points[simplex[mask]]
            for simplex, mask in zip(
                triag.simplices[target_in_simplex], facet_point_mask
            )
        ]
    )

    return visible_facets


def compute_extrapolation_weights(visible_facet_points, target_point, m):
    """
    Utility function to compute extrapolation weight according to [1].

    Parameters
    ----------
    visible_facet_points: np.ndarray, shape=(L, M)
        L facets visible from target_point
    target_point: numpy.ndarray, shape=(1, M)
        Value for which the extrapolation is performed (target point)

    Returns
    -------
    extrapolation_weights: np.ndarray, shape=(L)
        Weights for each visible facet, corresponding to the extrapolation
        weight for the respective triangulation simplex. Weigths sum to unity.

    References
    ----------
    .. [1] P. Alfred (1984). Triangular Extrapolation. Technical summary rept.,
    Univ. of Wisconsin-Madison. https://apps.dtic.mil/sti/pdfs/ADA144660.pdf
    """

    angles = np.array(
        [
            point_facet_angle(line, target_point.squeeze())
            for line in visible_facet_points
        ]
    )
    weights = np.arccos(angles) ** (m + 1)

    return weights / weights.sum()


class ParametrizedVisibleEdgesExtrapolator(ParametrizedNearestSimplexExtrapolator):
    def __init__(self, grid_points, params, m=1):
        """
        Extrapolator class blending extrapolations from all visible simplices
        in two grid-dimensions as discussed in [1]. For one grid-dimension this
        is equal to the ParametrizedNearestSimplexExtrapolator, the same holds
        for grids consisting of only one simplex or constellations,
        where only one simplex is visible from a target.

        Parameters
        ----------
        grid_points: np.ndarray, shape=(N, ...)
            Grid points at which templates exist. May be one ot two dimensional.
            Have to be sorted in accending order for 1D.
        params: np.ndarray, shape=(N, ...)
            Array of corresponding parameter values at each point in grid_points.
            First dimesion has to correspond to number of grid_points
        m: non-zero int >= 1
            Degree of smoothness wanted in the extrapolation region. See [1] for
            additional information. Defaults to 1.

        Raises
        ------
        ValueError:
            If m is not a non-zero integer

        Note
        ----
            Also calls pyirf.interpolation.ParametrizedNearestSimplexExtrapolator.__init__.

        References
        ----------
        .. [1] P. Alfred (1984). Triangular Extrapolation. Technical summary rept.,
        Univ. of Wisconsin-Madison. https://apps.dtic.mil/sti/pdfs/ADA144660.pdf
        """

        super().__init__(grid_points, params)

        # Test wether m is a number
        try:
            m > 0
        except TypeError:
            raise ValueError(f"Only positiv integers allowed for m, got {m}.")

        # Test wether m is a finite, positive integer
        if (m <= 0) or ~np.isfinite(m) or (m != int(m)):
            raise ValueError(f"Only positiv integers allowed for m, got {m}.")

        self.m = m

    def extrapolate(self, target_point):
        """
        Takes a grid of scalar values for a bunch of different parameters
        and extrapolates it to given value of those parameters. Falls back
        to ParametrizedNearestSimplexExtrapolator's behavior for 1D grids or
        cases, where only one simplex is visible in 2D.

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
            extrapolant = super().extrapolate(target_point)
        elif self.grid_dim == 2:
            visible_facet_points = find_visible_facets(self.grid_points, target_point)

            if visible_facet_points.shape[0] == 1:
                extrapolant = super().extrapolate(target_point)
            else:
                simplices_points = self.triangulation.points[
                    self.triangulation.simplices
                ]

                visible_simplices_indices = np.array(
                    [
                        find_simplex_to_facet(simplices_points, facet)
                        for facet in visible_facet_points
                    ]
                )

                extrapolation_weigths = compute_extrapolation_weights(
                    visible_facet_points, target_point, self.m
                )

                extrapolation_weigths = extrapolation_weigths.reshape(
                    extrapolation_weigths.shape[0],
                    *np.ones(self.params.ndim - 1, "int"),
                )

                # Function has to be copied outside list comprehention as the super() short-form
                # cannot be used inside it (at least until python 3.11)
                extrapolate2D = super()._extrapolate2D

                simplex_extrapolations = np.array(
                    [
                        extrapolate2D(
                            self.triangulation.simplices[ind], target_point
                        ).squeeze()
                        for ind in visible_simplices_indices
                    ]
                )

                extrapolant = np.sum(
                    extrapolation_weigths * simplex_extrapolations, axis=0
                )[np.newaxis, :]

        return extrapolant


class MomentMorphVisibleEdgesExtrapolator(MomentMorphNearestSimplexExtrapolator):
    def __init__(self, grid_points, bin_edges, binned_pdf, m=1, normalization=PDFNormalization.AREA):
        """
        Extrapolator class blending extrapolations from all visible simplices
        in two grid-dimensions as discussed in [1]. For one grid-dimension this
        is equal to the MomentMorphNearestSimplexExtrapolator, the same holds
        for grids consisting of only one simplex or constellations,
        where only one simplex is visible from a target.

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
        m: non-zero int >= 1
            Degree of smoothness wanted in the extrapolation region. See [1] for
            additional information. Defaults to 1.
        normalization : PDFNormalization
            How the PDF is normalized, defaults to PDFNormalization.AREA

        Raises
        ------
        ValueError:
            If m is not a non-zero integer

        Note
        ----
            Also calls pyirf.interpolation.MomentMorphNearestSimplexExtrapolator.__init__.

        References
        ----------
        .. [1] P. Alfred (1984). Triangular Extrapolation. Technical summary rept.,
        Univ. of Wisconsin-Madison. https://apps.dtic.mil/sti/pdfs/ADA144660.pdf
        """

        super().__init__(grid_points, bin_edges, binned_pdf, normalization)

        # Test wether m is a number
        try:
            m > 0
        except TypeError:
            raise ValueError(f"Only positiv integers allowed for m, got {m}.")

        # Test wether m is a finite, positive integer
        if (m <= 0) or ~np.isfinite(m) or (m != int(m)):
            raise ValueError(f"Only positiv integers allowed for m, got {m}.")

        self.m = m

    def extrapolate(self, target_point):
        """
        Takes a grid of scalar values for a bunch of different parameters
        and extrapolates it to given value of those parameters. Falls back
        to ParametrizedNearestSimplexExtrapolator's behavior for 1D grids or
        cases, where only one simplex is visible in 2D.

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
            extrapolant = super().extrapolate(target_point)
        elif self.grid_dim == 2:
            visible_facet_points = find_visible_facets(self.grid_points, target_point)

            if visible_facet_points.shape[0] == 1:
                extrapolant = super().extrapolate(target_point)
            else:
                simplices_points = self.triangulation.points[
                    self.triangulation.simplices
                ]

                visible_simplices_indices = np.array(
                    [
                        find_simplex_to_facet(simplices_points, facet)
                        for facet in visible_facet_points
                    ]
                )

                extrapolation_weigths = compute_extrapolation_weights(
                    visible_facet_points, target_point, self.m
                )

                extrapolation_weigths = extrapolation_weigths.reshape(
                    extrapolation_weigths.shape[0],
                    *np.ones(self.binned_pdf.ndim - 1, "int"),
                )

                # Function has to be copied outside list comprehention as the super() short-form
                # cannot be used inside it (at least until python 3.11)
                extrapolate2D = super()._extrapolate2D

                simplex_extrapolations = np.array(
                    [
                        extrapolate2D(
                            self.triangulation.simplices[ind], target_point
                        ).squeeze()
                        for ind in visible_simplices_indices
                    ]
                )

                extrapolant = np.sum(
                    extrapolation_weigths * simplex_extrapolations, axis=0
                )[np.newaxis, :]

        return extrapolant
