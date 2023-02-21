"""Base classes for interpolators"""
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.spatial import Delaunay

from pyirf.binning import bin_center

__all__ = ["BaseInterpolator", "ParametrizedInterpolator", "BinnedInterpolator"]


class BaseInterpolator(metaclass=ABCMeta):
    """
    Base class for all interpolators, only knowing grid-points,
    providing a common __call__-interface and doing sanity checks.
    """

    def __init__(self, grid_points):
        """BaseInterpolator constructor, doing sanity checks and ensuring
        correct shapes of inputs

        Parameters
        ----------
        grid_points: np.ndarray, shape=(n_points, n_dims):
            Grid points at which interpolation templates exist

        Raises
        ------
        TypeError:
            When grid_points is not a np.ndarray
        TypeError:
            When grid_point has dtype object
        ValueError:
            When there are too few points in grid_points to span a volume
            in the grid dimension.
        """
        if not isinstance(grid_points, np.ndarray):
            raise TypeError("Input grid_points is not a numpy array.")
        if grid_points.dtype == "O":
            raise TypeError("Input grid_points array cannot be of dtype object.")
        if not np.can_cast(grid_points.dtype, np.float128):
            raise TypeError("Input grid_points dtype incompatible with float.")

        self.grid_points = grid_points
        if self.grid_points.ndim == 1:
            self.grid_points = self.grid_points.reshape(*self.grid_points.shape, 1)
        self.n_points = self.grid_points.shape[0]
        self.grid_dim = self.grid_points.shape[1]

        # Check, if number of grid point theoretically suffices to span a volume
        # in the dimension indicated by grid
        if self.n_points < self.grid_dim + 1:
            raise ValueError(
                f"To few points for grid dimension, grid-dim is {self.grid_dim},"
                f" while there are only {self.n_points}. At least {self.grid_dim+1}"
                f" points needed to span a volume in {self.grid_dim} dimensions."
            )

        # Build triangulation to check if target is inside of the grid for
        # more then 1 dimension
        if self.grid_dim > 1:
            self.triangulation = Delaunay(self.grid_points)

    @abstractmethod
    def interpolate(self, target_point, **kwargs):
        """Overridable function for the actual interpolation code"""

    def _target_in_grid(self, target_point):
        """Check wether target_point lies within grids convex hull"""
        if self.grid_dim == 1:
            return (target_point >= self.grid_points.min()) and (
                target_point <= self.grid_points.max()
            )
        else:
            # Delaunay.find_simplex() returns -1 for points outside the grids convex hull
            simplex_ind = self.triangulation.find_simplex(target_point)
            return simplex_ind >= 0

    def __call__(self, target_point, extrapolator=None, **kwargs):
        """Providing a common __call__ interface sanity checking the target point

        Parameters
        ----------
        target_point: np.ndarray
            Target for inter-/extrapolation
        extrapolator: callable, optional
            Fall-Back extrapolator called when target_point is outside grid.
            Defaults to None, meaning no extrapolation is done.

        Raises
        ------
        TypeError:
            When target_point is not an np.ndarray
        ValueError:
            When more then one target_point is given
        ValueError:
            When target_point and grid_points have miss-matching dimensions
        ValueError:
            When target_point is outside of the grids convex hull but extrapolator is None

        Returns
        -------
        Interpolated or, if necessary extrapolated, result.
        """
        if not isinstance(target_point, np.ndarray):
            raise TypeError("Target point is not a numpy array.")

        if target_point.ndim == 1:
            target_point = target_point.reshape(1, *target_point.shape)
        elif target_point.shape[0] != 1:
            raise ValueError("Only one target_point per call supported.")

        if target_point.shape[1] != self.grid_dim:
            raise ValueError(
                "Missmatch between target-point and grid dimension."
                f" Grid has dimension {self.grid_dim}, target has dimension"
                f" {target_point.shape[1]}."
            )

        if self._target_in_grid(target_point):
            return self.interpolate(target_point, **kwargs)
        elif extrapolator is not None:
            print(f"Trying to extrapolate for point {target_point}.")
            return extrapolator(target_point, **kwargs)
        else:
            raise ValueError(
                "Target point outside grids convex hull and no extrapolator given."
            )


class ParametrizedInterpolator(BaseInterpolator):
    """
    Base class for all interpolators used with IRF components that can be 
    independently interpolated, e.g. parametrized ones like 3Gauss
    but also AEff, extending BaseInterpolators sanity checks.
    Derived from pyirf.interpolation.BaseInterpolator
    """

    def __init__(self, grid_points, params):
        """ParametrizedInterpolator constructor

        Parameters
        ----------
        grid_points, np.ndarray
            Grid points at which interpolation templates exist
        params: np.ndarray
            Corresponding parameter values at each point in grid_points.
            First dimesion has to correspond to number of grid_points

        Raises
        ------
        TypeError:
            When params is not a np.ndarray
        ValueError:
            When number of points grid_points and params is not matching

        Note
        ----
            Also calls pyirf.interpolation.BaseInterpolators.__call__
        """
        super().__init__(grid_points)

        if not isinstance(params, np.ndarray):
            raise TypeError("Input params is not a numpy array.")
        elif self.n_points != params.shape[0]:
            raise ValueError(
                "Shape missmatch, number of grid_points and rows in params not matching."
            )
        else:
            self.params = params

        if self.params.ndim == 1:
            self.params = self.params[..., np.newaxis]


class BinnedInterpolator(BaseInterpolator):
    """
    Base class for all interpolators used with binned IRF components like EDisp,
    extending BaseInterpolators sanity checks.
    Derived from pyirf.interpolation.BaseInterpolator
    """

    def __init__(self, grid_points, bin_edges, bin_contents):
        """BinnedInterpolator constructor

        Parameters
        ----------
        grid_points: np.ndarray
            Grid points at which interpolation templates exist
        bin_edges: np.ndarray
            Edges of the data binning
        bin_content: np.ndarray
            Content of each bin in bin_edges for
            each point in grid_points. First dimesion has to correspond to number
            of grid_points, last dimension has to correspond to number of bins for
            the quantity that should be interpolated (e.g. the Migra axis for EDisp)

        Raises
        ------
        TypeError:
            When bin_edges is not a np.ndarray
        TypeError:
            When bin_content is not a np.ndarray
        ValueError:
            When number of bins in bin_edges and contents bin_contents is
            not matching
        ValueError:
            When number of histograms in bin_contents and points in grid_points
            is not matching

        Note
        ----
            Also calls pyirf.interpolation.BaseInterpolators.__call__
        """
        super().__init__(grid_points)

        if not isinstance(bin_edges, np.ndarray):
            raise TypeError("Input bin_edges is not a numpy array.")
        elif not isinstance(bin_contents, np.ndarray):
            raise TypeError("Input bin_contents is not a numpy array.")
        elif bin_contents.shape[-1] != (bin_edges.shape[0] - 1):
            raise ValueError(
                f"Shape missmatch, bin_edges ({bin_edges.shape[0] - 1} bins) "
                f"and bin_contents ({bin_contents.shape[-1]} bins) not matching."
            )
        elif self.n_points != bin_contents.shape[0]:
            raise ValueError(
                f"Shape missmatch, number of grid_points ({self.n_points}) and "
                f"number of histograms in bin_contents ({bin_contents.shape[0]}) "
                "not matching."
            )
        else:
            self.bin_edges = bin_edges
            self.bin_mids = bin_center(self.bin_edges)
            self.bin_contents = bin_contents
