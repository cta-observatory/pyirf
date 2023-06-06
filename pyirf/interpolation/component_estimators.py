"""Classes to estimate (interpolate/extrapolate) actual IRF HDUs"""
import warnings

import astropy.units as u
import numpy as np
from pyirf.interpolation.base_interpolators import (
    DiscretePDFInterpolator,
    ParametrizedInterpolator,
)
from pyirf.interpolation.griddata_interpolator import GridDataInterpolator
from pyirf.interpolation.quantile_interpolator import QuantileInterpolator
from scipy.spatial import Delaunay

__all__ = [
    "BaseComponentEstimator",
    "DiscretePDFComponentEstimator",
    "ParametrizedComponentEstimator",
    "AEFFEstimator",
    "RAD_MAXEstimator",
]


class BaseComponentEstimator:
    """
    Base class for all Estimators working on specific IRF Components. While
    usable, it is encuraged to use the actual class for the respective IRF
    Component as it ensures further checks and if nessecarry e.g. unit handling.
    """

    def __init__(self, grid_points):
        """
        Base __init__, doing sanity checks on the grid, building a
        triangulated version of the grid and intantiating inter- and extrapolator.

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

    def _target_in_grid(self, target_point):
        """Check wether target_point lies within grids convex hull, uses
        simple comparison for 1D and Delaunay triangulation for >1D."""
        if self.grid_dim == 1:
            return (target_point >= self.grid_points.min()) and (
                target_point <= self.grid_points.max()
            )
        else:
            # Delaunay.find_simplex() returns -1 for points outside the grids convex hull
            simplex_ind = self.triangulation.find_simplex(target_point)
            return simplex_ind >= 0

    def __call__(self, target_point):
        """Inter-/ Extrapolation as needed and sanity checking of
        the target point

        Parameters
        ----------
        target_point: np.ndarray, shape=(1, n_dims)
            Target for inter-/extrapolation

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
        Warning:
            When target_points need extrapolation

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
            return self.interpolator(target_point)
        elif self.extrapolator is not None:
            warnings.warn(f"Target point {target_point} has to be extrapolated.")
            return self.extrapolator(target_point)
        else:
            raise ValueError(
                "Target point outside grids convex hull and no extrapolator given."
            )


class DiscretePDFComponentEstimator(BaseComponentEstimator):
    def __init__(
        self,
        grid_points,
        bin_edges,
        bin_contents,
        interpolator_cls=QuantileInterpolator,
        interpolator_kwargs=None,
        extrapolator_cls=None,
        extrapolator_kwargs=None,
    ):
        """
        __init__ for all discrete PDF components, calls BaseComponentEstimator's
        __init__ and instantiates inter- and extrapolator objects.

        Parameters
        ----------
        grid_points: np.ndarray, shape=(n_points, n_dims):
            Grid points at which interpolation templates exist
        bin_edges: np.ndarray, shape=(n_bins+1)
            Common set of bin-edges for all discretized PDFs
        bin_contents: np.ndarray, shape=(n_points, ..., n_bins)
            Discretized PDFs for all grid points and arbitrary further dimensions
            (in IRF term e.g. field-of-view offset bins). Actual interpolation dimension,
            meaning the dimensions that contains actual histograms, has to be along
            the last axis.
        interpolator_cls:
            pyirf interpolator class, defaults to QuantileInterpolator.
        interpolator_kwargs: dict
            Dict of all kwargs that are passed to the interpolator, defaults to
            None which is the same as passing an empty dict.
        extrapolator_cls:
            pyirf extrapolator class. Can be and defaults to ``None``,
            which raises an error if a target_point is outside the grid
            and extrapolation would be needed.
        extrapolator_kwargs: dict
            Dict of all kwargs that are passed to the extrapolator, defaults to
            None which is the same as passing an empty dict.

        Raises
        ------
        TypeError:
            When bin_edges is not a np.ndarray
        TypeError:
            When bin_content is not a np.ndarray
        TypeError:
            When interpolator_cls is not a BinnedInterpolator subclass.
        ValueError:
            When number of bins in bin_edges and contents bin_contents is
            not matching
        ValueError:
            When number of histograms in bin_contents and points in grid_points
            is not matching

        Note
        ----
            Also calls pyirf.component_estimators.BaseComponentEstimator.__init__
        """

        super().__init__(
            grid_points,
        )

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

        if interpolator_kwargs is None:
            interpolator_kwargs = {}

        if extrapolator_kwargs is None:
            extrapolator_kwargs = {}

        if not issubclass(interpolator_cls, DiscretePDFInterpolator):
            raise TypeError(
                f"interpolator_cls must be a DiscretePDFInterpolator subclass, got {interpolator_cls}"
            )

        self.interpolator = interpolator_cls(
            grid_points, bin_edges, bin_contents, **interpolator_kwargs
        )

        if extrapolator_cls is None:
            self.extrapolator = None
        else:
            self.extrapolator = extrapolator_cls(
                grid_points, bin_edges, bin_contents, **extrapolator_kwargs
            )


class ParametrizedComponentEstimator(BaseComponentEstimator):
    def __init__(
        self,
        grid_points,
        params,
        interpolator_cls=GridDataInterpolator,
        interpolator_kwargs=None,
        extrapolator_cls=None,
        extrapolator_kwargs=None,
    ):
        """
        __init__ for all parametrized components, calls BaseComponentEstimator's
        __init__ and instantiates inter- and extrapolator objects.

        Parameters
        ----------
        grid_points: np.ndarray, shape=(n_points, n_dims):
            Grid points at which interpolation templates exist
        params: np.ndarray, shape=(n_points, ..., n_params)
            Corresponding parameter values at each point in grid_points.
            First dimesion has to correspond to number of grid_points.
        interpolator_cls:
            pyirf interpolator class, defaults to GridDataInterpolator.
        interpolator_kwargs: dict
            Dict of all kwargs that are passed to the interpolator, defaults to
            None which is the same as passing an empty dict.
        extrapolator_cls:
            pyirf extrapolator class. Can be and defaults to ``None``,
            which raises an error if a target_point is outside the grid
            and extrapolation would be needed.
        extrapolator_kwargs: dict
            Dict of all kwargs that are passed to the extrapolator, defaults to
            None which is the same as passing an empty dict.

        Raises
        ------
        TypeError:
            When interpolator_cls is not a ParametrizedInterpolator subclass.
        TypeError:
            When params is not a np.ndarray
        ValueError:
            When number of points grid_points and params is not matching

        Note
        ----
            Also calls pyirf.component_estimators.BaseComponentEstimator.__init__
        """

        super().__init__(
            grid_points,
        )

        if not isinstance(params, np.ndarray):
            raise TypeError("Input params is not a numpy array.")
        elif self.n_points != params.shape[0]:
            raise ValueError(
                "Shape missmatch, number of grid_points and rows in params not matching."
            )

        if interpolator_kwargs is None:
            interpolator_kwargs = {}

        if extrapolator_kwargs is None:
            extrapolator_kwargs = {}

        if not issubclass(interpolator_cls, ParametrizedInterpolator):
            raise TypeError(
                f"interpolator_cls must be a ParametrizedInterpolator subclass, got {interpolator_cls}"
            )

        self.interpolator = interpolator_cls(grid_points, params, **interpolator_kwargs)

        if extrapolator_cls is None:
            self.extrapolator = None
        else:
            self.extrapolator = extrapolator_cls(
                grid_points, params, **extrapolator_kwargs
            )


class AEFFEstimator(ParametrizedComponentEstimator):
    @u.quantity_input(effective_area=u.m**2, min_effective_area=u.m**2)
    def __init__(
        self,
        grid_points,
        effective_area,
        interpolator_cls=GridDataInterpolator,
        interpolator_kwargs=None,
        extrapolator_cls=None,
        extrapolator_kwargs=None,
        min_effective_area=1 * u.m**2,
    ):
        """
        Estimator class for effective areas. Takes a grid of effective areas
        for a bunch of different parameters and inter-/extrapolates (log) effective areas
        to given value of those parameters.


        Parameters
        ----------
        grid_points: np.ndarray, shape=(n_points, n_dims):
            Grid points at which interpolation templates exist
        effective_area: np.ndarray of astropy.units.Quantity[area], shape=(n_points, ...)
            Grid of effective area. Dimensions but the first can in principle be freely
            chosen. Class is AEFF2D compatible, which would require
            shape=(n_points, n_energy_bins, n_fov_offset_bins).
        interpolator_cls:
            pyirf interpolator class, defaults to GridDataInterpolator.
        interpolator_kwargs: dict
            Dict of all kwargs that are passed to the interpolator, defaults to
            None which is the same as passing an empty dict.
        extrapolator_cls:
            pyirf extrapolator class. Can be and defaults to ``None``,
            which raises an error if a target_point is outside the grid
            and extrapolation would be needed.
        extrapolator_kwargs: dict
            Dict of all kwargs that are passed to the extrapolator, defaults to
            None which is the same as passing an empty dict.
        min_effective_area: astropy.units.Quantity[area]
            Minimum value of effective area to be considered for interpolation. Values
            lower then this value are set to this value. Defaults to 1 m**2.


        Note
        ----
            Also calls __init__ of pyirf.component_estimators.BaseComponentEstimator
            and ParametrizedEstimator
        """

        # get rid of units
        effective_area = effective_area.to_value(u.m**2)
        min_effective_area = min_effective_area.to_value(u.m**2)

        self.min_effective_area = min_effective_area

        # remove zeros and log it
        effective_area[
            effective_area < self.min_effective_area
        ] = self.min_effective_area
        effective_area = np.log(effective_area)

        super().__init__(
            grid_points=grid_points,
            params=effective_area,
            interpolator_cls=interpolator_cls,
            interpolator_kwargs=interpolator_kwargs,
            extrapolator_cls=extrapolator_cls,
            extrapolator_kwargs=extrapolator_kwargs,
        )

    def __call__(self, target_point):
        """
        Estimating effective area at target_point, inter-/extrapolates as needed and
        specified in __init__.

        Parameters
        ----------
        target_point: np.ndarray, shape=(1, n_dims)
            Target for inter-/extrapolation

        Returns
        -------
        aeff_interp: np.ndarray of (astropy.units.m)**2, shape=(n_points, ...)
            Interpolated Effective area array with same shape as input
            effective areas. For AEFF2D of shape (n_energy_bins, n_fov_offset_bins)

        """

        aeff_interp = super().__call__(target_point)

        # exp it and set to zero too low values
        aeff_interp = np.exp(aeff_interp)
        # 1.1 to correct for numerical uncertainty and interpolation
        aeff_interp[aeff_interp < self.min_effective_area * 1.1] = 0

        return u.Quantity(aeff_interp, u.m**2, copy=False)


class RAD_MAXEstimator(ParametrizedComponentEstimator):
    def __init__(
        self,
        grid_points,
        rad_max,
        interpolator_cls=GridDataInterpolator,
        interpolator_kwargs=None,
        extrapolator_cls=None,
        extrapolator_kwargs=None,
    ):
        """
        Estimator class for effective areas. Takes a grid of effective areas
        for a bunch of different parameters and inter-/extrapolates (log) effective areas
        to given value of those parameters.


        Parameters
        ----------
        grid_points: np.ndarray, shape=(n_points, n_dims):
            Grid points at which interpolation templates exist
        rad_max: np.ndarray, shape=(n_points, ...)
            Grid of theta cuts. Dimensions but the first can in principle be freely
            chosen. Class is RAD_MAX_2D compatible, which would require
            shape=(n_points, n_energy_bins, n_fov_offset_bins).
        interpolator_cls:
            pyirf interpolator class, defaults to GridDataInterpolator.
        interpolator_kwargs: dict
            Dict of all kwargs that are passed to the interpolator, defaults to
            None which is the same as passing an empty dict.
        extrapolator_cls:
            pyirf extrapolator class. Can be and defaults to ``None``,
            which raises an error if a target_point is outside the grid
            and extrapolation would be needed.
        extrapolator_kwargs: dict
            Dict of all kwargs that are passed to the extrapolator, defaults to
            None which is the same as passing an empty dict.

        Note
        ----
            Also calls __init__ of pyirf.component_estimators.BaseComponentEstimator
            and ParametrizedEstimator
        """

        super().__init__(
            grid_points=grid_points,
            params=rad_max,
            interpolator_cls=interpolator_cls,
            interpolator_kwargs=interpolator_kwargs,
            extrapolator_cls=extrapolator_cls,
            extrapolator_kwargs=extrapolator_kwargs,
        )

    def __call__(self, target_point):
        """
        Estimating effective area at target_point, inter-/extrapolates as needed and
        specified in __init__.

        Parameters
        ----------
        target_point: np.ndarray, shape=(1, n_dims)
            Target for inter-/extrapolation

        Returns
        -------
        rad_max_interp: np.ndarray, shape=(n_points, ...)
            Interpolated RAD_MAX table with same shape as input
            effective areas. For RAD_MAX_2D of shape (n_energy_bins, n_fov_offset_bins)

        """

        return super().__call__(target_point)
