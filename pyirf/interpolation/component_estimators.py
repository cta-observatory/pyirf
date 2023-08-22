"""Classes to estimate (interpolate/extrapolate) actual IRF HDUs"""
import warnings

import astropy.units as u
import numpy as np
from pyirf.utils import cone_solid_angle
from scipy.spatial import Delaunay

from .base_extrapolators import DiscretePDFExtrapolator, ParametrizedExtrapolator
from .base_interpolators import (
    DiscretePDFInterpolator,
    PDFNormalization,
    ParametrizedInterpolator,
)
from .griddata_interpolator import GridDataInterpolator
from .quantile_interpolator import QuantileInterpolator

__all__ = [
    "BaseComponentEstimator",
    "DiscretePDFComponentEstimator",
    "ParametrizedComponentEstimator",
    "EffectiveAreaEstimator",
    "RadMaxEstimator",
    "EnergyDispersionEstimator",
    "PSFTableEstimator",
]


class BaseComponentEstimator:
    """
    Base class for all Estimators working on specific IRF components.

    While usable, it is encuraged to use the actual class for the respective IRF
    component as it ensures further checks and if nessecarry e.g. unit handling.
    """

    def __init__(self, grid_points):
        """
        Base __init__, doing sanity checks on the grid, building a
        triangulated version of the grid and intantiating inter- and extrapolator.

        Parameters
        ----------
        grid_points: np.ndarray, shape=(n_points, n_dims):

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
    """
    Base class for all Estimators working on IRF components that represent discretized PDFs.

    While usable, it is encuraged to use the actual class for the respective IRF
    component as it ensures further checks and if nessecarry e.g. unit handling.
    """

    def __init__(
        self,
        grid_points,
        bin_edges,
        binned_pdf,
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
            Common set of bin-edges for all discretized PDFs.
        binned_pdf: np.ndarray, shape=(n_points, ..., n_bins)
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
            When bin_edges is not a np.ndarray.
        TypeError:
            When binned_pdf is not a np.ndarray..
        TypeError:
            When interpolator_cls is not a DiscretePDFInterpolator subclass.
        TypeError:
            When extrapolator_cls is not a DiscretePDFExtrapolator subclass.
        ValueError:
            When number of bins in bin_edges and contents in binned_pdf is
            not matching.
        ValueError:
            When number of histograms in binned_pdf and points in grid_points
            is not matching.

        Note
        ----
            Also calls pyirf.interpolation.BaseComponentEstimator.__init__
        """

        super().__init__(
            grid_points,
        )

        if not isinstance(binned_pdf, np.ndarray):
            raise TypeError("Input binned_pdf is not a numpy array.")
        elif self.n_points != binned_pdf.shape[0]:
            raise ValueError(
                f"Shape missmatch, number of grid_points ({self.n_points}) and "
                f"number of histograms in binned_pdf ({binned_pdf.shape[0]}) "
                "not matching."
            )
        elif not isinstance(bin_edges, np.ndarray):
            raise TypeError("Input bin_edges is not a numpy array.")
        elif binned_pdf.shape[-1] != (bin_edges.shape[0] - 1):
            raise ValueError(
                f"Shape missmatch, bin_edges ({bin_edges.shape[0] - 1} bins) "
                f"and binned_pdf ({binned_pdf.shape[-1]} bins) not matching."
            )

        # Make sure that 1D input is sorted in increasing order
        if self.grid_dim == 1:
            sorting_inds = np.argsort(self.grid_points.squeeze())

            self.grid_points = self.grid_points[sorting_inds]
            binned_pdf = binned_pdf[sorting_inds]

        if interpolator_kwargs is None:
            interpolator_kwargs = {}

        if extrapolator_kwargs is None:
            extrapolator_kwargs = {}

        if not issubclass(interpolator_cls, DiscretePDFInterpolator):
            raise TypeError(
                f"interpolator_cls must be a DiscretePDFInterpolator subclass, got {interpolator_cls}"
            )

        self.interpolator = interpolator_cls(
            self.grid_points, bin_edges, binned_pdf, **interpolator_kwargs
        )

        if extrapolator_cls is None:
            self.extrapolator = None
        elif not issubclass(extrapolator_cls, DiscretePDFExtrapolator):
            raise TypeError(
                f"extrapolator_cls must be a DiscretePDFExtrapolator subclass, got {extrapolator_cls}"
            )
        else:
            self.extrapolator = extrapolator_cls(
                self.grid_points, bin_edges, binned_pdf, **extrapolator_kwargs
            )


class ParametrizedComponentEstimator(BaseComponentEstimator):
    """
    Base class for all Estimators working on IRF components that represent parametrized
    or scalar quantities.

    While usable, it is encuraged to use the actual class for the respective IRF
    component as it ensures further checks and if nessecarry e.g. unit handling.
    """

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
            When extrapolator_cls is not a ParametrizedExtrapolator subclass.
        TypeError:
            When params is not a np.ndarray.
        ValueError:
            When number of points grid_points and params is not matching.

        Note
        ----
            Also calls pyirf.interpolation.BaseComponentEstimator.__init__
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

        # Make sure that 1D input is sorted in increasing order
        if self.grid_dim == 1:
            sorting_inds = np.argsort(self.grid_points.squeeze())

            self.grid_points = self.grid_points[sorting_inds]
            params = params[sorting_inds]

        if interpolator_kwargs is None:
            interpolator_kwargs = {}

        if extrapolator_kwargs is None:
            extrapolator_kwargs = {}

        if not issubclass(interpolator_cls, ParametrizedInterpolator):
            raise TypeError(
                f"interpolator_cls must be a ParametrizedInterpolator subclass, got {interpolator_cls}"
            )

        self.interpolator = interpolator_cls(
            self.grid_points, params, **interpolator_kwargs
        )

        if extrapolator_cls is None:
            self.extrapolator = None
        elif not issubclass(extrapolator_cls, ParametrizedExtrapolator):
            raise TypeError(
                f"extrapolator_cls must be a ParametrizedExtrapolator subclass, got {extrapolator_cls}"
            )
        else:
            self.extrapolator = extrapolator_cls(
                self.grid_points, params, **extrapolator_kwargs
            )


class EffectiveAreaEstimator(ParametrizedComponentEstimator):
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
        Estimator class for effective areas.

        Takes a grid of effective areas for a bunch of different parameters
        and inter-/extrapolates (log) effective areas to given value of
        those parameters.


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
            Also calls __init__ of pyirf.interpolation.BaseComponentEstimator
            and pyirf.interpolation.ParametrizedEstimator
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
            effective areas. For AEFF2D of shape (n_energy_bins, n_fov_offset_bins).
            Values lower or equal to __init__'s min_effective_area are set
            to zero.
        """

        aeff_interp = super().__call__(target_point)

        # exp it and set to zero too low values
        aeff_interp = np.exp(aeff_interp)
        # remove entries manipulated by min_effective_area
        aeff_interp[aeff_interp < self.min_effective_area] = 0

        return u.Quantity(aeff_interp, u.m**2, copy=False)


class RadMaxEstimator(ParametrizedComponentEstimator):
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
        Estimator class for RAD_MAX tables.

        Takes a grid of rad max values for a bunch of different parameters
        and inter-/extrapolates rad max values to given value of those parameters.


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
            Also calls __init__ of pyirf.interpolation.BaseComponentEstimator
            and pyirf.interpolation.ParametrizedEstimator
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
        Estimating rad max table at target_point, inter-/extrapolates as needed and
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


class EnergyDispersionEstimator(DiscretePDFComponentEstimator):
    def __init__(
        self,
        grid_points,
        migra_bins,
        energy_dispersion,
        interpolator_cls=QuantileInterpolator,
        interpolator_kwargs=None,
        extrapolator_cls=None,
        extrapolator_kwargs=None,
        axis=-2,
    ):
        """
        Estimator class for energy dispersions.

        Takes a grid of energy dispersions for a bunch of different parameters and
        inter-/extrapolates energy dispersions to given value of those parameters.


        Parameters
        ----------
        grid_points: np.ndarray, shape=(n_points, n_dims)
            Grid points at which interpolation templates exist
        migra_bins: np.ndarray, shape=(n_migration_bins+1)
            Common bin edges along migration axis.
        energy_dispersion: np.ndarray, shape=(n_points, ..., n_migration_bins, ...)
            EDISP MATRIX. Class is EDISP_2D compatible, which would require
            shape=(n_points, n_energy_bins, n_migration_bins, n_fov_offset_bins).
            This is assumed as default. If these axes are in different order
            or e.g. missing a fov_offset axis, the axis containing n_migration_bins
            has to be specified through axis.
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
        axis:
            Axis, along which the actual n_migration_bins are. Input is assumed to
            be EDISP_2D compatible, so this defaults to -2

        Note
        ----
            Also calls __init__ of pyirf.interpolation.BaseComponentEstimator
            and pyirf.interpolation.ParametrizedEstimator
        """

        self.axis = axis

        super().__init__(
            grid_points=grid_points,
            bin_edges=migra_bins,
            binned_pdf=np.swapaxes(energy_dispersion, axis, -1),
            interpolator_cls=interpolator_cls,
            interpolator_kwargs=interpolator_kwargs,
            extrapolator_cls=extrapolator_cls,
            extrapolator_kwargs=extrapolator_kwargs,
        )

    def __call__(self, target_point):
        """
        Estimating energy dispersions at target_point, inter-/extrapolates as needed and
        specified in __init__.

        Parameters
        ----------
        target_point: np.ndarray, shape=(1, n_dims)
            Target for inter-/extrapolation

        Returns
        -------
        edisp_interp: np.ndarray, shape=(n_points, ..., n_migration_bins, ...)
            Interpolated EDISP matrix with same shape as input matrices. For EDISP_2D
            of shape (n_points, n_energy_bins, n_migration_bins, n_fov_offset_bins)

        """

        return np.swapaxes(super().__call__(target_point), -1, self.axis)


class PSFTableEstimator(DiscretePDFComponentEstimator):
    @u.quantity_input(psf=u.sr**-1, source_offset_bins=u.deg)
    def __init__(
        self,
        grid_points,
        source_offset_bins,
        psf,
        interpolator_cls=QuantileInterpolator,
        interpolator_kwargs=None,
        extrapolator_cls=None,
        extrapolator_kwargs=None,
        axis=-1,
    ):
        """
        Estimator class for point spread functions.

        Takes a grid of psfs or a bunch of different parameters and
        inter-/extrapolates psfs to given value of those parameters.


        Parameters
        ----------
        grid_points: np.ndarray, shape=(n_points, n_dims)
            Grid points at which interpolation templates exist
        source_offset_bins: np.ndarray, shape=(n_source_offset_bins+1) of astropy.units.Quantity[deg]
            Common bin edges along source offset axis.
        psf: np.ndarray, shape=(n_points, ..., n_source_offset_bins) of astropy.units.Quantity[sr**-1]
            PSF Tables. Class is PSF_TABLE compatible, which would require
            shape=(n_points, n_energy_bins, n_fov_offset_bins, n_source_offset_bins).
            This is assumed as default. If these axes are in different order
            the axis containing n_source_offset_bins has to be specified through axis.
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
        axis:
            Axis, along which the actual n_source_offset_bins are. Input is assumed to
            be PSF_TABLE compatible, so this defaults to -1

        Note
        ----
            Also calls __init__ of pyirf.interpolation.BaseComponentEstimator
            and pyirf.interpolation.ParametrizedEstimator
        """

        self.axis = axis

        psf = np.swapaxes(psf, axis, -1)

        if interpolator_kwargs is None:
            interpolator_kwargs = {}

        if extrapolator_kwargs is None:
            extrapolator_kwargs = {}

        interpolator_kwargs.setdefault(
            "normalization", PDFNormalization.CONE_SOLID_ANGLE
        )
        extrapolator_kwargs.setdefault(
            "normalization", PDFNormalization.CONE_SOLID_ANGLE
        )

        super().__init__(
            grid_points=grid_points,
            bin_edges=source_offset_bins.to_value(u.rad),
            binned_pdf=psf,
            interpolator_cls=interpolator_cls,
            interpolator_kwargs=interpolator_kwargs,
            extrapolator_cls=extrapolator_cls,
            extrapolator_kwargs=extrapolator_kwargs,
        )

    def __call__(self, target_point):
        """
        Estimating energy dispersions at target_point, inter-/extrapolates as needed and
        specified in __init__.

        Parameters
        ----------
        target_point: np.ndarray, shape=(1, n_dims)
            Target for inter-/extrapolation

        Returns
        -------
        psf_interp: u.Quantity[sr-1], shape=(n_points, ..., n_source_offset_bins)
            Interpolated psf table with same shape as input matrices. For PSF_TABLE
            of shape (n_points, n_energy_bins, n_fov_offset_bins, n_source_offset_bins)

        """

        interpolated_psf_normed = super().__call__(target_point)

        # Undo normalisation to get a proper PSF and return
        return u.Quantity(np.swapaxes(interpolated_psf_normed, -1, self.axis), u.sr**-1, copy=False)
