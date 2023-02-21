"""Functions for performing interpolation of IRF to the values read from the data."""

import astropy.units as u
import numpy as np

from .griddata_interpolator import GridDataInterpolator
from .quantile_interpolator import QuantileInterpolator
from pyirf.utils import cone_solid_angle

__all__ = [
    "interpolate_effective_area_per_energy_and_fov",
    "interpolate_energy_dispersion",
    "interpolate_psf_table",
    "interpolate_rad_max",
]


@u.quantity_input(effective_area=u.m**2)
def interpolate_effective_area_per_energy_and_fov(
    effective_area,
    grid_points,
    target_point,
    min_effective_area=1 * u.m**2,
    method="linear",
):
    """
    Takes a grid of effective areas for a bunch of different parameters
    and interpolates (log) effective areas to given value of those parameters

    Parameters
    ----------
    effective_area: np.array of astropy.units.Quantity[area]
        grid of effective area, of shape (n_grid_points, n_fov_offset_bins, n_energy_bins)
    grid_points: np.array
        list of parameters corresponding to effective_area, of shape (n_grid_points, n_interp_dim)
    target_point: np.array
        values of parameters for which the interpolation is performed, of shape (n_interp_dim)
    min_effective_area: astropy.units.Quantity[area]
        Minimum value of effective area to be considered for interpolation
    method: 'linear’, ‘nearest’, ‘cubic’
        Interpolation method

    Returns
    -------
    aeff_interp: astropy.units.Quantity[area]
        Interpolated Effective area array with shape (n_energy_bins, n_fov_offset_bins)
    """

    # get rid of units
    effective_area = effective_area.to_value(u.m**2)
    min_effective_area = min_effective_area.to_value(u.m**2)

    # remove zeros and log it
    effective_area[effective_area < min_effective_area] = min_effective_area
    effective_area = np.log(effective_area)

    # interpolation
    interp = GridDataInterpolator(
        grid_points=grid_points,
        params=effective_area,
    )
    aeff_interp = interp(target_point, method=method)
    # exp it and set to zero too low values
    aeff_interp = np.exp(aeff_interp)
    # 1.1 to correct for numerical uncertainty and interpolation
    aeff_interp[aeff_interp < min_effective_area * 1.1] = 0
    return u.Quantity(aeff_interp, u.m**2, copy=False)


def interpolate_energy_dispersion(
    migra_bins, edisps, grid_points, target_point, quantile_resolution=1e-3
):
    """
    Takes a grid of energy dispersions for a bunch of different parameters
    and interpolates it to given value of those parameters.

    Parameters
    ----------
    migra_bins: numpy.ndarray, shape=(M+1)
        Common array of migration bin-edges

    edisps: numpy.ndarray, shape=(N, ..., M,...)
        The EDISP MATRIX, shape is assumed to be (N:n_grid_points, n_energy_bins, M:n_migration_bins, n_fov_offset_bins).

    grid_points: numpy.ndarray, shape=(N, O)
        Array of the N O-dimensional morphing parameter values corresponding to the N input templates. The pdf's quantiles
        are expected to vary linearly between these two reference points.

    target_point: numpy.ndarray, shape=(O)
        Value for which the interpolation is performed (target point)

    quantile_resolution: float
        Spacing between the quantiles that are computed in the interpolation. Defaults to 1e-3.

    Returns
    -------
    edisp_interp: numpy.ndarray, shape=(1,...,M,...)
        Interpolated and binned energy dispersion.
    """
    interp = QuantileInterpolator(
        bin_edges=migra_bins,
        bin_contents=edisps,
        grid_points=grid_points,
        axis=-2,
        quantile_resolution=quantile_resolution,
    )

    return interp(target_point)


@u.quantity_input(psf=u.sr**-1, source_offset_bins=u.deg)
def interpolate_psf_table(
    source_offset_bins, psfs, grid_points, target_point, quantile_resolution=1e-3
):
    """
    Takes a grid of PSF tables for a bunch of different parameters
    and interpolates it to given value of those parameters

    Parameters
    ----------
    source_offset_bins: numpy.ndarray, shape=(M+1), of astropy.units.Quantity[deg]
        Common array of source offset bin-edges

    psfs: numpy.ndarray, shape=(N, ..., M), of astropy.units.Quantity[sr**-1]
        The PSF_TABLE, shape is assumed to be (N:n_grid_points, n_energy_bins, n_fov_offset_bins, M:n_source_offset_bins).

    grid_points: numpy.ndarray, shape=(N, O)
        Array of the N O-dimensional morphing parameter values corresponding to the N input templates. The pdf's quantiles
        are expected to vary linearly between these two reference points.

    target_point: numpy.ndarray, shape=(O)
        Value for which the interpolation is performed (target point)

    quantile_resolution: float
        Spacing between the quantiles that are computed in the interpolation. Defaults to 1e-3.

    Returns
    -------
    psf_interp: numpy.ndarray, shape=(1, ..., M)
        Interpolated PSF table with shape (n_energy_bins, n_fov_offset_bins, n_source_offset_bins)
    """

    # Renormalize along the source offset axis to have a proper PDF
    omegas = np.diff(cone_solid_angle(source_offset_bins))
    psfs_normed = psfs * omegas

    interp = QuantileInterpolator(
        bin_edges=source_offset_bins,
        bin_contents=psfs_normed,
        grid_points=grid_points,
        axis=-1,
        quantile_resolution=quantile_resolution,
    )

    interpolated_psf_normed = interp(target_point)

    # Undo normalisation to get a proper PSF and return
    return interpolated_psf_normed / omegas


def interpolate_rad_max(
    rad_max,
    grid_points,
    target_point,
    method="linear",
):
    """
    Interpolates a grid of RAD_MAX tables for point-like IRFs to a target-point.
    Wrapper around scipy.interpolate.griddata [1].

    Parameters
    ----------
    rad_max: numpy.ndarray, shape=(N, M, ...)
        Theta-cuts for all combinations of grid-points, energy and fov_offset.
        Shape (N:n_grid_points, M:n_energy_bins, n_fov_offset_bins)

    grid_points: numpy.ndarray, shape=(N, O)
        Array of the N O-dimensional morphing parameter values corresponding to the N input templates.

    target_point: numpy.ndarray, shape=(O)
        Value for which the interpolation is performed (target point)

    method: 'linear', 'nearest', 'cubic'
        Interpolation method for scipy.interpolate.griddata [1]. Defaults to 'linear'.

    Returns
    -------
    rad_max_interp: numpy.ndarray, shape=(1, M, ...)
        Theta-cuts for the target grid-point, shape (1, M:n_energy_bins, n_fov_offset_bins)

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
    """

    interp = GridDataInterpolator(grid_points=grid_points, params=rad_max)
    return interp(target_point, method=method)
