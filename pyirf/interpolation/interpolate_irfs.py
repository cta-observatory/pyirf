"""Functions for performing interpolation of IRF to the values read from the data."""

import astropy.units as u
import numpy as np
from pyirf.utils import cone_solid_angle

from .griddata_interpolator import GridDataInterpolator
from .quantile_interpolator import QuantileInterpolator

__all__ = [
    "interpolate_energy_dispersion",
    "interpolate_psf_table",
]


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
