"""Functions for performing interpolation of IRF to the values read from the data."""

import numpy as np
import astropy.units as u
from scipy.interpolate import griddata


def interpolate_effective_area_per_energy_and_fov(effective_area, grid_points, target_point, min_effective_area=1. * u.Unit('m2'), method='linear'):
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

    _, _, n_energy_bins = effective_area.shape

    # get rid of units
    effective_area = effective_area.to_value(u.m**2)
    min_effective_area = min_effective_area.to_value(u.m**2)

    # remove zeros and log it
    effective_area[effective_area < min_effective_area] = min_effective_area
    effective_area = np.log(effective_area)

    # interpolation
    aeff_interp = griddata(grid_points, effective_area, target_point, method=method).T
    # exp it and set to zero too low values
    aeff_interp = np.exp(aeff_interp)
    aeff_interp[aeff_interp < min_effective_area * 1.1] = 0  # 1.1 to correct for numerical uncertainty and interpolation
    return u.Quantity(aeff_interp, u.m**2, copy=False)


def interpolate_energy_dispersion(energy_dispersions, grid_points, target_point, method='linear'):
    """
    Takes a grid of dispersion matrixes for a bunch of different parameters
    and interpolates it to given value of those parameters

    Parameters
    ----------
    energy_dispersions: np.array of astropy.units.Quantity[area]
        grid of effective area, of shape (n_grid_points, n_energy_bins, n_migration_bins, n_fov_offset_bins)
    grid_points: np.array
        list of parameters corresponding to energy_dispersions, of shape (n_grid_points, n_interp_dim)
    target_point: np.array
        values of parameters for which the interpolation is performed, of shape (n_interp_dim)
    method: 'linear’, ‘nearest’, ‘cubic’
        Interpolation method

    Returns
    -------
    matrix_interp: astropy.units.Quantity[area]
        Interpolated dispersion matrix 3D array with shape (n_energy_bins, n_migration_bins, n_fov_offset_bins)
    """

    # interpolation
    matrix_interp = griddata(grid_points, energy_dispersions, target_point, method=method)

    # now we need to renormalize along the migration axis
    norm = np.sum(matrix_interp, axis=1, keepdims=True)
    # By using out and where, it is ensured that columns with norm = 0 will have 0 values without raising an invalid value warning
    mig_norm = np.divide(matrix_interp, norm, out=np.zeros_like(matrix_interp), where=norm != 0)
    return mig_norm


