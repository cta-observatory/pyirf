"""Functions for performing interpolation of IRF to the values read from the data"""

import numpy as np
import astropy.units as u
from astropy.table import Table
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

    _, n_fov_offset_bins, n_energy_bins = effective_area.shape

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
        grid of effective area, of shape (n_grid_points, n_fov_offset_bins, n_migration_bins, n_energy_bins)
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

    _, n_fov_offset_bins, n_migration_bins, n_energy_bins = energy_dispersions.shape

    # interpolation
    matrix_interp = griddata(grid_points, energy_dispersions, target_point, method=method)
    matrix_interp = np.swapaxes(matrix_interp, 0, 2)

    # now we need to renormalize along the migration axis
    norm = np.sum(matrix_interp, axis=1, keepdims=True)
    # By using out and where, it is ensured that columns with norm = 0 will have 0 values without raising an invalid value warning
    mig_norm = np.divide(matrix_interp, norm, out=np.zeros_like(matrix_interp), where=norm != 0)
    return mig_norm


def read_fits_bins_lo_hi(file_name, ext_name, tag):
    """
    Reads from a fits file two arrays of tag_LO and tag_HI and joins them into a single array and adds unit

    Parameters
    ----------
    file_name: string
        file to be read
    ext_name: string
        name of the extension to read the data from in fits file
    tag: string
        name of the field in the extension to extract, _LO and _HI will be added

    Returns
    -------
    bins: astropy.units.Quantity[energy]
        bins
    """

    tag_lo = tag + '_LO'
    tag_hi = tag + '_HI'

    table = Table.read(file_name, hdu=ext_name)
    bins = list(table[tag_lo][0])
    bins.append(table[tag_hi][0][-1])
    return u.Quantity(bins, table[tag_lo].unit, copy=False)


def read_irf_grid(files, ext_name, field_name):
    """
    Reads in a grid of IRFs for a bunch of different parameters and stores them in lists

    Parameters
    ----------
    files: nested list
        files to be read, each element has a form [filename, values of interpolation parameters]
    ext_name: string
        name of the extension to read the data from in fits file
    field_name: string
        name of the field in the extension to extract

    Returns
    -------
    irfs_all: np.array
        array of IRFs
    grid_points: np.array
        list of parameters corresponding to effective_area, of shape (n_grid_points, n_interp_dim)
    energy_bins: astropy.units.Quantity[energy]
        energy bins
    theta_bins: astropy.units.Quantity[angle]
        theta bins
    """

    n_grid_point = len(files)
    interp_dim = len(files[0]) - 1  # number of parameters to interpolate
    grid_points = np.empty((n_grid_point, interp_dim))

    # open the first file to check the binning
    energy_bins = read_fits_bins_lo_hi(files[0][0], ext_name, 'ENERG')
    theta_bins = read_fits_bins_lo_hi(files[0][0], ext_name, 'THETA')

    n_theta = len(theta_bins) - 1  # number of bins in offset angle

    irfs_all = np.empty((n_grid_point, n_theta), dtype=np.object)
    for ifile, this_file in enumerate(files):
        file_name = this_file[0]
        grid_points[ifile, :] = this_file[1:]

        table = Table.read(file_name, hdu=ext_name)
        for i_th in range(n_theta):
            irfs_all[ifile, i_th] = table[field_name][0][i_th]

    # convert irfs to a simple array and add unit
    irfs_all = u.Quantity(np.array(irfs_all.tolist()), table[field_name].unit, copy=False)

    return irfs_all, grid_points, energy_bins, theta_bins


