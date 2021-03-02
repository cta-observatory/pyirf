"""
Functions for performing interpolation of IRF to the values read from the data
"""
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from scipy.interpolate import griddata

# definition from lstchain.io.io copied here to avoid dependency
dl2_params_lstcam_key = 'dl2/event/telescope/parameters/LST_LSTCam'


def read_mean_parameters_data(data_file, parameters):
    """
    Reads a DL2 data file and extracts the average values
    of the parameters for interpolation

    Parameters
    ----------
    data_file: ``string``
        path to the DL2 data file
    parameters: list of ``string``
        list of parameters as they can be evaluated from DL2 file

    Returns
    -------
    interp_pos: tuple
        tuple of average values of requested parameters
    """

    # read in the data
    interp_pos = []  # position for which to interpolate
    data = pd.read_hdf(data_file, key=dl2_params_lstcam_key)
    for par in parameters:
        # we use here eval function that is considered potentially dangerous
        # as it can execute arbitrary code, however this is the eval
        # function from pandas, that is very much limitted to just
        # the columns read from the file and math operations
        # so it should be safe here (and it adds a lot of flexibility)
        val = np.mean(data.eval(par[1]))
        interp_pos.append(val)

    return tuple(interp_pos)


def interpolate_effective_area(aeff_all, pars_all, interp_pars, min_effective_area=1. * u.Unit('m2'), method='linear'):
    """
    Takes a grid of effective areas for a bunch of different parameters
    and interpolates (log) effective areas to given value of those parameters

    Parameters
    ----------
    aeff_all: np.array of astropy.units.Quantity[area]
        grid of effective area, of shape (n_grid_points, n_fov_offset_bins, n_energy_bins)
    pars_all: np.array
        list of parameters corresponding to aeff_all, of shape (n_grid_points, n_interp_dim)
    interp_pars: np.array
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

    _, n_fov_offset_bins, n_energy_bins = aeff_all.shape

    # get rid of units
    aeff_all = aeff_all.to_value(u.m**2)
    min_effective_area = min_effective_area.to('m2').value

    # remove zeros and log it
    aeff_all[aeff_all < min_effective_area] = min_effective_area
    aeff_all = np.log(aeff_all)

    # interpolation
    aeff_interp = np.empty((n_energy_bins, n_fov_offset_bins))
    for i_th in range(n_fov_offset_bins):
        for i_en in range(n_energy_bins):
            aeff_interp[i_en, i_th] = griddata(pars_all, aeff_all[:, i_th, i_en], interp_pars, method=method)

    # exp it and set to zero too low values
    aeff_interp = np.exp(aeff_interp)
    aeff_interp[aeff_interp < min_effective_area * 1.1] = 0  # 1.1 to correct for numerical uncertainty and interpolation
    return u.Quantity(aeff_interp, u.m**2, copy=False)


def interpolate_dispersion_matrix(matrix_all, pars_all, interp_pars, method='linear'):
    """
    Takes a grid of dispersion matrixes for a bunch of different parameters
    and interpolates it to given value of those parameters

    Parameters
    ----------
    matrix_all: np.array of astropy.units.Quantity[area]
        grid of effective area, of shape (n_grid_points, n_fov_offset_bins, n_migration_bins, n_energy_bins)
    pars_all: np.array
        list of parameters corresponding to matrix_all, of shape (n_grid_points, n_interp_dim)
    interp_pars: np.array
        values of parameters for which the interpolation is performed, of shape (n_interp_dim)
    method: 'linear’, ‘nearest’, ‘cubic’
        Interpolation method

    Returns
    -------
    matrix_interp: astropy.units.Quantity[area]
        Interpolated dispersion matrix 3D array with shape (n_energy_bins, n_migration_bins, n_fov_offset_bins)
    """

    _, n_fov_offset_bins, n_migration_bins, n_energy_bins = matrix_all.shape

    # interpolation
    r_th = range(n_fov_offset_bins)
    r_en = range(n_energy_bins)
    r_mig = range(n_migration_bins)
    matrix_interp = np.empty((n_energy_bins, n_migration_bins, n_fov_offset_bins))
    # TO DO this part can be optimized because nested for loops take quite some time
    # but it is not a big problem because this has to be done only once per run
    for i_th in r_th:
        for i_en in r_en:
            for i_mig in r_mig:
                matrix_interp[i_en, i_mig, i_th] = griddata(pars_all, matrix_all[:, i_th, i_mig, i_en], interp_pars, method=method)

    # now we need to renormalize along the migration axis
    norm = np.sum(matrix_interp, axis=1, keepdims=True)
    mig_norm = np.divide(matrix_interp, norm, out=np.zeros_like(matrix_interp), where=norm != 0)
    return mig_norm


def read_unit_from_HDUL(hdul, ext_name, field_name):
    """
    Searches for a field in FITS header of a given extension and checks its unit

    Parameters
    ----------
    hdul: astropy.io.fits.hdu.image.PrimaryHDU
        FITS HDU
    ext_name: string
        name of the extension to read the data from in fits file
    field_name: string
        name of the field in the extension to extract

    Returns
    -------
    unit: astropy.units.core.Unit
        unit
    """
    keys = list(hdul[ext_name].header['TTYPE*'].keys())
    vals = list(hdul[ext_name].header['TTYPE*'].values())

    TTYPE = keys[vals.index(field_name)]
    TUNIT = TTYPE.replace('TYPE', 'UNIT')
    if hdul[ext_name].header[TUNIT] == '':
        return u.Unit('')  # dimentionless
    return u.format.Fits.parse(hdul[ext_name].header[TUNIT])


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

    with fits.open(file_name) as hdul:
        ext_tab = hdul[ext_name].data[0]
        bins = list(ext_tab[tag_lo])
        bins.append(ext_tab[tag_hi][-1])
        bins *= read_unit_from_HDUL(hdul, ext_name, tag_lo)
    return bins


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
    pars_all: np.array
        list of parameters corresponding to aeff_all, of shape (n_grid_points, n_interp_dim)
    energy_bins: astropy.units.Quantity[energy]
        energy bins
    theta_bins: astropy.units.Quantity[angle]
        theta bins
    """

    n_grid_point = len(files)
    interp_dim = len(files[0]) - 1  # number of parameters to interpolate
    pars_all = np.empty((n_grid_point, interp_dim))

    # open the first file to check the binning
    energy_bins = read_fits_bins_lo_hi(files[0][0], ext_name, 'ENERG')
    theta_bins = read_fits_bins_lo_hi(files[0][0], ext_name, 'THETA')

    n_theta = len(theta_bins) - 1  # number of bins in offset angle

    irfs_all = np.empty((n_grid_point, n_theta), dtype=np.object)
    for ifile, this_file in enumerate(files):
        file_name = this_file[0]
        pars_all[ifile, :] = this_file[1:]

        with fits.open(file_name) as hdul:
            for i_th in range(n_theta):
                ext_tab = hdul[ext_name].data[0]
                irfs_all[ifile, i_th] = ext_tab[field_name][i_th]

    # convert irfs to a simple array and add unit
    irfs_all = np.array(irfs_all.tolist()) * read_unit_from_HDUL(hdul, ext_name, field_name)

    return irfs_all, pars_all, energy_bins, theta_bins


def compare_irf_cuts(files, ext_name):
    """
    Reads in a list of IRF files and checks if the same cuts have been applied in all of them

    Parameters
    ----------
    files: list of strings
        files to be read
    ext_name: string
        name of the extension with cut values to read the data from in fits file

    Returns
    -------
    match: Boolean
        if the cuts are the same in all the files
    """
    with fits.open(files[0]) as hdul0:
        data0 = hdul0['THETA_CUTS'].data

    for file_name in files[1:]:
        with fits.open(file_name) as hdul:
            data = hdul['THETA_CUTS'].data
            if (data != data0).any():
                print("difference between file: " + files[0] + " and " + file_name + " in cut values: " + ext_name)
                return False

    return True
