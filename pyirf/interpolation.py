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


def read_mean_pars_data(data_file, pars):
    """
    Reads a DL2 data file and extracts the average values
    of the parameters for interpolation

    Parameters
    ----------
    data_file: ``string``
        path to the DL2 data file
    pars: list of ``string``
        list of parameters as they can be evaluated from DL2 file

    Returns
    -------
    interp_pos: tuple
        tuple of average values of requested parameters
    """

    # read in the data
    interp_pos = []  # position for which to interpolate
    data = pd.read_hdf(data_file, key=dl2_params_lstcam_key)
    for par in pars:
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

    n_grid_point, n_fov_offset_bins, n_energy_bins = aeff_all.shape

    # get rid of units
    aeff_all = aeff_all.to('m2').value
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
    return aeff_interp * u.Unit('m2')


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
    keys=list(hdul[ext_name].header['TTYPE*'].keys())
    vals=list(hdul[ext_name].header['TTYPE*'].values())

    print(keys)
    print(vals)
    TTYPE=keys[vals.index(field_name)]
    TUNIT=TTYPE.replace('TYPE','UNIT')
    return u.format.Fits.parse(hdul[ext_name].header[TUNIT])


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
    with fits.open(files[0][0]) as hdul:
        ext_tab = hdul[ext_name].data[0]
        energy_bins = list(ext_tab['ENERG_LO'])
        energy_bins.append(ext_tab['ENERG_HI'][-1])
        theta_bins = list(ext_tab['THETA_LO'])
        theta_bins.append(ext_tab['THETA_HI'][-1])

        # add units, maybe extracting them here is not needed since the units are supposed
        # to be standard so could be hardcoded, but just in case...
        energy_bins*=read_unit_from_HDUL(hdul, ext_name, 'ENERG_LO')
        theta_bins*=read_unit_from_HDUL(hdul, ext_name, 'THETA_LO')

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
    irfs_all = np.array(irfs_all.tolist())*read_unit_from_HDUL(hdul, ext_name, field_name)

    # TBD: check in the input fits file that the units are indeed TeV, deg
    return irfs_all, pars_all, energy_bins, theta_bins
