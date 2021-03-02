import pyirf.interpolation as interp
import json
from astropy.io import fits
import numpy as np
import astropy.units as u


def test_read_mean_parameters_data():
    """Test of reading of average parameters from a data DL2 file."""
    config_file = 'interp_test_data/interpol_irf.json'
    # definition from lstchain.io.io
    dl2_params_lstcam_key = 'dl2/event/telescope/parameters/LST_LSTCam'

    data_file = '../../data/dl2_LST-1.Run03642.0110.h5'
    with open(config_file) as pars_file:
        config = json.load(pars_file)
    pars = config['interpol_irf']['pars']

    interp_pos = interp.read_mean_parameters_data(data_file, dl2_params_lstcam_key, pars)
    # true average values in that file
    interp_pos_true = (0.8337625653926475, -63.52860081844798)
    assert np.allclose(interp_pos, interp_pos_true, rtol=0.01)


def test_interpolate_effective_area():
    """Test of interpolating of effective area using dummy model files."""
    n_en = 20
    n_th = 1
    en = np.logspace(-2, 2, n_en)
    # applying a simple sigmoid function
    aeff0 = 1.e4 / (1 + 1 / en**2) * u.Unit('m2')

    # assume that for parameters 'x' and 'y' the Aeff scales x*y*Aeff0
    x = [0.9, 1.1]
    y = [8., 11.5]
    n_grid = len(x) * len(y)
    aeff = np.empty((n_grid, n_th, n_en))
    pars = np.empty((n_grid, 2))
    i_grid = 0
    for xx in x:
        for yy in y:
            aeff[i_grid, 0, :] = aeff0 * xx * yy / 10
            pars[i_grid, :] = np.array([xx, yy])
            i_grid += 1
    aeff *= u.Unit('m2')
    pars0 = [1, 10]
    min_aeff = 1 * u.Unit('m2')
    aeff_interp = interp.interpolate_effective_area(aeff, pars, pars0, min_effective_area=min_aeff, method='linear')
    # allowing for 3% accuracy except of close to the minimum value of Aeff
    assert np.allclose(aeff_interp[:, 0], aeff0, rtol=0.03, atol=min_aeff)


def test_read_unit_from_HDUL():
    """Test of reading units from a field in a fits files."""
    with fits.open('interp_test_data/pyirf_eventdisplay_68.fits.gz') as hdul:
        unit = interp.read_unit_from_HDUL(hdul, "EFFECTIVE_AREA", "EFFAREA")
        unit_true = u.Unit("m2")
    assert unit == unit_true


def test_interpolate_dispersion_matrix():
    """Test of interpolation of energy dispersion matrix using a simple dummy model."""
    x = [0.9, 1.1]
    y = [8., 11.5]
    n_grid = len(x) * len(y)
    n_offset = 1
    n_en = 30
    n_mig = 20
    clip_level = 1.e-3

    # define simple dummy bias and resolution model using two parameters x and y
    def get_bias_std(i_en, x, y):
        i_en = i_en + 3 * ((x - 1) + (y - 10.))
        de = n_en - i_en
        de[de < 0] = 0.
        bias = de**0.5 + n_mig / 2
        rms = 5 - 2 * (i_en / n_en)
        bias[i_en < 3] = 2 * n_mig  # return high values to zero out part of the table
        rms[i_en < 3] = 0
        return bias, rms

    en = np.arange(n_en)[:, np.newaxis]
    mig = np.arange(n_mig)[np.newaxis, :]

    # auxiliary function to compute profile of the 2D distribution
    # used to check if the expected and interpolated matrixes are similar
    def calc_mean_std(matrix):
        n_en = matrix.shape[0]
        means = np.empty(n_en)
        stds = np.empty(n_en)
        for i_en in np.arange(n_en):
            w = matrix[i_en, :]
            if np.sum(w) > 0:
                means[i_en] = np.average(mig[0, :], weights=w)
                stds[i_en] = np.sqrt(np.cov(mig[0, :], aweights=w))
            else:  # we need to skip the empty columns
                means[i_en] = -1
                stds[i_en] = -1
        return means, stds

    # generate true values
    interp_pars = (1, 10)
    bias, sigma = get_bias_std(en, *interp_pars)
    mig_true = np.exp(-(mig - bias)**2 / (2 * sigma**2))
    mig_true[mig_true < clip_level] = 0

    # generate a grid of migration matrixes
    i_grid = 0
    pars_all = np.empty((n_grid, 2))
    mig_all = np.empty((n_grid, n_offset, n_mig, n_en))
    for xx in x:
        for yy in y:
            bias, sigma = get_bias_std(en, xx, yy)
            mig_all[i_grid, 0, :, :] = (np.exp(-(mig - bias)**2 / (2 * sigma**2))).T
            pars_all[i_grid, :] = (xx, yy)
            i_grid += 1
    # do the interpolation and compare the results with expected ones
    mig_interp = interp.interpolate_dispersion_matrix(mig_all, pars_all, interp_pars, method='linear')

    # check if all the energy bins have normalization 1 or 0 (can happen because of empty bins)
    sums = np.sum(mig_interp[:, :, 0], axis=1)
    assert np.logical_or(np.isclose(sums, 0., atol=1.e-5), np.isclose(sums, 1., atol=1.e-5)).min()

    # now check if we reconstruct the mean and sigma roughly fine after interpolation
    bias0, stds0 = calc_mean_std(mig_true)  # true
    bias, stds = calc_mean_std(mig_interp[:, :, 0])  # interpolated

    # first remove the bins that are empty in true value
    idxs = bias0 > 0
    bias0 = bias0[idxs]
    bias = bias[idxs]
    stds0 = stds0[idxs]
    stds = stds[idxs]
    # allowing for a 0.6 bin size error on the interpolated values
    assert np.allclose(bias, bias0, atol=0.6, rtol=0.)
    assert np.allclose(stds, stds0, atol=0.6, rtol=0.)


def test_compare_irf_cuts():
    """Test of cut consistency using 3 files: two same ones and one different."""
    file1a = 'interp_test_data/pyirf_eventdisplay_68.fits.gz'
    file1b = 'interp_test_data/pyirf_eventdisplay_68_copy.fits.gz'
    file2 = 'interp_test_data/pyirf_eventdisplay_80.fits.gz'

    match = interp.compare_irf_cuts([file1a, file1b], 'THETA_CUTS')
    assert match
    match = interp.compare_irf_cuts([file1a, file1b, file2], 'THETA_CUTS')
    assert not match
