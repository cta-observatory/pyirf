import pyirf.interpolation as interp
import numpy as np
import astropy.units as u
import pytest

def test_interpolate_effective_area_per_energy_and_fov():
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
    pars0 = (1, 10)
    min_aeff = 1 * u.Unit('m2')
    aeff_interp = interp.interpolate_effective_area_per_energy_and_fov(aeff, pars, pars0, min_effective_area=min_aeff, method='linear')
    # allowing for 3% accuracy except of close to the minimum value of Aeff
    assert np.allclose(aeff_interp[:, 0], aeff0, rtol=0.03, atol=min_aeff)


def calc_mean_std(matrix, vals):
    """Auxiliary function to compute mean and std from 'matrix' along an axis which values are in 'values'."""
    n_en = matrix.shape[0]
    means = np.empty(n_en)
    stds = np.empty(n_en)
    for i_en in np.arange(n_en):
        w = matrix[i_en, :]
        if np.sum(w) > 0:
            means[i_en] = np.average(vals, weights=w)
            stds[i_en] = np.sqrt(np.cov(vals, aweights=w))
        else:  # we need to skip the empty columns
            means[i_en] = -1
            stds[i_en] = -1
    return means, stds


def test_interpolate_energy_dispersion():
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

    # generate true values
    interp_pars = (1, 10)
    bias, sigma = get_bias_std(en, *interp_pars)
    mig_true = np.exp(-(mig - bias)**2 / (2 * sigma**2))
    mig_true[mig_true < clip_level] = 0

    # generate a grid of migration matrixes
    i_grid = 0
    pars_all = np.empty((n_grid, 2))
    mig_all = np.empty((n_grid, n_en, n_mig, n_offset))
    for xx in x:
        for yy in y:
            bias, sigma = get_bias_std(en, xx, yy)
            mig_all[i_grid, :, :, 0] = (np.exp(-(mig - bias)**2 / (2 * sigma**2)))
            pars_all[i_grid, :] = (xx, yy)
            i_grid += 1
    # do the interpolation and compare the results with expected ones
    mig_interp = interp.interpolate_energy_dispersion(mig_all, pars_all, interp_pars, method='linear')

    # check if all the energy bins have normalization 1 or 0 (can happen because of empty bins)
    sums = np.sum(mig_interp[:, :, 0], axis=1)
    assert np.logical_or(np.isclose(sums, 0., atol=1.e-5), np.isclose(sums, 1., atol=1.e-5)).min()

    # now check if we reconstruct the mean and sigma roughly fine after interpolation
    bias0, stds0 = calc_mean_std(mig_true, mig[0, :])  # true
    bias, stds = calc_mean_std(mig_interp[:, :, 0], mig[0, :])  # interpolated

    # first remove the bins that are empty in true value
    idxs = bias0 > 0
    bias0 = bias0[idxs]
    bias = bias[idxs]
    stds0 = stds0[idxs]
    stds = stds[idxs]
    # allowing for a 0.6 bin size error on the interpolated values
    assert np.allclose(bias, bias0, atol=0.6, rtol=0.)
    assert np.allclose(stds, stds0, atol=0.6, rtol=0.)

@pytest.mark.parametrize("cumulative", [False, True])
def test_interpolate_psf_table(cumulative):
    """Test of interpolation of PSF tables using a simple dummy model"""
    from pyirf.utils import cone_solid_angle
    x = [0.9, 1.1]
    y = [8., 11.5]
    n_grid = len(x) * len(y)
    n_offset = 1
    n_en = 30
    n_src_off = 20

    # define simple dummy model for angular resolution using two parameters x and y
    def get_sigma(i_en, x, y):
        i_en = i_en + 3 * ((x - 1) + (y - 10.))
        return np.exp(-(i_en - 30) / 20)

    en = np.arange(n_en)[:, np.newaxis, np.newaxis]
    src_bins = np.arange(n_src_off + 1) * u.deg
    omegas = np.diff(cone_solid_angle(src_bins))
    src_off = np.arange(n_src_off)[np.newaxis, np.newaxis, :]

    # generate true values
    interp_pars = (1, 10)
    sigma = get_sigma(en, *interp_pars)
    psf_true = np.exp(-src_off**2 / (2 * sigma**2)) * u.Unit('sr-1')

    # generate a grid of PSF tables
    i_grid = 0
    pars_all = np.empty((n_grid, 2))
    psfs_all = np.empty((n_grid, n_en, n_offset, n_src_off))
    for xx in x:
        for yy in y:
            sigma = get_sigma(en, xx, yy)
            psfs_all[i_grid] = np.exp(-src_off**2 / (2 * sigma**2))
            pars_all[i_grid, :] = (xx, yy)
            i_grid += 1
    psfs_all *= u.Unit('sr-1')

    # do the interpolation and compare the results with expected ones
    psf_interp = interp.interpolate_psf_table(psfs_all, pars_all, interp_pars, src_bins, cumulative=cumulative, method='linear')

    # check if all the energy bins have normalization 1 or 0 (can happen because of empty bins)
    sums = np.sum(psf_interp * omegas, axis=2)
    assert np.logical_or(np.isclose(sums, 0., atol=1.e-5), np.isclose(sums, 1., atol=1.e-5)).min()

    # check the first two moments of the distribution for every energy
    mean_true, std_true = calc_mean_std(psf_true[:, 0, :], src_off[0, 0, :])
    mean_interp, std_interp = calc_mean_std(psf_interp[:, 0, :], src_off[0, 0, :])

    # check if they are within 10% + 0.25 of bin size
    assert np.allclose(mean_interp, mean_true, atol=0.25, rtol=0.1)
    assert np.allclose(std_interp, std_true, atol=0.25, rtol=0.1)
