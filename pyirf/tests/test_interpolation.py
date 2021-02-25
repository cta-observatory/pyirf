import pyirf.interpolation as interp
import json
import pytest
import numpy as np
import astropy.units as u


# temporarily disabled (until the data are available in repository)
def _test_read_mean_pars_data():
    config_file = 'interp_test_data/interpol_irf.json'
    data_file = 'interp_test_data/dl2_LST-1.Run03642.0110.h5'
    with open(config_file) as pars_file:
        config = json.load(pars_file)
    pars = config['interpol_irf']['pars']

    interp_pos = interp.read_mean_pars_data(data_file, pars)
    # true average values in that file
    interp_pos_true = (0.8337625653926475, -63.52860081844798)
    assert np.allclose(interp_pos, interp_pos_true, rtol=0.01)


def test_interpolate_effective_area():
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
    print('Aeff=', aeff)
    print('Aeff0=', aeff0)
    print('Aeff_intp=', aeff_interp)
    print('ratio=', aeff_interp[:, 0] / aeff0)
    # allowing for 3% accuracy except of close to the minimum value of Aeff
    assert np.allclose(aeff_interp[:, 0], aeff0, rtol=0.03, atol=min_aeff)
