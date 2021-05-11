from astropy.table import QTable
import astropy.units as u
import numpy as np
from scipy.stats import norm
from itertools import product


def test_empty_bias_resolution():
    from pyirf.benchmarks import energy_bias_resolution

    events = QTable({
        'true_energy': [] * u.TeV,
        'reco_energy': [] * u.TeV,
    })

    table = energy_bias_resolution(
        events,
        [1, 10, 100] * u.TeV
    )

    assert np.all(np.isnan(table["bias"]))
    assert np.all(np.isnan(table["resolution"]))

def test_energy_bias_resolution():
    from pyirf.benchmarks import energy_bias_resolution

    np.random.seed(1337)

    TRUE_RES_1 = 0.2
    TRUE_RES_2 = 0.05
    TRUE_BIAS_1 = 0.1
    TRUE_BIAS_2 = -0.05

    true_bias = np.append(np.full(1000, TRUE_BIAS_1), np.full(1000, TRUE_BIAS_2))
    true_resolution = np.append(np.full(1000, TRUE_RES_1), np.full(1000, TRUE_RES_2))

    true_energy = np.concatenate([
        [0.5], # below bin 1 to test with underflow
        np.full(999, 5.0),
        np.full(999, 50.0),
        [500], # above bin 2 to test with overflow
    ]) * u.TeV
    reco_energy = true_energy * (1 + np.random.normal(true_bias, true_resolution))

    events = QTable({
        'true_energy': true_energy,
        'reco_energy': reco_energy,
    })

    bias_resolution = energy_bias_resolution(
        events,
        [1, 10, 100] * u.TeV
    )

    bias = bias_resolution['bias'].quantity
    resolution = bias_resolution['resolution'].quantity

    assert len(bias) == len(resolution) == 2

    assert u.isclose(bias[0], TRUE_BIAS_1, rtol=0.05)
    assert u.isclose(bias[1], TRUE_BIAS_2, rtol=0.05)
    assert u.isclose(resolution[0], TRUE_RES_1, rtol=0.05)
    assert u.isclose(resolution[1], TRUE_RES_2, rtol=0.05)



def test_energy_bias_resolution():
    from pyirf.benchmarks import energy_bias_resolution_from_energy_dispersion
    from pyirf.binning import bin_center

    # create a toy energy dispersion
    n_migra_bins = 500
    true_bias = np.array([
        [0.5, 0.6],
        [0, 0.1],
        [-0.1, -0.2],
    ])
    true_resolution = np.array([
        [0.4, 0.5],
        [0.2, 0.3],
        [0.1, 0.15],
    ])

    n_energy_bins, n_fov_bins = true_bias.shape
    energy_bins = np.geomspace(10, 1000, n_energy_bins + 1)
    energy_center = bin_center(energy_bins)
    migra_bins = np.geomspace(0.2, 5, n_migra_bins + 1)

    cdf = np.empty((n_energy_bins, n_migra_bins + 1, n_fov_bins))
    for energy_bin, fov_bin in product(range(n_energy_bins), range(n_fov_bins)):

        energy = energy_center[energy_bin]
        mu = (1 + true_bias[energy_bin, fov_bin]) * energy
        sigma = true_resolution[energy_bin, fov_bin] * energy
        reco_energy = migra_bins * energy

        cdf[energy_bin, :, fov_bin] = norm.cdf(reco_energy, mu, sigma)

    edisp = cdf[:, 1:, :] - cdf[:, :-1, :]

    bias, resolution = energy_bias_resolution_from_energy_dispersion(
        edisp,
        migra_bins,
    )

    assert np.allclose(bias, true_bias, atol=0.01)
    assert np.allclose(resolution, true_resolution, atol=0.01)
