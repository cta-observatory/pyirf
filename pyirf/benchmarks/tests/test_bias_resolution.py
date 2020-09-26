from astropy.table import QTable
import astropy.units as u
import numpy as np


def test_energy_bias_resolution():
    from pyirf.benchmarks import energy_bias_resolution

    np.random.seed(1337)

    TRUE_RES_1 = 0.2
    TRUE_RES_2 = 0.05
    TRUE_BIAS_1 = 0.1
    TRUE_BIAS_2 = -0.05

    true_bias = np.append(np.full(1000, TRUE_BIAS_1), np.full(1000, TRUE_BIAS_2))
    true_resolution = np.append(np.full(1000, TRUE_RES_1), np.full(1000, TRUE_RES_2))

    true_energy = np.append(np.full(1000, 5.0), np.full(1000, 50.0)) * u.TeV
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
