from astropy.table import QTable
import astropy.units as u
import numpy as np


def test_angular_resolution():
    from pyirf.benchmarks import angular_resolution

    np.random.seed(1337)

    TRUE_RES_1 = 0.2
    TRUE_RES_2 = 0.05
    true_resolution = np.append(np.full(1000, TRUE_RES_1), np.full(1000, TRUE_RES_2))

    events = QTable({
        'true_energy': np.append(np.full(1000, 5.0), np.full(1000, 50.0)) * u.TeV,
        'theta': np.abs(np.random.normal(0, true_resolution)) * u.deg
    })

    ang_res = angular_resolution(
        events,
        [1, 10, 100] * u.TeV
    )['angular_resolution'].quantity

    assert len(ang_res) == 2
    assert u.isclose(ang_res[0], TRUE_RES_1 * u.deg, rtol=0.05)
    assert u.isclose(ang_res[1], TRUE_RES_2 * u.deg, rtol=0.05)
