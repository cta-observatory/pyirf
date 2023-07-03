from astropy.table import QTable
import astropy.units as u
import numpy as np
import pytest


def test_empty_angular_resolution():
    from pyirf.benchmarks import angular_resolution

    events = QTable({
        'true_energy': [] * u.TeV,
        'theta': [] * u.deg,
    })

    table = angular_resolution(
        events,
        [1, 10, 100] * u.TeV
    )

    assert np.all(np.isnan(table["angular_resolution"]))

@pytest.mark.parametrize("unit", (u.deg, u.rad))
def test_angular_resolution(unit):
    from pyirf.benchmarks import angular_resolution

    np.random.seed(1337)

    TRUE_RES_1 = 0.2
    TRUE_RES_2 = 0.05
    true_resolution = np.append(np.full(1000, TRUE_RES_1), np.full(1000, TRUE_RES_2))

    events = QTable({
        'true_energy': np.concatenate([
            [0.5], # below bin 1 to test with underflow
            np.full(999, 5.0),
            np.full(999, 50.0),
            [500], # above bin 2 to test with overflow
        ]) * u.TeV,
        'theta': np.abs(np.random.normal(0, true_resolution)) * u.deg
    })

    events['theta'] = events['theta'].to(unit)

    # add nans to test if nans are ignored
    events["true_energy"].value[500] = np.nan
    events["true_energy"].value[1500] = np.nan

    table = angular_resolution(
        events,
        [1, 10, 100] * u.TeV,
    )

    ang_res = table['angular_resolution']
    assert len(ang_res) == 2
    assert u.isclose(ang_res[0], TRUE_RES_1 * u.deg, rtol=0.05)
    assert u.isclose(ang_res[1], TRUE_RES_2 * u.deg, rtol=0.05)

    # one value in each bin is nan, which is ignored
    np.testing.assert_array_equal(table["n_events"], [998, 998])
