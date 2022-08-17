import numpy as np
import astropy.units as u
from astropy.table import QTable, Table
import pytest
from pyirf.exceptions import MissingColumns, WrongColumnUnit


def test_is_scalar():
    from pyirf.utils import is_scalar

    assert is_scalar(1.0)
    assert is_scalar(5 * u.m)
    assert is_scalar(np.array(5))

    assert not is_scalar([1, 2, 3])
    assert not is_scalar([1, 2, 3] * u.m)
    assert not is_scalar(np.ones(5))
    assert not is_scalar(np.ones((3, 4)))


def test_cone_solid_angle():
    from pyirf.utils import cone_solid_angle

    # whole sphere
    assert u.isclose(cone_solid_angle(np.pi * u.rad), 4 * np.pi * u.sr)

    # half the sphere
    assert u.isclose(cone_solid_angle(90 * u.deg), 2 * np.pi * u.sr)

    # zero
    assert u.isclose(cone_solid_angle(0 * u.deg), 0 * u.sr)


def test_check_table():
    from pyirf.utils import check_table

    t = QTable({'bar': [0, 1, 2] * u.TeV})

    with pytest.raises(MissingColumns):
        check_table(t, required_columns=['foo'])

    t = QTable({'bar': [0, 1, 2] * u.TeV})
    with pytest.raises(WrongColumnUnit):
        check_table(t, required_units={'bar': u.m})

    t = QTable({'bar': [0, 1, 2] * u.m})
    with pytest.raises(MissingColumns):
        check_table(t, required_units={'foo': u.m})

    # m is convertible
    check_table(t, required_units={'bar': u.cm})

    t = Table({'bar': [0, 1, 2]})
    with pytest.raises(WrongColumnUnit):
        check_table(t, required_units={'bar': u.cm})

def test_calculate_theta():
    from pyirf.utils import calculate_theta

    true_az = true_alt = u.Quantity([1.0], u.deg)
    t = QTable({'reco_alt': true_alt, 'reco_az': true_az})

    assert u.isclose(
        calculate_theta(
            events=t,
            assumed_source_az=true_az,
            assumed_source_alt=true_alt,
        ),
        0.0 * u.deg,
    )

    t = Table({'reco_alt': [1.0], 'reco_az': [1.0]})
    with pytest.raises(WrongColumnUnit):
        calculate_theta(t, true_az, true_alt)

def test_calculate_source_fov_offset():
    from pyirf.utils import calculate_source_fov_offset

    a = u.Quantity([1.0], u.deg)
    t = QTable({
        'pointing_az': a,
        'pointing_alt': a,
        'true_az': a,
        'true_alt': a,
    })

    assert u.isclose(calculate_source_fov_offset(t), 0.0 * u.deg)

def test_check_histograms():
    from pyirf.binning import create_histogram_table
    from pyirf.utils import check_histograms

    events1 = QTable({
        'reco_energy': [1, 1, 10, 100, 100, 100] * u.TeV,
    })
    events2 = QTable({
        'reco_energy': [100, 100, 100] * u.TeV,
    })
    bins = [0.5, 5, 50, 500] * u.TeV

    hist1 = create_histogram_table(events1, bins)
    hist2 = create_histogram_table(events2, bins)
    check_histograms(hist1, hist2)

    hist3 = create_histogram_table(events1, [0, 10] * u.TeV)
    with pytest.raises(ValueError):
        check_histograms(hist1, hist3)
