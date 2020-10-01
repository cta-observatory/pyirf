import numpy as np
import astropy.units as u
from astropy.table import QTable
import pytest


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
    from pyirf.exceptions import MissingColumns, WrongColumnUnit
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
