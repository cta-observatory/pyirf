import numpy as np
import astropy.units as u


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
