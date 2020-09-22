import numpy as np
import astropy.units as u


def test_is_scalar():
    from pyirf.cuts import is_scalar

    assert is_scalar(1.0)
    assert is_scalar(5 * u.m)
    assert is_scalar(np.array(5))

    assert not is_scalar([1, 2, 3])
    assert not is_scalar([1, 2, 3] * u.m)
    assert not is_scalar(np.ones(5))
    assert not is_scalar(np.ones((3, 4)))
