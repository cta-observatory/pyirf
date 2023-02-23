import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

from pyirf.utils import gadf3gauss_to_multigauss, normalize_multigauss


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

    t = QTable({"bar": [0, 1, 2] * u.TeV})

    with pytest.raises(MissingColumns):
        check_table(t, required_columns=["foo"])

    t = QTable({"bar": [0, 1, 2] * u.TeV})
    with pytest.raises(WrongColumnUnit):
        check_table(t, required_units={"bar": u.m})

    t = QTable({"bar": [0, 1, 2] * u.m})
    with pytest.raises(MissingColumns):
        check_table(t, required_units={"foo": u.m})

    # m is convertible
    check_table(t, required_units={"bar": u.cm})


def test_multigauss_to_gadf():
    from pyirf.utils import multigauss_to_gadf3gauss

    multigauss = np.recarray(
        shape=(1),
        dtype=[
            ("p_1", "<f4"),
            ("p_2", "<f4"),
            ("p_3", "<f4"),
            ("sigma_1", "<f4"),
            ("sigma_2", "<f4"),
            ("sigma_3", "<f4"),
        ],
    )

    multigauss[:] = (0.5, 0.25, 0.25, 0.1, 0.1, 0.1)

    expected_gadf3gauss = np.recarray(
        shape=(1),
        dtype=[
            ("scale", "<f4"),
            ("ampl_2", "<f4"),
            ("ampl_3", "<f4"),
            ("sigma_1", "<f4"),
            ("sigma_2", "<f4"),
            ("sigma_3", "<f4"),
        ],
    )

    expected_gadf3gauss[:] = (
        25 * (1 / u.deg**2).to(1 / u.sr).value,
        0.5,
        0.5,
        0.1,
        0.1,
        0.1,
    )

    transformed_gadf3gauss = multigauss_to_gadf3gauss(multigauss)

    for name in expected_gadf3gauss.dtype.names:
        assert transformed_gadf3gauss[name] == expected_gadf3gauss[name]


def test_gadf_to_multigauss():
    from pyirf.utils import gadf3gauss_to_multigauss

    gadf3gauss = np.recarray(
        shape=(1),
        dtype=[
            ("scale", "<f4"),
            ("ampl_2", "<f4"),
            ("ampl_3", "<f4"),
            ("sigma_1", "<f4"),
            ("sigma_2", "<f4"),
            ("sigma_3", "<f4"),
        ],
    )

    gadf3gauss[:] = (25 * (1 / u.deg**2).to(1 / u.sr).value, 0.5, 0.5, 0.1, 0.1, 0.1)

    expected_multigauss = np.recarray(
        shape=(1),
        dtype=[
            ("p_1", "<f4"),
            ("p_2", "<f4"),
            ("p_3", "<f4"),
            ("sigma_1", "<f4"),
            ("sigma_2", "<f4"),
            ("sigma_3", "<f4"),
        ],
    )

    expected_multigauss[:] = (0.5, 0.25, 0.25, 0.1, 0.1, 0.1)

    transformed_multigauss = gadf3gauss_to_multigauss(gadf3gauss)

    for name in expected_multigauss.dtype.names:
        assert expected_multigauss[name] == transformed_multigauss[name]


def test_normalize_multigauss():
    from pyirf.utils import normalize_multigauss

    multigauss = np.recarray(
        shape=(1),
        dtype=[
            ("p_1", "<f4"),
            ("p_2", "<f4"),
            ("p_3", "<f4"),
            ("sigma_1", "<f4"),
            ("sigma_2", "<f4"),
            ("sigma_3", "<f4"),
        ],
    )

    multigauss[:] = (1, 0.5, 0.5, 0.1, 0.1, 0.1)

    normed_multigauss = normalize_multigauss(multigauss)

    expected_multigauss = np.recarray(
        shape=(1),
        dtype=[
            ("p_1", "<f4"),
            ("p_2", "<f4"),
            ("p_3", "<f4"),
            ("sigma_1", "<f4"),
            ("sigma_2", "<f4"),
            ("sigma_3", "<f4"),
        ],
    )

    expected_multigauss[:] = (0.5, 0.25, 0.25, 0.1, 0.1, 0.1)

    for name in expected_multigauss.dtype.names:
        assert expected_multigauss[name] == normed_multigauss[name]


def test_normalize_3gadfgauss():
    from pyirf.utils import normalize_gadf3gauss

    gadf3gauss = np.recarray(
        shape=(1),
        dtype=[
            ("scale", "<f4"),
            ("ampl_2", "<f4"),
            ("ampl_3", "<f4"),
            ("sigma_1", "<f4"),
            ("sigma_2", "<f4"),
            ("sigma_3", "<f4"),
        ],
    )

    gadf3gauss[:] = (
        2 * 25 * (1 / u.deg**2).to(1 / u.sr).value,
        0.5,
        0.5,
        0.1,
        0.1,
        0.1,
    )

    normed_gadf3gauss = normalize_gadf3gauss(gadf3gauss)

    expected_gadf3gauss = np.recarray(
        shape=(1),
        dtype=[
            ("scale", "<f4"),
            ("ampl_2", "<f4"),
            ("ampl_3", "<f4"),
            ("sigma_1", "<f4"),
            ("sigma_2", "<f4"),
            ("sigma_3", "<f4"),
        ],
    )

    expected_gadf3gauss[:] = (
        25 * (1 / u.deg**2).to(1 / u.sr).value,
        0.5,
        0.5,
        0.1,
        0.1,
        0.1,
    )

    for name in expected_gadf3gauss.dtype.names:
        assert normed_gadf3gauss[name] == expected_gadf3gauss[name]
