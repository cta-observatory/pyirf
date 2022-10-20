import astropy.units as u
import pytest
import numpy as np


def test_table_interpolation():

    from pyirf.spectral import TableInterpolationSpectrum

    # log log
    energy = [1, 10, 100] * u.TeV
    flux = [50, 5, 0.5] / (u.GeV * u.m**2 * u.sr * u.s)

    spectrum = TableInterpolationSpectrum(energy, flux)
    assert u.isclose(spectrum(5 * u.TeV), 10 / (u.GeV * u.m**2 * u.sr * u.s))


    # lin lin
    energy = [1, 2, 3] * u.TeV
    flux = [10, 8, 6] / (u.GeV * u.m**2 * u.sr * u.s)

    spectrum = TableInterpolationSpectrum(energy, flux, log_energy=False, log_flux=False)
    assert u.isclose(spectrum(1.5 * u.TeV), 9 / (u.GeV * u.m**2 * u.sr * u.s))



def test_powerlaw_integrate_cone_invalid():
    from pyirf.spectral import PowerLaw, POINT_SOURCE_FLUX_UNIT

    point_source = PowerLaw(
        normalization=1 * POINT_SOURCE_FLUX_UNIT,
        index=-2,
        e_ref=500 * u.GeV,
    )
    with pytest.raises(ValueError):
        point_source.integrate_cone(0 * u.deg, 2 * u.deg)


@pytest.mark.parametrize(
    "outer,expected",
    zip((90 * u.deg, 180 * u.deg), (2 * np.pi * u.sr, 4 * np.pi * u.sr))
)
def test_powerlaw_integrate_cone(outer, expected):
    from pyirf.spectral import PowerLaw, POINT_SOURCE_FLUX_UNIT, DIFFUSE_FLUX_UNIT
    diffuse_flux = PowerLaw(
        normalization=1 * DIFFUSE_FLUX_UNIT,
        index=-2,
        e_ref=500 * u.GeV,
    )


    integrated = diffuse_flux.integrate_cone(0 * u.rad, outer)
    assert integrated.normalization.unit.is_equivalent(POINT_SOURCE_FLUX_UNIT)
    assert u.isclose(integrated.normalization, diffuse_flux.normalization * expected)
    assert integrated.index == diffuse_flux.index
    assert integrated.e_ref == diffuse_flux.e_ref


def test_powerlaw():
    from pyirf.spectral import PowerLaw

    with pytest.raises(TypeError):
        PowerLaw(normalization=1e-10, index=-2)

    with pytest.raises(u.UnitsError):
        PowerLaw(normalization=1e-10 / u.TeV, index=-2)

    with pytest.raises(ValueError):
        PowerLaw(normalization=1e-10 / u.TeV / u.m**2 / u.s, index=2)

    # check we get a reasonable unit out of astropy independent of input unit
    unit = u.TeV**-1 * u.m**-2 * u.s**-1
    power_law = PowerLaw(1e-10 * unit, -2.65)
    assert power_law(1 * u.TeV).unit == unit
    assert power_law(1 * u.GeV).unit == unit
