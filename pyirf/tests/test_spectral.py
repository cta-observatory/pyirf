import astropy.units as u
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
