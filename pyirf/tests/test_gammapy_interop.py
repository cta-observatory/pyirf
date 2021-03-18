import pytest
import astropy.units as u
import numpy as np

pytest.importorskip('gammapy')


def test_effective_area_table_2d():
    from pyirf.gammapy import create_effective_area_2d_table

    true_energy_bins = [0.1, 1, 10, 100] * u.TeV
    fov_offset_bins = [0, 1, 2] * u.deg
    shape = (len(true_energy_bins) - 1, len(fov_offset_bins) - 1)
    aeff = np.random.uniform(0, 1e5, size=shape) * u.m**2

    aeff_gammapy = create_effective_area_2d_table(aeff, true_energy_bins, fov_offset_bins)
    # test str repr works

    str(aeff_gammapy)
