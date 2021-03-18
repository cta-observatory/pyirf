import pytest
import astropy.units as u
import numpy as np

pytest.importorskip('gammapy')


true_energy_bins = [0.1, 1, 10, 100] * u.TeV
fov_offset_bins = [0, 1, 2] * u.deg
source_offset_bins = np.linspace(0, 1, 20) * u.deg
migration_bins = np.geomspace(0.2, 5, 10)


def test_effective_area_table_2d():
    from pyirf.gammapy import create_effective_area_table_2d

    shape = (len(true_energy_bins) - 1, len(fov_offset_bins) - 1)
    aeff = np.random.uniform(0, 1e5, size=shape) * u.m**2

    aeff_gammapy = create_effective_area_table_2d(aeff, true_energy_bins, fov_offset_bins)
    # test str repr works

    str(aeff_gammapy)


def test_psf_3d():
    from pyirf.gammapy import create_psf_3d

    shape = (len(true_energy_bins) - 1, len(fov_offset_bins) - 1, len(source_offset_bins) - 1)
    psf = np.zeros(shape) / u.sr
    psf3d = create_psf_3d(psf, true_energy_bins, source_offset_bins,  fov_offset_bins)
    str(psf3d)


def test_energy_dispersion():
    from pyirf.gammapy import create_energy_dispersion_2d

    shape = (len(true_energy_bins) - 1, len(migration_bins) - 1, len(fov_offset_bins) - 1)
    edisp = np.zeros(shape)
    edisp2d = create_energy_dispersion_2d(edisp, true_energy_bins, migration_bins, fov_offset_bins)
    str(edisp2d)
