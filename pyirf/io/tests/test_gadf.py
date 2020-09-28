'''
Test export to GADF format
'''
import astropy.units as u
import numpy as np
from astropy.io import fits
import pytest
import tempfile


def test_effective_area2d():
    '''Test our effective area is readable by gammapy'''
    pytest.importorskip('gammapy')
    from pyirf.io import create_aeff2d_hdu
    from gammapy.irf import EffectiveAreaTable2D

    e_bins = np.geomspace(0.1, 100, 31) * u.TeV
    fov_bins = [0, 1, 2, 3] * u.deg
    area = np.full((30, 3), 1e6) * u.m**2

    for point_like in [True, False]:
        with tempfile.NamedTemporaryFile(suffix='.fits') as f:
            hdu = create_aeff2d_hdu(area, e_bins, fov_bins, point_like=point_like)

            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)

            # test reading with gammapy works
            aeff2d = EffectiveAreaTable2D.read(f.name)
            assert u.allclose(area, aeff2d.data.data, atol=1e-16 * u.m**2)


def test_energy_dispersion():
    '''Test our energy dispersion is readable by gammapy'''
    pytest.importorskip('gammapy')
    from pyirf.io import create_energy_dispersion_hdu
    from gammapy.irf import EnergyDispersion2D

    e_bins = np.geomspace(0.1, 100, 31) * u.TeV
    migra_bins = np.linspace(0.2, 5, 101)
    fov_bins = [0, 1, 2, 3] * u.deg
    edisp = np.zeros((30, 100, 3))
    edisp[:, 50, :] = 1.0


    for point_like in [True, False]:
        with tempfile.NamedTemporaryFile(suffix='.fits') as f:
            hdu = create_energy_dispersion_hdu(
                edisp, e_bins, migra_bins, fov_bins, point_like=point_like
            )

            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)

            # test reading with gammapy works
            edisp2d = EnergyDispersion2D.read(f.name, 'EDISP')
            assert u.allclose(edisp, edisp2d.data.data, atol=1e-16)


def test_psf_table():
    '''Test our psf is readable by gammapy'''
    pytest.importorskip('gammapy')
    from pyirf.io import create_psf_table_hdu
    from pyirf.utils import cone_solid_angle
    from gammapy.irf import PSF3D

    e_bins = np.geomspace(0.1, 100, 31) * u.TeV
    source_bins = np.linspace(0, 1, 101) * u.deg
    fov_bins = [0, 1, 2, 3] * u.deg
    psf = np.zeros((30, 100, 3))
    psf[:, 0, :] = 1
    psf = psf / cone_solid_angle(source_bins[1])

    for point_like in [True, False]:
        with tempfile.NamedTemporaryFile(suffix='.fits') as f:
            hdu = create_psf_table_hdu(
                psf, e_bins, source_bins, fov_bins, point_like=point_like
            )

            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)

            # test reading with gammapy works
            psf3d = PSF3D.read(f.name, 'PSF')

            # gammapy does not transpose psf when reading from fits,
            # unlike how it handles effective area and edisp
            # see https://github.com/gammapy/gammapy/issues/3025
            assert u.allclose(psf, psf3d.psf_value.T, atol=1e-16 / u.sr)


# gammapy uses inconsistent axis order, should be fixed before gammapy 1.0
# see https://github.com/gammapy/gammapy/issues/2067
# TODO: remove xfail when this is fixed in gammapy and bump the required gammapy version
@pytest.mark.xfail
def test_background_2d():
    '''Test our background hdu is readable by gammapy'''

    pytest.importorskip('gammapy')
    from pyirf.io import create_background_2d_hdu
    from gammapy.irf import Background2D

    e_bins = np.geomspace(0.1, 100, 31) * u.TeV
    fov_bins = [0, 1, 2, 3] * u.deg
    background = np.column_stack([
        np.geomspace(1e9, 1e3, 3),
        np.geomspace(0.5e9, 0.5e3, 3),
        np.geomspace(1e8, 1e2, 3),
    ]) * u.Unit('TeV-1 s-1 sr-1')

    with tempfile.NamedTemporaryFile(suffix='.fits') as f:
        hdu = create_background_2d_hdu(background, e_bins, fov_bins)

        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)

        # test reading with gammapy works
        bg2d = Background2D.read(f.name, 'BACKGROUND')

        assert u.allclose(background, bg2d.data.data, atol=1e-16 * u.Unit('TeV-1 s-1 sr-1'))
