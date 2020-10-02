'''
Test export to GADF format
'''
import astropy.units as u
import numpy as np
from astropy.io import fits
import pytest
import tempfile


e_bins = np.geomspace(0.1, 100, 31) * u.TeV
migra_bins = np.linspace(0.2, 5, 101)
fov_bins = [0, 1, 2, 3] * u.deg
source_bins = np.linspace(0, 1, 101) * u.deg


@pytest.fixture
def aeff2d_hdus():
    from pyirf.io import create_aeff2d_hdu

    area = np.full((len(e_bins) - 1, len(fov_bins) - 1), 1e6) * u.m**2

    hdus = [
        create_aeff2d_hdu(area, e_bins, fov_bins, point_like=point_like)
        for point_like in [True, False]
    ]

    return area, hdus


@pytest.fixture
def edisp_hdus():
    from pyirf.io import create_energy_dispersion_hdu

    edisp = np.zeros((len(e_bins) - 1, len(migra_bins) - 1, len(fov_bins) - 1))
    edisp[:, 50, :] = 1.0

    hdus = [
        create_energy_dispersion_hdu(
            edisp, e_bins, migra_bins, fov_bins, point_like=point_like
        )
        for point_like in [True, False]
    ]

    return edisp, hdus


@pytest.fixture
def psf_hdu():
    from pyirf.io import create_psf_table_hdu
    from pyirf.utils import cone_solid_angle

    psf = np.zeros((len(e_bins) - 1, len(source_bins) - 1, len(fov_bins) - 1))
    psf[:, 0, :] = 1
    psf = psf / cone_solid_angle(source_bins[1])

    hdu = create_psf_table_hdu(
        psf, e_bins, source_bins, fov_bins, point_like=False
    )
    return psf, hdu


@pytest.fixture
def bg_hdu():
    from pyirf.io import create_background_2d_hdu

    background = np.column_stack([
        np.geomspace(1e9, 1e3, 3),
        np.geomspace(0.5e9, 0.5e3, 3),
        np.geomspace(1e8, 1e2, 3),
    ]) * u.Unit('TeV-1 s-1 sr-1')

    hdu = create_background_2d_hdu(background, e_bins, fov_bins)

    return background, hdu


@pytest.fixture
def rad_max_hdu():
    from pyirf.io import create_rad_max_hdu

    rad_max = np.full((len(e_bins) - 1, len(fov_bins) - 1), 0.1) * u.deg
    hdu = create_rad_max_hdu(rad_max, e_bins, fov_bins)

    return rad_max, hdu


def test_effective_area2d_gammapy(aeff2d_hdus):
    '''Test our effective area is readable by gammapy'''
    from gammapy.irf import EffectiveAreaTable2D

    area, hdus = aeff2d_hdus

    for hdu in hdus:
        with tempfile.NamedTemporaryFile(suffix='.fits') as f:
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)
            # test reading with gammapy works
            aeff2d = EffectiveAreaTable2D.read(f.name)
            assert u.allclose(area, aeff2d.data.data, atol=1e-16 * u.m**2)


def test_effective_area2d_schema(aeff2d_hdus):
    '''Test our effective area is readable by gammapy'''
    from ogadf_schema.irfs import AEFF_2D

    _, hdus = aeff2d_hdus

    for hdu in hdus:
        AEFF_2D.validate_hdu(hdu)


def test_energy_dispersion_gammapy(edisp_hdus):
    '''Test our energy dispersion is readable by gammapy'''
    from gammapy.irf import EnergyDispersion2D

    edisp, hdus = edisp_hdus

    for hdu in hdus:
        with tempfile.NamedTemporaryFile(suffix='.fits') as f:
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)

            # test reading with gammapy works
            edisp2d = EnergyDispersion2D.read(f.name, 'EDISP')
            assert u.allclose(edisp, edisp2d.data.data, atol=1e-16)


def test_energy_dispersion_schema(edisp_hdus):
    from ogadf_schema.irfs import EDISP_2D

    _, hdus = edisp_hdus

    for hdu in hdus:
        EDISP_2D.validate_hdu(hdu)


def test_psf_table_gammapy(psf_hdu):
    '''Test our psf is readable by gammapy'''
    from gammapy.irf import PSF3D

    psf, hdu = psf_hdu

    with tempfile.NamedTemporaryFile(suffix='.fits') as f:
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)

        # test reading with gammapy works
        psf3d = PSF3D.read(f.name, 'PSF')

        # gammapy does not transpose psf when reading from fits,
        # unlike how it handles effective area and edisp
        # see https://github.com/gammapy/gammapy/issues/3025
        assert u.allclose(psf, psf3d.psf_value.T, atol=1e-16 / u.sr)


def test_psf_schema(psf_hdu):
    from ogadf_schema.irfs import PSF_TABLE

    _, hdu = psf_hdu
    PSF_TABLE.validate_hdu(hdu)


# gammapy uses inconsistent axis order, should be fixed before gammapy 1.0
# see https://github.com/gammapy/gammapy/issues/2067
# TODO: remove xfail when this is fixed in gammapy and bump the required gammapy version
@pytest.mark.xfail
def test_background_2d_gammapy(bg_hdu):
    '''Test our background hdu is readable by gammapy'''
    from gammapy.irf import Background2D

    background, hdu = bg_hdu

    with tempfile.NamedTemporaryFile(suffix='.fits') as f:
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)

        # test reading with gammapy works
        bg2d = Background2D.read(f.name, 'BACKGROUND')

        assert u.allclose(background, bg2d.data.data, atol=1e-16 * u.Unit('TeV-1 s-1 sr-1'))


def test_background_2d_schema(bg_hdu):
    from ogadf_schema.irfs import BKG_2D

    _, hdu = bg_hdu
    BKG_2D.validate_hdu(hdu)


def test_rad_max_schema(rad_max_hdu):
    from ogadf_schema.irfs import RAD_MAX

    _, hdu = rad_max_hdu
    RAD_MAX.validate_hdu(hdu)
