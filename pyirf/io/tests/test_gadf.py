'''
Test export to / import from GADF format
'''
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import QTable
from pyirf.io import gadf
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

    psf = np.zeros((len(e_bins) - 1, len(fov_bins) - 1, len(source_bins) - 1))
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
        np.geomspace(1e9, 1e3, len(e_bins) - 1),
        np.geomspace(0.5e9, 0.5e3, len(e_bins) - 1),
        np.geomspace(1e8, 1e2, len(e_bins) - 1),
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
            assert u.allclose(area, aeff2d.quantity, atol=1e-16 * u.m**2)


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
            assert u.allclose(edisp, edisp2d.quantity, atol=1e-16)


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
        assert u.allclose(psf, psf3d.quantity, atol=1e-16 / u.sr)


def test_psf_schema(psf_hdu):
    from ogadf_schema.irfs import PSF_TABLE

    _, hdu = psf_hdu
    PSF_TABLE.validate_hdu(hdu)


def test_background_2d_gammapy(bg_hdu):
    '''Test our background hdu is readable by gammapy'''
    from gammapy.irf import Background2D

    background, hdu = bg_hdu

    with tempfile.NamedTemporaryFile(suffix='.fits') as f:
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)

        # test reading with gammapy works
        bg2d = Background2D.read(f.name, 'BACKGROUND')

        assert u.allclose(background, bg2d.quantity, atol=1e-16 * u.Unit('TeV-1 s-1 sr-1'))


def test_background_2d_schema(bg_hdu):
    from ogadf_schema.irfs import BKG_2D

    _, hdu = bg_hdu
    BKG_2D.validate_hdu(hdu)


def test_rad_max_schema(rad_max_hdu):
    from ogadf_schema.irfs import RAD_MAX

    _, hdu = rad_max_hdu
    RAD_MAX.validate_hdu(hdu)


def test_read_cuts():
    """Test of reading cuts."""
    from pyirf.io.gadf import read_irf_cuts

    enbins = np.logspace(-2, 3) * u.TeV
    thcuts = np.linspace(0.5, 0.1) * u.deg
    names = ("Energy", "Theta2")
    t1 = QTable([enbins, thcuts], names=names)
    hdu = fits.BinTableHDU(t1, header=fits.Header(), name='THETA CUTS')
    with tempfile.NamedTemporaryFile(suffix='.fits') as f:
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)
        cuts = read_irf_cuts(f.name)
        assert u.allclose(cuts["Energy"], t1["Energy"], rtol=1.e-15)

        # now check on two files
        cuts = read_irf_cuts([f.name, f.name])[0]
        assert u.allclose(cuts["Theta2"], t1["Theta2"], rtol=1.e-15)


def test_compare_irf_cuts_files():
    """Test of comparing cuts."""
    from pyirf.io.gadf import compare_irf_cuts_in_files

    enbins = np.logspace(-2, 3) * u.TeV
    thcuts1 = np.linspace(0.5, 0.1) * u.deg
    thcuts2 = np.linspace(0.6, 0.2) * u.deg
    names = ("Energy", "Theta2")
    t1 = QTable([enbins, thcuts1], names=names)
    hdu1 = fits.BinTableHDU(t1, header=fits.Header(), name='THETA CUTS')
    t2 = QTable([enbins, thcuts2], names=names)
    hdu2 = fits.BinTableHDU(t2, header=fits.Header(), name='THETA CUTS')
    with tempfile.NamedTemporaryFile(suffix='.fits') as f1:
        fits.HDUList([fits.PrimaryHDU(), hdu1]).writeto(f1.name)
        with tempfile.NamedTemporaryFile(suffix='.fits') as f2:
            fits.HDUList([fits.PrimaryHDU(), hdu2]).writeto(f2.name)
            assert compare_irf_cuts_in_files([f1.name, f1.name])
            assert compare_irf_cuts_in_files([f1.name, f2.name]) is False


def test_read_write_energy_dispersion(edisp_hdus):
    """Test consistency of reading and writing for migration matrix."""

    edisp, hdus = edisp_hdus

    for hdu in hdus:
        with tempfile.NamedTemporaryFile(suffix='.fits') as f:
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)
            edisp2d, e_bins, mig_bins, th_bins = gadf.read_energy_dispersion_hdu(f.name, extname="EDISP")
            # check if the values of migration matrix are the same
            assert u.allclose(edisp, edisp2d, atol=1e-16)
            # check if the sequence of variables is fine
            bins_shape = (e_bins.shape[0] - 1, mig_bins.shape[0] - 1, th_bins.shape[0] - 1)
            assert bins_shape == edisp2d.shape

            # now check with reading two files
            edisp2d, e_bins, mig_bins, th_bins = gadf.read_energy_dispersion_hdu([f.name, f.name], extname="EDISP")
            assert u.allclose(edisp, edisp2d[0], atol=1e-16)
            bins_shape = (2, e_bins.shape[0] - 1, mig_bins.shape[0] - 1, th_bins.shape[0] - 1)
            assert bins_shape == edisp2d.shape

            # now try to read it as a wrong IRF type
            with pytest.raises(ValueError):
                gadf.read_aeff2d_hdu(f.name, extname="EDISP")


def test_read_write_effective_area2d(aeff2d_hdus):
    """Test consistency of reading and writing for effective area."""

    area, hdus = aeff2d_hdus

    for hdu in hdus:
        with tempfile.NamedTemporaryFile(suffix='.fits') as f:
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(f.name)
            aeff2d, e_bins, th_bins = gadf.read_aeff2d_hdu(f.name, extname="EFFECTIVE AREA")
            assert u.allclose(area, aeff2d, atol=-1e-16 * u.m**2)
            bins_shape = (e_bins.shape[0] - 1, th_bins.shape[0] - 1)
            assert bins_shape == aeff2d.shape

            # now check with reading two files
            aeff2d, e_bins, th_bins = gadf.read_aeff2d_hdu([f.name, f.name], extname="EFFECTIVE AREA")
            assert u.allclose(area, aeff2d[0], atol=-1e-16 * u.m**2)
            bins_shape = (2, e_bins.shape[0] - 1, th_bins.shape[0] - 1)
            assert bins_shape == aeff2d.shape

            # now try to read it as a wrong IRF type
            with pytest.raises(ValueError):
                gadf.read_energy_dispersion_hdu(f.name, extname="EFFECTIVE AREA")
