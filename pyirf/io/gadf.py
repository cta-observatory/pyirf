from astropy.table import QTable
import astropy.units as u
from astropy.io.fits import Header, BinTableHDU
import numpy as np

from ..version import __version__


__all__ = [
    'create_aeff2d_hdu',
    'create_energy_dispersion_hdu',
    'create_psf_table_hdu',
    'create_rad_max_hdu',
]


DEFAULT_HEADER = Header()
DEFAULT_HEADER['CREATOR'] = f'pyirf v{__version__}'
DEFAULT_HEADER['HDUDOC'] = 'https://gamma-astro-data-formats.readthedocs.io'
DEFAULT_HEADER['HDUVERS'] = '0.2'
DEFAULT_HEADER['HDUCLASS'] = 'GADF'


def _add_header_cards(header, **header_cards):
    for k, v in header_cards.items():
        header[k] = v


@u.quantity_input(effective_area=u.m**2, true_energy_bins=u.TeV, fov_offset_bins=u.deg)
def create_aeff2d_hdu(
    effective_area, true_energy_bins, fov_offset_bins,
    extname='EFFECTIVE AREA', point_like=True, **header_cards
):
    aeff = QTable()
    aeff['ENERG_LO'] = u.Quantity(true_energy_bins[:-1], ndmin=2).to(u.TeV)
    aeff['ENERG_HI'] = u.Quantity(true_energy_bins[1:], ndmin=2).to(u.TeV)
    aeff['THETA_LO'] = u.Quantity(fov_offset_bins[:-1], ndmin=2).to(u.deg)
    aeff['THETA_HI'] = u.Quantity(fov_offset_bins[1:], ndmin=2).to(u.deg)
    aeff['EFFAREA'] = effective_area[np.newaxis, ...].to(u.m**2)

    # required header keywords
    header = DEFAULT_HEADER.copy()
    header['HDUCLAS1'] = 'RESPONSE'
    header['HDUCLAS2'] = 'EFF_AREA'
    header['HDUCLAS3'] = 'POINT-LIKE' if point_like else 'FULL-ENCLOSURE'
    header['HDUCLAS4'] = 'AEFF_2D'
    _add_header_cards(header, **header_cards)

    return BinTableHDU(aeff, header=header, name=extname)


@u.quantity_input(
    psf=u.sr**-1, true_energy_bins=u.TeV, fov_offset_bins=u.deg,
    source_offset_bins=u.deg,
)
def create_psf_table_hdu(
    psf, true_energy_bins, source_offset_bins, fov_offset_bins,
    point_like=True,
    extname='PSF', **header_cards
):

    psf = QTable({
        'ENERG_LO': u.Quantity(true_energy_bins[:-1], ndmin=2).to(u.TeV),
        'ENERG_HI': u.Quantity(true_energy_bins[1:], ndmin=2).to(u.TeV),
        'THETA_LO': u.Quantity(fov_offset_bins[:-1], ndmin=2).to(u.deg),
        'THETA_HI': u.Quantity(fov_offset_bins[1:], ndmin=2).to(u.deg),
        'RAD_LO': u.Quantity(source_offset_bins[:-1], ndmin=2).to(u.deg),
        'RAD_HI': u.Quantity(source_offset_bins[1:], ndmin=2).to(u.deg),
        'RPSF': psf[np.newaxis, ...].to(1 / u.sr),
    })

    # required header keywords
    header = DEFAULT_HEADER.copy()
    header['HDUCLAS1'] = 'RESPONSE'
    header['HDUCLAS2'] = 'PSF'
    header['HDUCLAS3'] = 'POINT-LIKE' if point_like else 'FULL-ENCLOSURE'
    header['HDUCLAS4'] = 'PSF_TABLE'
    _add_header_cards(header, **header_cards)

    return BinTableHDU(psf, header=header, name=extname)


@u.quantity_input(
    true_energy_bins=u.TeV,
    fov_offset_bins=u.deg,
)
def create_energy_dispersion_hdu(
    energy_dispersion,
    true_energy_bins,
    migration_bins,
    fov_offset_bins,
    point_like=True,
    extname='EDISP', **header_cards
):

    psf = QTable({
        'ENERG_LO': u.Quantity(true_energy_bins[:-1], ndmin=2).to(u.TeV),
        'ENERG_HI': u.Quantity(true_energy_bins[1:], ndmin=2).to(u.TeV),
        'MIGRA_LO': u.Quantity(migration_bins[:-1], ndmin=2).to(u.one),
        'MIGRA_HI': u.Quantity(migration_bins[1:], ndmin=2).to(u.one),
        'THETA_LO': u.Quantity(fov_offset_bins[:-1], ndmin=2).to(u.deg),
        'THETA_HI': u.Quantity(fov_offset_bins[1:], ndmin=2).to(u.deg),
        'MATRIX': u.Quantity(energy_dispersion[np.newaxis, ...]).to(u.one),
    })

    # required header keywords
    header = DEFAULT_HEADER.copy()
    header['HDUCLAS1'] = 'RESPONSE'
    header['HDUCLAS2'] = 'EDISP'
    header['HDUCLAS3'] = 'POINT-LIKE' if point_like else 'FULL-ENCLOSURE'
    header['HDUCLAS4'] = 'EDISP_2D'
    _add_header_cards(header, **header_cards)

    return BinTableHDU(psf, header=header, name=extname)


@u.quantity_input(
    psf=u.sr**-1, true_energy_bins=u.TeV, fov_offset_bins=u.deg,
    source_offset_bins=u.deg,
)
def create_rad_max_hdu(
    reco_energy_bins, fov_offset_bins, rad_max,
    point_like=True,
    extname='RAD_MAX', **header_cards
):
    rad_max_table = QTable({
        'ENERG_LO': u.Quantity(reco_energy_bins[:-1], ndmin=2).to(u.TeV),
        'ENERG_HI': u.Quantity(reco_energy_bins[1:], ndmin=2).to(u.TeV),
        'THETA_LO': u.Quantity(fov_offset_bins[:-1], ndmin=2).to(u.deg),
        'THETA_HI': u.Quantity(fov_offset_bins[1:], ndmin=2).to(u.deg),
        'RAD_MAX': rad_max[np.newaxis, ...].to(u.deg)
    })

    # required header keywords
    header = DEFAULT_HEADER.copy()
    header['HDUCLAS1'] = 'RESPONSE'
    header['HDUCLAS2'] = 'RAD_MAX'
    header['HDUCLAS3'] = 'POINT-LIKE'
    header['HDUCLAS4'] = 'RAD_MAX_2D'
    _add_header_cards(header, **header_cards)

    return BinTableHDU(rad_max_table, header=header, name=extname)
