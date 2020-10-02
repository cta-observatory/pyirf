from astropy.table import QTable
import astropy.units as u
from astropy.io.fits import Header, BinTableHDU
import numpy as np
from astropy.time import Time

from ..version import __version__


__all__ = [
    "create_aeff2d_hdu",
    "create_energy_dispersion_hdu",
    "create_psf_table_hdu",
    "create_rad_max_hdu",
]


DEFAULT_HEADER = Header()
DEFAULT_HEADER["CREATOR"] = f"pyirf v{__version__}"
# fmt: off
DEFAULT_HEADER["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
DEFAULT_HEADER["HDUVERS"] = "0.2"
DEFAULT_HEADER["HDUCLASS"] = "GADF"


def _add_header_cards(header, **header_cards):
    for k, v in header_cards.items():
        header[k] = v


@u.quantity_input(
    effective_area=u.m ** 2, true_energy_bins=u.TeV, fov_offset_bins=u.deg
)
def create_aeff2d_hdu(
    effective_area,
    true_energy_bins,
    fov_offset_bins,
    extname="EFFECTIVE AREA",
    point_like=True,
    **header_cards,
):
    """
    Create a fits binary table HDU in GADF format for effective area.
    See the specification at
    https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html

    Parameters
    ----------
    effective_area: astropy.units.Quantity[area]
        Effective area array, must have shape (n_energy_bins, n_fov_offset_bins)
    true_energy_bins: astropy.units.Quantity[energy]
        Bin edges in true energy
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
        For Point-Like IRFs, only giving a single bin is appropriate.
    point_like: bool
        If the provided effective area was calculated after applying a direction cut,
        pass ``True``, else ``False`` for a full-enclosure effective area.
    extname: str
        Name for BinTableHDU
    **header_cards
        Additional metadata to add to the header, use this to set e.g. TELESCOP or
        INSTRUME.
    """
    aeff = QTable()
    aeff["ENERG_LO"] = u.Quantity(true_energy_bins[:-1], ndmin=2).to(u.TeV)
    aeff["ENERG_HI"] = u.Quantity(true_energy_bins[1:], ndmin=2).to(u.TeV)
    aeff["THETA_LO"] = u.Quantity(fov_offset_bins[:-1], ndmin=2).to(u.deg)
    aeff["THETA_HI"] = u.Quantity(fov_offset_bins[1:], ndmin=2).to(u.deg)
    # transpose because FITS uses opposite dimension order than numpy
    aeff["EFFAREA"] = effective_area.T[np.newaxis, ...].to(u.m ** 2)

    # required header keywords
    header = DEFAULT_HEADER.copy()
    header["HDUCLAS1"] = "RESPONSE"
    header["HDUCLAS2"] = "EFF_AREA"
    header["HDUCLAS3"] = "POINT-LIKE" if point_like else "FULL-ENCLOSURE"
    header["HDUCLAS4"] = "AEFF_2D"
    header["DATE"] = Time.now().utc.iso
    _add_header_cards(header, **header_cards)

    return BinTableHDU(aeff, header=header, name=extname)


@u.quantity_input(
    psf=u.sr ** -1,
    true_energy_bins=u.TeV,
    fov_offset_bins=u.deg,
    source_offset_bins=u.deg,
)
def create_psf_table_hdu(
    psf,
    true_energy_bins,
    source_offset_bins,
    fov_offset_bins,
    extname="PSF",
    **header_cards,
):
    """
    Create a fits binary table HDU in GADF format for the PSF table.
    See the specification at
    https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/psf/psf_table/index.html

    Parameters
    ----------
    psf: astropy.units.Quantity[(solid angle)^-1]
        Point spread function array, must have shape
        (n_energy_bins, n_fov_offset_bins, n_source_offset_bins)
    true_energy_bins: astropy.units.Quantity[energy]
        Bin edges in true energy
    source_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the source offset.
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
        For Point-Like IRFs, only giving a single bin is appropriate.
    extname: str
        Name for BinTableHDU
    **header_cards
        Additional metadata to add to the header, use this to set e.g. TELESCOP or
        INSTRUME.
    """

    psf = QTable(
        {
            "ENERG_LO": u.Quantity(true_energy_bins[:-1], ndmin=2).to(u.TeV),
            "ENERG_HI": u.Quantity(true_energy_bins[1:], ndmin=2).to(u.TeV),
            "THETA_LO": u.Quantity(fov_offset_bins[:-1], ndmin=2).to(u.deg),
            "THETA_HI": u.Quantity(fov_offset_bins[1:], ndmin=2).to(u.deg),
            "RAD_LO": u.Quantity(source_offset_bins[:-1], ndmin=2).to(u.deg),
            "RAD_HI": u.Quantity(source_offset_bins[1:], ndmin=2).to(u.deg),
            # transpose as FITS uses opposite dimension order
            "RPSF": psf.T[np.newaxis, ...].to(1 / u.sr),
        }
    )

    # required header keywords
    header = DEFAULT_HEADER.copy()
    header["HDUCLAS1"] = "RESPONSE"
    header["HDUCLAS2"] = "PSF"
    header["HDUCLAS3"] = "FULL-ENCLOSURE"
    header["HDUCLAS4"] = "PSF_TABLE"
    header["DATE"] = Time.now().utc.iso
    _add_header_cards(header, **header_cards)

    return BinTableHDU(psf, header=header, name=extname)


@u.quantity_input(
    true_energy_bins=u.TeV, fov_offset_bins=u.deg,
)
def create_energy_dispersion_hdu(
    energy_dispersion,
    true_energy_bins,
    migration_bins,
    fov_offset_bins,
    point_like=True,
    extname="EDISP",
    **header_cards,
):
    """
    Create a fits binary table HDU in GADF format for the energy dispersion.
    See the specification at
    https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html

    Parameters
    ----------
    energy_dispersion: numpy.ndarray
        Energy dispersion array, must have shape
        (n_energy_bins, n_migra_bins, n_source_offset_bins)
    true_energy_bins: astropy.units.Quantity[energy]
        Bin edges in true energy
    migration_bins: numpy.ndarray
        Bin edges for the relative energy migration (``reco_energy / true_energy``)
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
        For Point-Like IRFs, only giving a single bin is appropriate.
    point_like: bool
        If the provided effective area was calculated after applying a direction cut,
        pass ``True``, else ``False`` for a full-enclosure effective area.
    extname: str
        Name for BinTableHDU
    **header_cards
        Additional metadata to add to the header, use this to set e.g. TELESCOP or
        INSTRUME.
    """

    edisp = QTable(
        {
            "ENERG_LO": u.Quantity(true_energy_bins[:-1], ndmin=2).to(u.TeV),
            "ENERG_HI": u.Quantity(true_energy_bins[1:], ndmin=2).to(u.TeV),
            "MIGRA_LO": u.Quantity(migration_bins[:-1], ndmin=2).to(u.one),
            "MIGRA_HI": u.Quantity(migration_bins[1:], ndmin=2).to(u.one),
            "THETA_LO": u.Quantity(fov_offset_bins[:-1], ndmin=2).to(u.deg),
            "THETA_HI": u.Quantity(fov_offset_bins[1:], ndmin=2).to(u.deg),
            # transpose as FITS uses opposite dimension order
            "MATRIX": u.Quantity(energy_dispersion.T[np.newaxis, ...]).to(u.one),
        }
    )

    # required header keywords
    header = DEFAULT_HEADER.copy()
    header["HDUCLAS1"] = "RESPONSE"
    header["HDUCLAS2"] = "EDISP"
    header["HDUCLAS3"] = "POINT-LIKE" if point_like else "FULL-ENCLOSURE"
    header["HDUCLAS4"] = "EDISP_2D"
    header["DATE"] = Time.now().utc.iso
    _add_header_cards(header, **header_cards)

    return BinTableHDU(edisp, header=header, name=extname)


#: Unit to store background rate in GADF format
#:
#: see https://github.com/open-gamma-ray-astro/gamma-astro-data-formats/issues/153
#: for a discussion on why this is MeV not TeV as everywhere else
GADF_BACKGROUND_UNIT = u.Unit("MeV-1 s-1 sr-1")


@u.quantity_input(
    background=GADF_BACKGROUND_UNIT, reco_energy_bins=u.TeV, fov_offset_bins=u.deg,
)
def create_background_2d_hdu(
    background_2d,
    reco_energy_bins,
    fov_offset_bins,
    extname="BACKGROUND",
    **header_cards,
):
    """
    Create a fits binary table HDU in GADF format for the background 2d table.
    See the specification at
    https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/bkg/index.html#bkg-2d

    Parameters
    ----------
    background_2d: astropy.units.Quantity[(MeV s sr)^-1]
        Background rate, must have shape
        (n_energy_bins, n_fov_offset_bins)
    reco_energy_bins: astropy.units.Quantity[energy]
        Bin edges in reconstructed energy
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
    extname: str
        Name for BinTableHDU
    **header_cards
        Additional metadata to add to the header, use this to set e.g. TELESCOP or
        INSTRUME.
    """

    bkg = QTable(
        {
            "ENERG_LO": u.Quantity(reco_energy_bins[:-1], ndmin=2).to(u.TeV),
            "ENERG_HI": u.Quantity(reco_energy_bins[1:], ndmin=2).to(u.TeV),
            "THETA_LO": u.Quantity(fov_offset_bins[:-1], ndmin=2).to(u.deg),
            "THETA_HI": u.Quantity(fov_offset_bins[1:], ndmin=2).to(u.deg),
            # transpose as FITS uses opposite dimension order
            "BKG": background_2d.T[np.newaxis, ...].to(GADF_BACKGROUND_UNIT),
        }
    )

    # required header keywords
    header = DEFAULT_HEADER.copy()
    header["HDUCLAS1"] = "RESPONSE"
    header["HDUCLAS2"] = "BKG"
    header["HDUCLAS3"] = "FULL-ENCLOSURE"
    header["HDUCLAS4"] = "BKG_2D"
    header["DATE"] = Time.now().utc.iso
    _add_header_cards(header, **header_cards)

    return BinTableHDU(bkg, header=header, name=extname)


@u.quantity_input(
    rad_max=u.deg,
    reco_energy_bins=u.TeV,
    fov_offset_bins=u.deg,
)
def create_rad_max_hdu(
    rad_max,
    reco_energy_bins,
    fov_offset_bins,
    point_like=True,
    extname="RAD_MAX",
    **header_cards,
):
    """
    Create a fits binary table HDU in GADF format for the directional cut.
    See the specification at
    https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html

    Parameters
    ----------
    rad_max: astropy.units.Quantity[angle]
        Array of the directional (theta) cut.
        Must have shape (n_reco_energy_bins, n_fov_offset_bins)
    reco_energy_bins: astropy.units.Quantity[energy]
        Bin edges in reconstructed energy
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
        For Point-Like IRFs, only giving a single bin is appropriate.
    extname: str
        Name for BinTableHDU
    **header_cards
        Additional metadata to add to the header, use this to set e.g. TELESCOP or
        INSTRUME.
    """
    rad_max_table = QTable(
        {
            "ENERG_LO": u.Quantity(reco_energy_bins[:-1], ndmin=2).to(u.TeV),
            "ENERG_HI": u.Quantity(reco_energy_bins[1:], ndmin=2).to(u.TeV),
            "THETA_LO": u.Quantity(fov_offset_bins[:-1], ndmin=2).to(u.deg),
            "THETA_HI": u.Quantity(fov_offset_bins[1:], ndmin=2).to(u.deg),
            # transpose as FITS uses opposite dimension order
            "RAD_MAX": rad_max.T[np.newaxis, ...].to(u.deg),
        }
    )

    # required header keywords
    header = DEFAULT_HEADER.copy()
    header["HDUCLAS1"] = "RESPONSE"
    header["HDUCLAS2"] = "RAD_MAX"
    header["HDUCLAS3"] = "POINT-LIKE"
    header["HDUCLAS4"] = "RAD_MAX_2D"
    header["DATE"] = Time.now().utc.iso
    _add_header_cards(header, **header_cards)

    return BinTableHDU(rad_max_table, header=header, name=extname)
