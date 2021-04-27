from astropy.table import QTable
import astropy.units as u
from astropy.io.fits import Header, BinTableHDU
import numpy as np
from astropy.time import Time
import pyirf.binning as binning
from pyirf.cuts import compare_irf_cuts

from ..version import __version__


__all__ = [
    "create_aeff2d_hdu",
    "create_energy_dispersion_hdu",
    "create_psf_table_hdu",
    "create_rad_max_hdu",
    "compare_irf_cuts_in_files",
    "read_fits_bins_lo_hi",
    "read_irf_grid",
    "read_aeff2d_hdu",
    "read_energy_dispersion_hdu",
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
    aeff["ENERG_LO"], aeff["ENERG_HI"] = binning.split_bin_lo_hi(true_energy_bins[np.newaxis, :].to(u.TeV))
    aeff["THETA_LO"], aeff["THETA_HI"] = binning.split_bin_lo_hi(fov_offset_bins[np.newaxis, :].to(u.deg))
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
    source_offset_bins=u.deg,
    fov_offset_bins=u.deg,
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

    psf_ = QTable()
    psf_["ENERG_LO"], psf_["ENERG_HI"] = binning.split_bin_lo_hi(true_energy_bins[np.newaxis, :].to(u.TeV))
    psf_["THETA_LO"], psf_["THETA_HI"] = binning.split_bin_lo_hi(fov_offset_bins[np.newaxis, :].to(u.deg))
    psf_["RAD_LO"], psf_["RAD_HI"] = binning.split_bin_lo_hi(source_offset_bins[np.newaxis, :].to(u.deg))
    # transpose as FITS uses opposite dimension order
    psf_["RPSF"] = psf.T[np.newaxis, ...].to(1 / u.sr)

    # required header keywords
    header = DEFAULT_HEADER.copy()
    header["HDUCLAS1"] = "RESPONSE"
    header["HDUCLAS2"] = "PSF"
    header["HDUCLAS3"] = "FULL-ENCLOSURE"
    header["HDUCLAS4"] = "PSF_TABLE"
    header["DATE"] = Time.now().utc.iso
    _add_header_cards(header, **header_cards)

    return BinTableHDU(psf_, header=header, name=extname)


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

    edisp = QTable()
    edisp["ENERG_LO"], edisp["ENERG_HI"] = binning.split_bin_lo_hi(true_energy_bins[np.newaxis, :].to(u.TeV))
    edisp["MIGRA_LO"], edisp["MIGRA_HI"] = binning.split_bin_lo_hi(migration_bins[np.newaxis, :])
    edisp["THETA_LO"], edisp["THETA_HI"] = binning.split_bin_lo_hi(fov_offset_bins[np.newaxis, :].to(u.deg))
    # transpose as FITS uses opposite dimension order
    edisp["MATRIX"] = u.Quantity(energy_dispersion.T[np.newaxis, ...]).to(u.one)
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

    bkg = QTable()
    bkg["ENERG_LO"], bkg["ENERG_HI"] = binning.split_bin_lo_hi(reco_energy_bins[np.newaxis, :].to(u.TeV))
    bkg["THETA_LO"], bkg["THETA_HI"] = binning.split_bin_lo_hi(fov_offset_bins[np.newaxis, :].to(u.deg))
    # transpose as FITS uses opposite dimension order
    bkg["BKG"] = background_2d.T[np.newaxis, ...].to(GADF_BACKGROUND_UNIT)

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

    rad_max_table = QTable()
    rad_max_table["ENERG_LO"], rad_max_table["ENERG_HI"] = binning.split_bin_lo_hi(reco_energy_bins[np.newaxis, :].to(u.TeV))
    rad_max_table["THETA_LO"], rad_max_table["THETA_HI"] = binning.split_bin_lo_hi(fov_offset_bins[np.newaxis, :].to(u.deg))
    # transpose as FITS uses opposite dimension order
    rad_max_table["RAD_MAX"] = rad_max.T[np.newaxis, ...].to(u.deg)

    # required header keywords
    header = DEFAULT_HEADER.copy()
    header["HDUCLAS1"] = "RESPONSE"
    header["HDUCLAS2"] = "RAD_MAX"
    header["HDUCLAS3"] = "POINT-LIKE"
    header["HDUCLAS4"] = "RAD_MAX_2D"
    header["DATE"] = Time.now().utc.iso
    _add_header_cards(header, **header_cards)

    return BinTableHDU(rad_max_table, header=header, name=extname)


def compare_irf_cuts_in_files(files, extname='THETA_CUTS'):
    """
    Reads in a list of IRF files and checks if the same cuts have been applied in all of them

    Parameters
    ----------
    files: list of strings
        files to be read
    extname: string
        name of the extension with cut values to read the data from in fits file
    Returns
    -------
    match: Boolean
        if the cuts are the same in all the files
    """
    cuts = read_irf_cuts(files, extname=extname)
    return compare_irf_cuts(cuts)


def read_irf_cuts(files, extname='THETA_CUTS'):
    """
    Reads in a list of IRF files, extracts cuts from them and returns them as a list

    Parameters
    ----------
    files: list of strings
        files to be read
    extname: string
        name of the extension with cut values to read the data from in fits file
    Returns
    -------
    cuts: list
        list of cuts
    """
    if isinstance(files, str):
        files = [files]

    cuts = list()
    for file_name in files:
        table = QTable.read(file_name, hdu=extname)
        cuts.append(table)

    # if the function is run on single file do not need the first axis dimension
    if len(files) == 1:
        cuts = cuts[0]

    return cuts


def read_fits_bins_lo_hi(file_name, extname, tags):
    """
    Reads from a fits file two arrays of tag_LO and tag_HI.
    If more then one file is given it checks that all of them are consistent before returning the value

    Parameters
    ----------
    file_name: string or list of string
        file to be read, if a list of
    extname: string
        name of the extension to read the data from in fits file
    tags: list of string
        names of the field in the extension to extract, _LO and _HI will be added

    Returns
    -------
    bins: list of astropy.units.Quantity
        list of ntuples (LO, HI) of bins (with size of extnames)
    """

    # to allow the function work on either single file or a list of files convert a single string into a list
    if isinstance(file_name, str):
        file_name = [file_name]

    # same to allow to run over single tag
    if isinstance(tags, str):
        tags = [tags]

    tags_lo_hi = [(tag + '_LO', tag + '_HI') for tag in tags]

    old_table = None
    bins = list()
    for this_file in file_name:
        table = QTable.read(this_file, hdu=extname)
        for tag_lo, tag_hi in tags_lo_hi:
            if old_table is not None:
                if not old_table[tag_lo][0].shape == table[tag_lo][0].shape:
                    raise ValueError('Non matching bins in ' + extname)
                if not ((old_table[tag_lo][0] == table[tag_lo][0]).all() and (old_table[tag_hi][0] == table[tag_hi][0]).all()):
                    raise ValueError('Non matching bins in ' + extname)
            else:
                bins.append([table[tag_lo][0], table[tag_hi][0]])
        old_table = table

    return bins


def read_irf_grid(files, extname, field_name):
    """
    Reads in a grid of IRFs for a bunch of different parameters and stores them in lists

    Parameters
    ----------
    files: string or list of strings
        files to be read
    extname: string
        name of the extension to read the data from in fits file
    field_name: string
        name of the field in the extension to extract

    Returns
    -------
    irfs_all: np.array
        array of IRFs, first axis specify the file number(if more then one is given),
        second axis is the offset angle, rest of the axes are IRF-specific
    theta_bins: astropy.units.Quantity[angle]
        theta bins
    """

    # to allow the function work on either single file or a list of files convert a single string into a list
    if isinstance(files, str):
        files = [files]

    irfs_all = list()

    for this_file in files:
        # [0] because there the IRFs are written as a single row of the table
        # we transpose because of a different axis sequence in fits file and in pyirf
        # and multiplication by u.one is to have Quantity also for migration matrix
        irfs_all.append(QTable.read(this_file, hdu=extname)[field_name][0].T * u.one)

    # convert the list to a multidimentional table
    irfs_all = np.array(irfs_all) * irfs_all[0].unit

    # if the function is run on single file do not need the first axis dimension
    if len(files) == 1:
        irfs_all = irfs_all[0]

    return irfs_all


def read_aeff2d_hdu(file_name, extname="EFFECTIVE AREA"):
    """
    Reads effective area from FITS file

    Parameters
    ----------
    file_name: string or list of strings
        file(s) to be read
    extname:
        Name for BinTableHDU

    Returns
    -------
    effective_area: astropy.units.Quantity[area]
        Effective area array of shape (n_energy_bins, n_fov_offset_bins) or (n_files, n_energy_bins, n_fov_offset_bins)
    true_energy_bins: astropy.units.Quantity[energy]
        Bin edges in true energy
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
    """

    field_name = "EFFAREA"
    effective_area = read_irf_grid(file_name, extname, field_name)
    true_energy_bins, fov_offset_bins = read_fits_bins_lo_hi(file_name, extname, ["ENERG", "THETA"])
    true_energy_bins = binning.join_bin_lo_hi(*true_energy_bins)
    fov_offset_bins = binning.join_bin_lo_hi(*fov_offset_bins)
    return effective_area, true_energy_bins, fov_offset_bins


def read_energy_dispersion_hdu(file_name, extname="EDISP"):
    """
    Reads energy dispersion table from a FITS file

    Parameters
    ----------
    file_name: string or list of strings
        file(s) to be read
    extname:
        Name for BinTableHDU

    Returns
    -------
    energy_dispersion: numpy.ndarray
        Energy dispersion array, with shape
        (n_energy_bins, n_migra_bins, n_source_offset_bins)
        or (n_files, n_energy_bins, n_migra_bins, n_source_offset_bins)
    true_energy_bins: astropy.units.Quantity[energy]
        Bin edges in true energy
    migration_bins: numpy.ndarray
        Bin edges for the relative energy migration (``reco_energy / true_energy``)
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
    """

    field_name = "MATRIX"
    energy_dispersion = read_irf_grid(file_name, extname, field_name)
    true_energy_bins, migration_bins, fov_offset_bins = read_fits_bins_lo_hi(file_name, extname, ["ENERG", "MIGRA", "THETA"])
    true_energy_bins = binning.join_bin_lo_hi(*true_energy_bins)
    migration_bins = binning.join_bin_lo_hi(*migration_bins)
    fov_offset_bins = binning.join_bin_lo_hi(*fov_offset_bins)

    return energy_dispersion, true_energy_bins, migration_bins, fov_offset_bins
