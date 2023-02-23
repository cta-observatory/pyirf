import astropy.units as u
import numpy as np
from astropy.coordinates.angle_utilities import angular_separation

from .exceptions import MissingColumns, WrongColumnUnit

__all__ = [
    "is_scalar",
    "calculate_theta",
    "calculate_source_fov_offset",
    "check_histograms",
    "cone_solid_angle",
    "normalize_multigauss",
    "normalize_gadf3gauss",
    "multigauss_to_gadf3gauss",
    "gadf3gauss_to_multigauss",
]


def is_scalar(val):
    """Workaround that also supports astropy quantities

    Parameters
    ----------
    val : object
        Any object (value, list, etc...)

    Returns
    -------
    result: bool
        True is if input object is a scalar, False otherwise.
    """
    result = np.array(val, copy=False).shape == tuple()
    return result


@u.quantity_input(assumed_source_az=u.deg, assumed_source_alt=u.deg)
def calculate_theta(events, assumed_source_az, assumed_source_alt):
    """Calculate sky separation between assumed and reconstructed positions.

    Parameters
    ----------
    events : astropy.QTable
        Astropy Table object containing the reconstructed events information.
    assumed_source_az: astropy.units.Quantity
        Assumed Azimuth angle of the source.
    assumed_source_alt: astropy.units.Quantity
        Assumed Altitude angle of the source.

    Returns
    -------
    theta: astropy.units.Quantity
        Angular separation between the assumed and reconstructed positions
        in the sky.
    """
    theta = angular_separation(
        assumed_source_az,
        assumed_source_alt,
        events["reco_az"],
        events["reco_alt"],
    )

    return theta.to(u.deg)


def calculate_source_fov_offset(events, prefix="true"):
    """Calculate angular separation between true and pointing positions.

    Parameters
    ----------
    events : astropy.QTable
        Astropy Table object containing the reconstructed events information.

    prefix: str
        Column prefix for az / alt, can be used to calculate reco or true
        source fov offset.

    Returns
    -------
    theta: astropy.units.Quantity
        Angular separation between the true and pointing positions
        in the sky.
    """
    theta = angular_separation(
        events[f"{prefix}_az"],
        events[f"{prefix}_alt"],
        events["pointing_az"],
        events["pointing_alt"],
    )

    return theta.to(u.deg)


def check_histograms(hist1, hist2, key="reco_energy"):
    """
    Check if two histogram tables have the same binning

    Parameters
    ----------
    hist1: ``~astropy.table.Table``
        First histogram table, as created by
        ``~pyirf.binning.create_histogram_table``
    hist2: ``~astropy.table.Table``
        Second histogram table
    """

    # check binning information and add to output
    for k in ("low", "center", "high"):
        k = key + "_" + k
        if not np.all(hist1[k] == hist2[k]):
            raise ValueError(
                "Binning for signal_hist and background_hist must be equal"
            )


def cone_solid_angle(angle):
    """Calculate the solid angle of a view cone.

    Parameters
    ----------
    angle: astropy.units.Quantity or astropy.coordinates.Angle
        Opening angle of the view cone.

    Returns
    -------
    solid_angle: astropy.units.Quantity
        Solid angle of a view cone with opening angle ``angle``.

    """
    solid_angle = 2 * np.pi * (1 - np.cos(angle)) * u.sr
    return solid_angle


def check_table(table, required_columns=None, required_units=None):
    """Check a table for required columns and units.

    Parameters
    ----------
    table: astropy.table.QTable
        Table to check
    required_columns: iterable[str]
        Column names that are required to be present
    required_units: Mapping[str->astropy.units.Unit]
        Required units for columns as a Mapping from column names to units.
        Checks if the units are convertible, not if they are identical

    Raises
    ------
    MissingColumns: If any of the columns specified in ``required_columns`` or
        as keys in ``required_units are`` not present in the table.
    WrongColumnUnit: if any column has the wrong unit
    """
    if required_columns is not None:
        missing = set(required_columns) - set(table.colnames)
        if missing:
            raise MissingColumns(missing)

    if required_units is not None:
        for col, expected in required_units.items():
            if col not in table.colnames:
                raise MissingColumns(col)

            unit = table[col].unit
            if not expected.is_equivalent(unit):
                raise WrongColumnUnit(col, unit, expected)


def gadf3gauss_to_multigauss(gadf3gauss):
    """
    Convert a GADF 3Gauss format PSF [1] to the sum of three gausians

    Parameters
    ----------
    gadf3gauss: numpy.recarray
        Array containing GADF 3Gauss parameters specified in
        [1] as fields.

    Returns
    -------
    multigauss: numpy.recarray
        Array containing multi gauss parameters as fields named
        p_i and sigma_i with i = {1, 2, 3}. Result is normalized.

    References
    ----------
    .. [1] https://gamma-astro-data-formats.readthedocs.io/en/v0.3/irfs/full_enclosure/psf/psf_3gauss/index.html
    """
    multigauss = np.recarray(gadf3gauss.shape, dtype=gadf3gauss.dtype)

    multigauss.dtype.names = (
        "sigma_1",
        "sigma_2",
        "sigma_3",
        "p_1",
        "p_2",
        "p_3",
    )

    for key in ["sigma_1", "sigma_2", "sigma_3"]:
        multigauss[key] = gadf3gauss[key]

    multigauss["p_1"] = np.nan_to_num(
        (2 * (gadf3gauss["sigma_1"] * u.deg) ** 2 * gadf3gauss["scale"] * 1 / u.sr).to(
            u.dimensionless_unscaled
        )
    )

    multigauss["p_2"] = np.nan_to_num(
        (
            2
            * (gadf3gauss["sigma_2"] * u.deg) ** 2
            * gadf3gauss["scale"]
            * 1
            / u.sr
            * gadf3gauss["ampl_2"]
        )
    ).to(u.dimensionless_unscaled)

    multigauss["p_3"] = np.nan_to_num(
        (
            2
            * (gadf3gauss["sigma_3"] * u.deg) ** 2
            * gadf3gauss["scale"]
            * 1
            / u.sr
            * gadf3gauss["ampl_3"]
        )
    ).to(u.dimensionless_unscaled)

    multigauss = normalize_multigauss(multigauss)

    return multigauss.view(np.recarray)


def multigauss_to_gadf3gauss(multigauss):
    """
    Convert the sum of three gaussians to GADF 3Gauss format PSF [1]

    Parameters
    ----------
    multigauss: numpy.recarray
        Array containing multi gauss parameters as fields named
        p_i and sigma_i with i = {1, 2, 3}

    Returns
    -------
    gadf3gauss: numpy.recarray
        Array containing GADF 3Gauss parameters specified in
        [1] as fields. Result is normalized.

    References
    ----------
    .. [1] https://gamma-astro-data-formats.readthedocs.io/en/v0.3/irfs/full_enclosure/psf/psf_3gauss/index.html
    """
    multigauss = normalize_multigauss(multigauss)

    gadf3gauss = np.recarray(multigauss.shape, dtype=multigauss.dtype)

    gadf3gauss.dtype.names = (
        "scale",
        "ampl_2",
        "ampl_3",
        "sigma_1",
        "sigma_2",
        "sigma_3",
    )

    for key in ["sigma_1", "sigma_2", "sigma_3"]:
        gadf3gauss[key] = multigauss[key]

    scale = (multigauss["p_1"] / (2 * (multigauss["sigma_1"] * u.deg) ** 2)).to(
        1 / u.sr
    )

    gadf3gauss["scale"] = np.nan_to_num(scale)

    gadf3gauss["ampl_2"] = np.nan_to_num(
        (multigauss["p_2"] / (2 * scale * (multigauss["sigma_2"] * u.deg) ** 2))
    ).to(u.dimensionless_unscaled)

    gadf3gauss["ampl_3"] = np.nan_to_num(
        (multigauss["p_3"] / (2 * scale * (multigauss["sigma_3"] * u.deg) ** 2))
    ).to(u.dimensionless_unscaled)

    return gadf3gauss.view(np.recarray)


def normalize_multigauss(multigauss):
    """Normalizes the sum of three gaussians centered at zero

    Parameters
    ----------
    multigauss: numpy.recarray
        Array containing multi gauss parameters as fields named
        p_i and sigma_i with i = {1, 2, 3}

    Returns
    -------
    normalized_multigauss: numpy.recarray
        Mutligauss with normalized parameters p_i
    """

    normed_gauss = multigauss.copy()

    norm = np.nansum([normed_gauss[p] for p in ["p_1", "p_2", "p_3"]])

    for p in ["p_1", "p_2", "p_3"]:
        normed_gauss[p] = np.nan_to_num(normed_gauss[p] / norm)

    return normed_gauss.view(np.recarray)


def normalize_gadf3gauss(gadf3gauss):
    """Normalizes an energy dependent 3Gauss PSF in GADF format [1]

    Parameters
    ----------
    gadf3gauss: numpy.recarray
        Array containing GADF 3Gauss parameters specified in
        [1] as fields

    Returns
    -------
    normalized_gadf3gauss: numpy.recarray
        GADF 3Gauss with normalized parameter values scale, ampl_2 and ampl_3

    References
    ----------
    .. [1] https://gamma-astro-data-formats.readthedocs.io/en/v0.3/irfs/full_enclosure/psf/psf_3gauss/index.html
    """
    multigauss = gadf3gauss_to_multigauss(gadf3gauss)

    normed_multigauss = normalize_multigauss(multigauss)

    normed_gadf3gauss = multigauss_to_gadf3gauss(normed_multigauss)

    return normed_gadf3gauss.view(np.recarray)
