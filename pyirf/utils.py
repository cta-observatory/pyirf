import numpy as np
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation

from .exceptions import MissingColumns, WrongColumnUnit


__all__ = [
    "is_scalar",
    "calculate_theta",
    "calculate_source_fov_offset",
    "check_histograms",
    "cone_solid_angle",
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


def calculate_source_fov_offset(events):
    """Calculate angular separation between true and pointing positions.

    Parameters
    ----------
    events : astropy.QTable
        Astropy Table object containing the reconstructed events information.

    Returns
    -------
    theta: astropy.units.Quantity
        Angular separation between the true and pointing positions
        in the sky.
    """
    theta = angular_separation(
        events["true_az"],
        events["true_alt"],
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
