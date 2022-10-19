"""
Utility functions for binning
"""

import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.table import QTable


#: Index returned by `calculate_bin_indices` for underflown values
UNDERFLOW_INDEX = np.iinfo(np.int64).min
#: Index returned by `calculate_bin_indices` for overflown values
OVERFLOW_INDEX = np.iinfo(np.int64).max


def bin_center(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def join_bin_lo_hi(bin_lo, bin_hi):
    """
    Function joins bins into lo and hi part,
    e.g. [0, 1, 2] and [1, 2, 4] into [0, 1, 2, 4]
    It works on multidimentional arrays as long as the binning is in the last axis

    Parameters
    ----------
    bin_lo: np.array or u.Quantity
        Lo bin edges array
    bin_hi: np.array or u.Quantity
        Hi bin edges array

    Returns
    -------
    bins: np.array of u.Quantity
        The joint bins
    """

    if np.allclose(bin_lo[...,1:], bin_hi[...,:-1], rtol=1.e-5):
        last_axis=len(bin_lo.shape)-1
        bins = np.concatenate((bin_lo, bin_hi[...,-1:]), axis=last_axis)
        return bins
    else:
        raise ValueError('Not matching bin edges')


def split_bin_lo_hi(bins):
    """
    Inverted function to join_bin_hi_lo,
    e.g. it splits [0, 1, 2, 4] into [0, 1, 2] and [1, 2, 4]

    Parameters
    ----------
    bins: np.array of u.Quantity
        The joint bins

    Returns
    -------
    bin_lo: np.array or u.Quantity
        Lo bin edges array
    bin_hi: np.array or u.Quantity
        Hi bin edges array
    """
    bin_lo=bins[...,:-1]
    bin_hi=bins[...,1:]
    return bin_lo, bin_hi

def add_overflow_bins(bins, positive=True):
    """
    Add under and overflow bins to a bin array.

    Parameters
    ----------
    bins: np.array or u.Quantity
        Bin edges array
    positive: bool
        If True, the underflow array will start at 0, if not at ``-np.inf``
    """
    lower = 0 if positive else -np.inf
    upper = np.inf

    if hasattr(bins, "unit"):
        lower *= bins.unit
        upper *= bins.unit

    if bins[0] > lower:
        bins = np.append(lower, bins)

    if bins[-1] < upper:
        bins = np.append(bins, upper)

    return bins


@u.quantity_input(e_min=u.TeV, e_max=u.TeV)
def create_bins_per_decade(e_min, e_max, bins_per_decade=5):
    """
    Create a bin array with bins equally spaced in logarithmic energy
    with ``bins_per_decade`` bins per decade.

    Parameters
    ----------
    e_min: u.Quantity[energy]
        Minimum energy, inclusive
    e_max: u.Quantity[energy]
        Maximum energy, non-inclusive
        If the endpoint exactly matches the ``n_bins_per_decade`` requirement,
        it will be included.
    n_bins_per_decade: int
        number of bins per decade

    Returns
    -------
    bins: u.Quantity[energy]
        The created bin array, will have units of e_min

    """
    unit = e_min.unit
    log_lower = np.log10(e_min.to_value(unit))
    log_upper = np.log10(e_max.to_value(unit))

    step = 1 / bins_per_decade
    # include endpoint if reasonably close
    eps = step / 10000
    bins = 10 ** np.arange(log_lower, log_upper + eps, step)
    return u.Quantity(bins, e_min.unit, copy=False)


def calculate_bin_indices(data, bins):
    """
    Calculate bin indices for given data array and bins.
    Underflow will be `UNDERFLOW_INDEX` and overflow `OVERFLOW_INDEX`.

    If the bins already include underflow / overflow bins, e.g.
    `bins[0] = -np.inf` and `bins[-1] = np.inf`, using the result of this
    function will always be a valid index into the resulting histogram.


    Parameters
    ----------
    data: ``~np.ndarray`` or ``~astropy.units.Quantity``
        Array with the data

    bins: ``~np.ndarray`` or ``~astropy.units.Quantity``
        Array or Quantity of bin edges. Must have the same unit as ``data`` if a Quantity.


    Returns
    -------
    bin_index: np.ndarray[int]
        Indices of the histogram bin the values in data belong to.
        Under- and overflown values will have values of `UNDERFLOW_INDEX`
        and `OVERFLOW_INDEX` respectively.
    """

    if hasattr(data, "unit"):
        if not hasattr(bins, "unit"):
            raise TypeError(f"If ``data`` is a Quantity, so must ``bins``, got {bins}")
        unit = data.unit
        data = data.to_value(unit)
        bins = bins.to_value(unit)

    n_bins = len(bins) - 1
    idx = np.digitize(data, bins) - 1

    underflow = (idx < 0)
    overflow = (idx >= n_bins)
    idx[underflow] = UNDERFLOW_INDEX
    idx[overflow] = OVERFLOW_INDEX
    valid = ~underflow & ~overflow
    return idx, valid


def create_histogram_table(events, bins, key="reco_energy"):
    """
    Histogram a variable from events data into an astropy table.

    Parameters
    ----------
    events : ``astropy.QTable``
        Astropy Table object containing the reconstructed events information.
    bins: ``~np.ndarray`` or ``~astropy.units.Quantity``
        Array or Quantity of bin edges.
        It must have the same units as ``data`` if a Quantity.
    key : ``string``
        Variable to histogram from the events table.

    Returns
    -------
    hist: ``astropy.QTable``
        Astropy table containg the histogram.
    """
    hist = QTable()
    hist[key + "_low"] = bins[:-1]
    hist[key + "_high"] = bins[1:]
    hist[key + "_center"] = 0.5 * (hist[key + "_low"] + hist[key + "_high"])
    hist["n"], _ = np.histogram(events[key], bins)

    # also calculate weighted number of events
    if "weight" in events.colnames:
        hist["n_weighted"], _ = np.histogram(
            events[key], bins, weights=events["weight"]
        )
    else:
        hist["n_weighted"] = hist["n"]

    # create counts per particle type, only works if there is at least 1 event
    if 'particle_type' in events.colnames and len(events) > 0:
        by_particle = events.group_by('particle_type')

        for group_key, group in zip(by_particle.groups.keys, by_particle.groups):
            particle = group_key['particle_type']

            hist["n_" + particle], _ = np.histogram(group[key], bins)

            # also calculate weighted number of events
            col = "n_" + particle
            if "weight" in events.colnames:
                hist[col + "_weighted"], _ = np.histogram(
                    group[key], bins, weights=group["weight"]
                )
            else:
                hist[col + "_weighted"] = hist[col]

    return hist


def resample_histogram1d(data, old_edges, new_edges, axis=0):
    """
    Rebinning of a histogram by interpolation along a given axis.

    Parameters
    ----------
    data : ``numpy.ndarray`` or ``astropy.units.Quantity``
        Histogram.
    old_edges : ``numpy.array`` or ``astropy.units.Quantity``
        Binning used to calculate ``data``.
        ``len(old_edges) - 1`` needs to equal the length of ``data``
        along interpolation axis (``axis``).
        If quantity, needs to be compatible to ``new_edges``.
    new_edges : ``numpy.array`` or ``astropy.units.Quantity``
        Binning of new histogram.
        If quantity, needs to be compatible to ``old_edges``.
    axis : int
        Interpolation axis.

    Returns
    -------
    ``numpy.ndarray`` or ``astropy.units.Quantity``
        Interpolated histogram with dimension according to ``data`` and ``new_edges``.
        If ``data`` is a quantity, this has the same unit.
    """

    data_unit = None
    if isinstance(data, u.Quantity):
        data_unit = data.unit
        data = data.to_value(data_unit)

    over_underflow_bin_width = old_edges[-2] - old_edges[1]
    old_edges = u.Quantity(
        np.nan_to_num(
            old_edges,
            posinf=old_edges[-2] + over_underflow_bin_width,
            neginf=old_edges[1] - over_underflow_bin_width,
        )
    )

    new_edges = u.Quantity(np.nan_to_num(new_edges))

    old_edges = old_edges.to(new_edges.unit)

    cumsum = np.insert(data.cumsum(axis=axis), 0, 0, axis=axis)

    norm = data.sum(axis=axis, keepdims=True)
    norm[norm == 0] = 1
    cumsum /= norm

    f_integral = interp1d(
        old_edges, cumsum, bounds_error=False,
        fill_value=(0, 1), kind="quadratic",
        axis=axis,
    )

    values = np.diff(f_integral(new_edges), axis=axis) * norm
    if data_unit:
        values = u.Quantity(values, unit=data_unit)

    return values
