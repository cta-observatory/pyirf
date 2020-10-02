import numpy as np
from astropy.table import Table, QTable
from scipy.ndimage.filters import gaussian_filter1d
import astropy.units as u

from .binning import calculate_bin_indices, bin_center

__all__ = [
    'calculate_percentile_cut',
    'evaluate_binned_cut',
]


def calculate_percentile_cut(
    values, bin_values, bins, fill_value, percentile=68, min_value=None, max_value=None,
    smoothing=None, min_events=10,
):
    """
    Calculate cuts as the percentile of a given quantity in bins of another
    quantity.

    Parameters
    ----------
    values: ``~numpy.ndarray`` or ``~astropy.units.Quantity``
        The values for which the cut should be calculated
    bin_values: ``~numpy.ndarray`` or ``~astropy.units.Quantity``
        The values used to sort the ``values`` into bins
    bins: ``~numpy.ndarray`` or ``~astropy.units.Quantity``
        Bin edges
    fill_value: float or quantity
        Value for bins with less than ``min_events``,
        must have same unit as values
    percentile: float
        The percentile to calculate in each bin as a percentage,
        i.e. 0 <= percentile <= 100.
    min_value: float or quantity or None
        If given, cuts smaller than this value are replaced with ``min_value``
    max_value: float or quantity or None
        If given, cuts larger than this value are replaced with ``max_value``
    smoothing: float or None
        If given, apply a gaussian filter of width ``sigma`` in terms
        of bins.
    min_events: int
        Bins with less events than this number are replaced with ``fill_value``
    """
    # create a table to make use of groupby operations
    table = Table({"values": values, "bin_values": bin_values}, copy=False)
    unit = table["values"].unit

    # make sure units match
    if unit is not None:
        fill_value = u.Quantity(fill_value).to(unit)

    table["bin_index"] = calculate_bin_indices(table["bin_values"].quantity, bins)

    cut_table = QTable()
    cut_table["low"] = bins[:-1]
    cut_table["high"] = bins[1:]
    cut_table["center"] = bin_center(bins)
    cut_table["cut"] = fill_value

    # use groupby operations to calculate the percentile in each bin
    by_bin = table.group_by("bin_index")

    # fill only the non-empty bins
    cut = by_bin["values"].groups.aggregate(lambda g: np.percentile(g, percentile))
    if unit is not None:
        cut = cut.quantity.to(unit)

    # replace bins with too few events with fill_value
    n_events = by_bin["values"].groups.aggregate(len)
    cut[n_events < min_events] = fill_value

    # assign to full table, index lookup needed in case of empty bins
    cut_table["cut"][by_bin.groups.keys["bin_index"]] = cut

    if min_value is not None:
        if unit is not None:
            min_value = u.Quantity(min_value).to(unit)
        invalid = cut_table["cut"] < min_value
        cut_table["cut"] = np.where(invalid, min_value, cut_table["cut"])

    if max_value is not None:
        if unit is not None:
            max_value = u.Quantity(max_value).to(unit)
        invalid = cut_table["cut"] > max_value
        cut_table["cut"] = np.where(invalid, max_value, cut_table["cut"])

    if smoothing is not None:
        if unit is not None:
            cut = cut_table['cut'].to_value(unit)

        cut = gaussian_filter1d(cut, smoothing, mode='nearest')

        if unit is not None:
            cut = u.Quantity(cut, unit=unit, copy=False)

        cut_table['cut'] = cut

    return cut_table


def evaluate_binned_cut(values, bin_values, cut_table, op):
    """
    Evaluate a binned cut as defined in cut_table on given events

    Parameters
    ----------
    values: ``~numpy.ndarray`` or ``~astropy.units.Quantity``
        The values on which the cut should be evaluated
    bin_values: ``~numpy.ndarray`` or ``~astropy.units.Quantity``
        The values used to sort the ``values`` into bins
    cut_table: ``~astropy.table.Table``
        A table describing the binned cuts, e.g. as created by
        ``~pyirf.cuts.calculate_percentile_cut``.
        Required columns:
        - `low`: lower edges of the bins
        - `high`: upper edges of the bins,
        - `cut`: cut value
    op: callable(a, b) -> bool
        A function taking two arguments, comparing element-wise and
        returning an array of booleans.
        Must support vectorized application.
    """
    if not isinstance(cut_table, QTable):
        raise ValueError('cut_table needs to be an astropy.table.QTable')

    bins = np.append(cut_table["low"], cut_table["high"][-1])
    bin_index = calculate_bin_indices(bin_values, bins)
    return op(values, cut_table["cut"][bin_index])
