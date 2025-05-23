import numpy as np
from astropy.table import Table, QTable
from scipy.ndimage import gaussian_filter1d
import astropy.units as u

from .binning import calculate_bin_indices, bin_center
from .compat import COPY_IF_NEEDED

__all__ = [
    "calculate_percentile_cut",
    "evaluate_binned_cut",
    "evaluate_binned_cut_by_index",
    "compare_irf_cuts",
]


def calculate_percentile_cut(
    values,
    bin_values,
    bins,
    fill_value,
    percentile=68,
    min_value=None,
    max_value=None,
    smoothing=None,
    min_events=10,
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
    edges: ``~numpy.ndarray`` or ``~astropy.units.Quantity``
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
    # we use a normal table here to avoid astropy/astropy#13840
    table = Table({"values": values}, copy=COPY_IF_NEEDED)
    unit = table["values"].unit

    # make sure units match
    if unit is not None:
        fill_value = u.Quantity(fill_value).to(unit)

        if min_value is not None:
            min_value = u.Quantity(min_value).to_value(unit)

        if max_value is not None:
            max_value = u.Quantity(max_value).to_value(unit)

    bin_index, valid = calculate_bin_indices(bin_values, bins)
    by_bin = table[valid].group_by(bin_index[valid])

    n_bins = len(bins) - 1
    cut_table = QTable()
    cut_table["low"] = bins[:-1]
    cut_table["high"] = bins[1:]
    cut_table["center"] = bin_center(bins)
    cut_table["n_events"] = 0

    unit = None
    if hasattr(fill_value, 'unit'):
        unit = fill_value.unit
        fill_value = fill_value.value

    percentile = np.asanyarray(percentile)
    if percentile.shape == ():
        cut_table["cut"] = np.asanyarray(fill_value, values.dtype)
    else:
        cut_table["cut"] = np.full((n_bins, len(percentile)), fill_value, dtype=values.dtype)

    if unit is not None:
        cut_table["cut"].unit = unit

    for bin_idx, group in zip(by_bin.groups.keys, by_bin.groups):
        # replace bins with too few events with fill_value
        n_events = len(group)
        cut_table["n_events"][bin_idx] = n_events

        if n_events < min_events:
            cut_table["cut"].value[bin_idx] = fill_value
        else:
            value = np.nanpercentile(group["values"], percentile)
            if min_value is not None or max_value is not None:
                value = np.clip(value, min_value, max_value)

            cut_table["cut"].value[bin_idx] = value

    if smoothing is not None:
        cut_table["cut"].value[:] = gaussian_filter1d(
            cut_table["cut"].value,
            smoothing,
            mode="nearest",
        )

    return cut_table


def evaluate_binned_cut_by_index(values, bin_index, valid, cut_table, op):
    """
    Evaluate a binned cut as defined in cut_table with pre-computed bin index.

    This is an optimization over evaluating `evaluate_binned_cut`
    multiple times with the same values to prevent re-computation of the index.


    Parameters
    ----------
    values: ``~numpy.ndarray`` or ``~astropy.units.Quantity``
        The values on which the cut should be evaluated
    bin_index: ``~numpy.ndarray``
        The index into ``cut_table`` corresponding to the entries in ``values``.        
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

    result = np.zeros(len(bin_index), dtype=bool)
    result[valid] = op(values[valid], cut_table["cut"][bin_index[valid]])
    return result


def evaluate_binned_cut(values, bin_values, cut_table, op):
    """
    Evaluate a binned cut as defined in cut_table on given events.

    Events with bin_values outside the bin edges defined in cut table
    will be set to False.

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


    Returns
    -------
    result: np.ndarray[bool]
        A mask for each entry in ``values`` indicating if the event
        passes the bin specific cut given in cut table.
    """
    if not isinstance(cut_table, QTable):
        raise ValueError("cut_table needs to be an astropy.table.QTable")

    bins = np.append(cut_table["low"], cut_table["high"][-1])
    bin_index, valid = calculate_bin_indices(bin_values, bins)
    return evaluate_binned_cut_by_index(values, bin_index, valid, cut_table, op)


def compare_irf_cuts(cuts):
    """
    checks if the same cuts have been applied in all of them

    Parameters
    ----------
    cuts: list of QTables
        list of cuts each entry in the list correspond to one set of IRFs
    Returns
    -------
    match: Boolean
        if the cuts are the same in all the files
    """
    for i in range(len(cuts) - 1):
        if (cuts[i] != cuts[i + 1]).any():
            return False
    return True
