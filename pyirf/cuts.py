import operator

import numpy as np
from astropy.table import Table

from .binning import calculate_bin_indices
from .utils import is_scalar


def calculate_percentile_cut(
    values,
    bin_values,
    bins,
    fill_value,
    percentile=68,
    min_value=None,
    max_value=None,
):
    '''
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
        Value inserted for empty bins
    percentile: float
        The percentile to calculate in each bin as a percentage,
        i.e. 0 <= percentile <= 100.
    min_value: float or quantity or None
        If given, cuts smaller than this value are replaced with ``min_value``
    max_value: float or quantity or None
        If given, cuts larger than this value are replaced with ``max_value``
    '''

    # create a table to make use of groupby operations
    table = Table({'values': values, 'bin_values': bin_values}, copy=False)

    table['bin_index'] = calculate_bin_indices(
        table['bin_values'].quantity, bins
    )

    cut_table = Table()
    cut_table['low'] = bins[:-1]
    cut_table['high'] = bins[1:]
    cut_table['cut'] = fill_value

    # use groupby operations to calculate the percentile in each bin
    by_bin = table.group_by('bin_index')

    # fill only the non-empty bins
    cut_table['cut'][by_bin.groups.keys['bin_index']] = (
        by_bin['values']
        .groups.aggregate(lambda g: np.percentile(g, percentile))
        .quantity.to_value(cut_table['cut'].unit)
    )

    if min_value is not None:
        invalid = cut_table['cut'] < min_value
        cut_table['cut'] = np.where(invalid, min_value, cut_table['cut'])

    if max_value is not None:
        invalid = cut_table['cut'] > max_value
        cut_table['cut'] = np.where(invalid, max_value, cut_table['cut'])

    return cut_table


def evaluate_binned_cut(values, bin_values, cut_table, op):
    '''
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
            `low`: lower edges of the bins
            `high`: upper edges of the bins,
            `cut`: cut value
    op: binary operator function
        A function taking two arguments, comparing element-wise and
        returning an array of booleans.
    '''
    bins = np.append(cut_table['low'].quantity, cut_table['high'].quantity[-1])
    bin_index = calculate_bin_indices(bin_values, bins)
    return op(values, cut_table['cut'][bin_index].quantity)


def is_selected(events, cut_definition, bin_index=None):
    '''
    Retun a boolean mask, if the given ``events`` survive the cuts defined
    in ``cut_definition``.
    This function supports bin-wise cuts when given the ``bin_index`` argument.

    Parameters
    ----------
    events: ``~astropy.table.QTable``
        events table
    cut_definition: dict
        A dict describing the cuts to make.
        The keys are column names in ``events`` to which a cut should be applied.
        The values must be dictionaries with the key ``'operator'`` containing the
        name of the binary comparison operator to use and the key ``'cut_values'``,
        which is either a single number of quantity or a Quantity or an array
        containing the cut value for each bin.
    bin_index: np.ndarray[int]
        Bin index for each event in the ``events`` table, only needed if
        bin-wise cut values are used.

    Returns
    -------
    selected: np.ndarray[bool]
        Boolean mask if an event survived the specified cuts.
    '''
    mask = np.ones(len(events), dtype=np.bool)

    for key, definition in cut_definition.items():

        op = getattr(operator, definition['operator'])

        # for a single number, just use the value

        if is_scalar(definition['cut_values']):
            cut_value = definition['cut_values']

        # if it is an array, it is per bin, so we get the correct
        # cut value for each event
        else:
            if bin_index is None:
                raise ValueError(
                    'You need to provide `bin_index` if cut_values are per bin'
                )
            cut_value = np.asanyarray(definition['cut_values'])[bin_index]

        mask &= op(events[key], cut_value)

    return mask
