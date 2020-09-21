import operator
import numpy as np


def is_scalar(val):
    '''Workaround that also supports astropy quantities'''
    return np.array(val, copy=False).shape == tuple()


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
