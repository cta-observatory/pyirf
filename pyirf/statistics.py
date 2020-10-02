import numpy as np

from .utils import is_scalar


def li_ma_significance(n_on, n_off, alpha=0.2):
    """
    Calculate the Li & Ma significance.

    Formula (17) doi.org/10.1086/161295

    This functions returns 0 significance when n_on < alpha * n_off
    instead of the negative sensitivities that would result from naively
    evaluating the formula.

    Parameters
    ----------
    n_on: integer or array like
        Number of events for the on observations
    n_off: integer of array like
        Number of events for the off observations
    alpha: float
        Ratio between the on region and the off region size or obstime.

    Returns
    -------
    s_lima: float or array
        The calculated significance
    """

    scalar = is_scalar(n_on)

    n_on = np.array(n_on, copy=False, ndmin=1)
    n_off = np.array(n_off, copy=False, ndmin=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_on = n_on / (n_on + n_off)
        p_off = n_off / (n_on + n_off)

        t1 = n_on * np.log(((1 + alpha) / alpha) * p_on)
        t2 = n_off * np.log((1 + alpha) * p_off)

        # lim x+->0  (x log(x)) = 0
        t1[n_on == 0] = 0
        t2[n_off == 0] = 0

        ts = t1 + t2

        significance = np.sqrt(ts * 2)

    significance[n_on < (alpha * n_off)] = 0

    if scalar:
        return significance[0]

    return significance
