import astropy.units as u
import numpy as np
from scipy.optimize import brentq
from astropy.table import QTable
import logging

from .statistics import li_ma_significance


log = logging.getLogger(__name__)


@u.quantity_input(t_obs=u.hour, t_ref=u.hour)
def relative_sensitivity(
    n_on,
    n_off,
    alpha,
    t_obs,
    t_ref=u.Quantity(50, u.hour),
    target_significance=5,
    significance_function=li_ma_significance,
    initial_guess=0.01,
):
    '''
    Calculate the relative sensitivity defined as the flux
    relative to the reference source that is detectable with
    significance ``target_significance`` in time ``t_ref``.

    Given measured ``n_on`` and ``n_off`` during a time period ``t_obs``,
    we estimate the number of gamma events ``n_signal`` as ``n_on - alpha * n_off``.

    The number of background events ``n_background` is estimated as ``n_off * alpha``.

    In the end, we find the relative sensitivity as the scaling factor for ``n_signal``
    that yields a significance of ``target_significance``.


    Parameters
    ----------
    n_on: int or array-like
        Number of signal-like events for the on observations
    n_off: int or array-like
        Number of signal-like events for the off observations
    alpha: float
        Scaling factor between on and off observations.
        1 / number of off regions for wobble observations.
    t_obs: astropy.units.Quantity of type time
        Total observation time
    t_ref: astropy.units.Quantity of type time
        Reference time for the detection
    significance: float
        Significance necessary for a detection
    significance_function: function
        A function f(n_on, n_off, alpha) -> significance in sigma
        Used to calculate the significance, default is the Li&Ma
        likelihood ratio test formula.
        Li, T-P., and Y-Q. Ma.
        "Analysis methods for results in gamma-ray astronomy."
        The Astrophysical Journal 272 (1983): 317-324.
        Formula (17)
    initial_guess: float
        Initial guess for the root finder
    '''
    ratio = (t_ref / t_obs).to(u.one)
    n_on = n_on * ratio
    n_off = n_off * ratio

    n_background = n_off * alpha
    n_signal = n_on - n_background

    if np.isnan(n_on) or np.isnan(n_off):
        return np.nan

    if n_on == 0 or n_off == 0:
        return np.nan

    if n_signal <= 0:
        return np.nan

    def equation(relative_flux):
        n_on = n_signal * relative_flux + n_background
        s = significance_function(n_on, n_off, alpha)
        return s - target_significance

    try:
        # brentq needs a lower and an upper bound
        # lower can be trivially set to zero, but the upper bound is more tricky
        # we will use the simple, analytically  solvable significance formula and scale it
        # with 10 to be sure it's above the Li and Ma solution
        # so rel * n_signal / sqrt(n_background) = target_significance
        upper_bound =  10 * target_significance * np.sqrt(n_background) / n_signal
        result = brentq(
            equation,
            0, upper_bound,
        )
    except (RuntimeError, ValueError):
        log.warn(
            'Could not calculate relative significance for'
            f' n_signal={n_signal:.1f}, n_off={n_off:.1f}, returning nan'
        )
        return np.nan

    return result


@u.quantity_input(t_obs=u.hour, t_ref=u.hour)
def calculate_sensitivity(
    signal_hist,
    background_hist,
    alpha,
    t_obs,
    t_ref=u.Quantity(50, u.hour),
    target_significance=5,
    significance_function=li_ma_significance,
):
    assert len(signal_hist) == len(background_hist)

    sensitivity = QTable()

    # check binning information and add to output
    for k in ('low', 'center', 'high'):
        k = 'reco_energy_' + k
        if not np.all(signal_hist[k] == background_hist[k]):
            raise ValueError('Binning for signal_hist and background_hist must be equal')

        sensitivity[k] = signal_hist[k]

    # add event number information
    sensitivity['n_signal'] = signal_hist['n']
    sensitivity['n_signal_weighted'] = signal_hist['n_weighted']
    sensitivity['n_background'] = background_hist['n']
    sensitivity['n_background_weighted'] = background_hist['n_weighted']

    sensitivity['relative_sensitivity'] = [
        relative_sensitivity(
            n_on=n_signal_hist + alpha * n_background_hist,
            n_off=n_background_hist,
            alpha=alpha,
            t_obs=t_obs,
        )
        for n_signal_hist, n_background_hist in zip(signal_hist['n_weighted'], background_hist['n_weighted'])
    ]

    # safety checks
    invalid = (
        (sensitivity['n_signal_weighted'] < 10) |
        (sensitivity['n_signal_weighted'] < 0.05 * sensitivity['n_background_weighted'])
    )
    sensitivity['relative_sensitivity'][invalid] = np.nan

    return sensitivity
