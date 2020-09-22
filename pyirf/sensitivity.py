import astropy.units as u
import numpy as np
from scipy.optimize import newton
import warnings
from astropy.table import QTable


from .statistics import li_ma_significance


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
        return significance_function(n_on, n_off, alpha) - target_significance

    try:
        result = newton(
            equation,
            x0=initial_guess,
        )
    except RuntimeError:
        warnings.warn('Could not calculate relative significance, returning nan')
        return np.nan

    if result.size == 1:
        return result[0]

    return result


@u.quantity_input(t_obs=u.hour, t_ref=u.hour)
def calculate_sensitivity(
    signal,
    background,
    alpha,
    t_obs,
    t_ref=u.Quantity(50, u.hour),
    target_significance=5,
    significance_function=li_ma_significance,
    initial_guess=0.5,
):
    assert len(signal) == len(background)

    sensitivity = QTable()

    # check binning information and add to output
    for k in ('low', 'center', 'high'):
        k = 'reco_energy_' + k
        if not np.all(signal[k] == background[k]):
            raise ValueError('Binning for signal and background must be equal')

        sensitivity[k] = signal[k]

    # add event number information
    sensitivity['n_signal'] = signal['n']
    sensitivity['n_signal_weighted'] = signal['n_weighted']
    sensitivity['n_background'] = background['n']
    sensitivity['n_background_weighted'] = background['n_weighted']

    sensitivity['relative_sensitivity'] = [
        relative_sensitivity(
            n_on=n_signal + alpha * n_background,
            n_off=n_background,
            alpha=1.0,
            t_obs=t_obs,
        )
        for n_signal, n_background in zip(signal['n_weighted'], background['n_weighted'])
    ]

    return sensitivity
