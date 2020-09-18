import astropy.units as u
import numpy as np
from scipy.optimize import newton
import warnings


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
    initial_guess=0.5,
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

    ratio = (t_ref / t_obs).si
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

    return result


# make the function accept numpy arrays for n_on, n_off, 
# so we can provide all energy bins
relative_sensitivity = np.vectorize(
    relative_sensitivity,
    excluded=[
        't_obs',
        't_ref',
        'alpha',
        'target_significance',
        'significance_function',
    ]
)
