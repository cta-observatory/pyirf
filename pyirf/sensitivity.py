"""
Functions to calculate sensitivity
"""
import numpy as np
from scipy.optimize import brentq
import logging

from astropy.table import QTable
import astropy.units as u

from .statistics import li_ma_significance
from .utils import check_histograms, cone_solid_angle
from .binning import create_histogram_table, bin_center


__all__ = ["relative_sensitivity", "calculate_sensitivity"]


log = logging.getLogger(__name__)


def _relative_sensitivity(
    n_on,
    n_off,
    alpha,
    min_significance=5,
    min_signal_events=10,
    min_excess_over_background=0.05,
    significance_function=li_ma_significance,
):
    if np.isnan(n_on) or np.isnan(n_off):
        return np.nan

    if n_on < 0 or n_off < 0:
        raise ValueError(f'n_on and n_off must be positive, got {n_on}, {n_off}')

    n_background = n_off * alpha
    n_signal = n_on - n_background

    if n_signal <= 0:
        return np.inf

    def equation(relative_flux):
        n_on = n_signal * relative_flux + n_background
        s = significance_function(n_on, n_off, alpha)
        return s - min_significance

    try:
        # brentq needs a lower and an upper bound
        # we will use the simple, analytically  solvable significance formula and scale it
        # with 10 to be sure it's above the Li and Ma solution
        # so rel * n_signal / sqrt(n_background) = target_significance
        if n_off > 1:
            relative_flux_naive = min_significance * np.sqrt(n_background) / n_signal
            upper_bound = 10 * relative_flux_naive
            lower_bound = 0.01 * relative_flux_naive
        else:
            upper_bound = 100
            lower_bound = 1e-4

        relative_flux = brentq(equation, lower_bound, upper_bound)

    except (RuntimeError, ValueError) as e:
        log.warn(
            "Could not calculate relative significance for"
            f" n_signal={n_signal:.1f}, n_off={n_off:.1f}, returning nan {e}"
        )
        return np.nan

    # less than min_sigma
    if relative_flux > 1:
        return np.nan

    # scale to achieved flux level
    n_signal = n_signal * relative_flux
    min_excess = min_excess_over_background * n_background
    min_signal = max(min_signal_events, min_excess)

    # if we violate the min signal events condition,
    # increase flux until we meet the requirement
    if n_signal < min_signal:
        scale = min_signal / n_signal
    else:
        scale = 1.0

    return relative_flux * scale


_relative_sensitivity_vectorized = np.vectorize(
    _relative_sensitivity,
    excluded=['significance_function']
)


def relative_sensitivity(
    n_on,
    n_off,
    alpha,
    min_significance=5,
    min_signal_events=10,
    min_excess_over_background=0.05,
    significance_function=li_ma_significance,
):
    """
    Calculate the relative sensitivity defined as the flux
    relative to the reference source that is detectable with
    significance ``target_significance``.

    Given measured ``n_on`` and ``n_off``,
    we estimate the number of gamma events ``n_signal`` as ``n_on - alpha * n_off``.

    The number of background events ``n_background` is estimated as ``n_off * alpha``.

    In the end, we find the relative sensitivity as the scaling factor for ``n_signal``
    that yields a significance of ``target_significance``.

    The reference time should be incorporated by appropriately weighting the events
    before calculating ``n_on`` and ``n_off``.

    All input values with the exception of ``significance_function``
    must be broadcastable to a single, common shape.

    Parameters
    ----------
    n_on: int or array-like
        Number of signal-like events for the on observations
    n_off: int or array-like
        Number of signal-like events for the off observations
    alpha: float or array-like
        Scaling factor between on and off observations.
        1 / number of off regions for wobble observations.
    min_significance: float or array-like
        Significance necessary for a detection
    min_signal_events: int or array-like
        Minimum number of signal events required.
        The relative flux will be scaled up from the one yielding ``min_significance``
        if this condition is violated.
    min_excess_over_background: float or array-like
        Minimum number of signal events expressed as the proportion of the
        background.
        So the required number of signal events will be
        ``min_excess_over_background * alpha * n_off``.
        The relative flux will be scaled up from the one yielding ``min_significance``
        if this condition is violated.
    significance_function: function
        A function f(n_on, n_off, alpha) -> significance in sigma
        Used to calculate the significance, default is the Li&Ma
        likelihood ratio test formula.
        Li, T-P., and Y-Q. Ma.
        "Analysis methods for results in gamma-ray astronomy."
        The Astrophysical Journal 272 (1983): 317-324.
        Formula (17)
    """
    return _relative_sensitivity_vectorized(
        n_on=n_on,
        n_off=n_off,
        alpha=alpha,
        min_significance=min_significance,
        min_signal_events=min_signal_events,
        min_excess_over_background=min_excess_over_background,
        significance_function=significance_function,
    )


def calculate_sensitivity(
    signal_hist,
    background_hist,
    alpha,
    min_significance=5,
    min_signal_events=10,
    min_excess_over_background=0.05,
    significance_function=li_ma_significance,
):
    """
    Calculate sensitivity for DL2 event lists in bins of reconstructed energy.

    Sensitivity is defined as the minimum flux detectable with ``target_significance``
    sigma significance in a certain time.

    This time must be incorporated into the event weights.

    Two conditions are required for the sensitivity:
    - At least ten weighted signal events
    - The weighted signal must be larger than 5 % of the weighted background
    - At least 5 sigma (so relative_sensitivity > 1)

    If the conditions are not met, the sensitivity will be set to nan.

    Parameters
    ----------
    signal_hist: astropy.table.QTable
        Histogram of detected signal events as a table.
        Required columns: n and n_weighted.
        See ``pyirf.binning.create_histogram_table``
    background_hist: astropy.table.QTable
        Histogram of detected events as a table.
        Required columns: n and n_weighted.
        See ``pyirf.binning.create_histogram_table``
    alpha: float
        Size ratio of signal region to background region
    min_significance: float
        Significance necessary for a detection
    min_signal_events: int
        Minimum number of signal events required.
        The relative flux will be scaled up from the one yielding ``min_significance``
        if this condition is violated.
    min_excess_over_background: float
        Minimum number of signal events expressed as the proportion of the
        background.
        So the required number of signal events will be
        ``min_excess_over_background * alpha * n_off``.
        The relative flux will be scaled up from the one yielding ``min_significance``
        if this condition is violated.
    significance_function: callable
        A function with signature (n_on, n_off, alpha) -> significance.
        Default is the Li & Ma likelihood ratio test.

    Returns
    -------
    sensitivity_table: astropy.table.QTable
        Table with sensitivity information.
        Contains weighted and unweighted number of signal and background events
        and the ``relative_sensitivity``, the scaling applied to the signal events
        that yields ``target_significance`` sigma of significance according to
        the ``significance_function``
    """
    check_histograms(signal_hist, background_hist)

    n_on = signal_hist["n_weighted"] + alpha * background_hist["n_weighted"]

    # convert any quantities to arrays,
    # since quantitites don't work with vectorize
    n_on = u.Quantity(n_on, copy=False).to_value(u.one)
    n_off = u.Quantity(background_hist["n_weighted"], copy=False).to_value(u.one)

    # calculate sensitivity in each bin
    rel_sens = relative_sensitivity(
        n_on=n_on,
        n_off=n_off,
        alpha=alpha,
        min_significance=min_significance,
        min_signal_events=min_signal_events,
        min_excess_over_background=min_excess_over_background,
        significance_function=significance_function,
    )

    # fill output table
    s = QTable()
    for key in ("low", "high", "center"):
        k = "reco_energy_" + key
        s[k] = signal_hist[k]

    s["n_signal"] = signal_hist["n"] * rel_sens
    s["n_signal_weighted"] = signal_hist["n_weighted"] * rel_sens
    s["n_background"] = background_hist["n"]
    s["n_background_weighted"] = background_hist["n_weighted"]

    # copy also "n_proton" / "n_electron_weighted" etc. if available
    for k in filter(lambda c: c.startswith('n_') and c != 'n_weighted', background_hist.colnames):
        s[k] = background_hist[k]

    s["significance"] = significance_function(
        n_on=s["n_signal_weighted"] + alpha * s["n_background_weighted"],
        n_off=s["n_background_weighted"],
        alpha=alpha,
    )
    s["relative_sensitivity"] = rel_sens

    return s


def estimate_background(
    events, reco_energy_bins, theta_cuts, alpha, background_radius
):
    '''
    Estimate the number of background events for a point-like sensitivity.

    Due to limited statistics, it is often not possible to just apply the same
    theta cut to the background events as to the signal events around an assumed
    source position.

    Here we calculate the expected number of background events for the off
    regions by taking all background events up to `background_radius` away from
    the camera center and then scale these to the size of the off region,
    which is scaled by 1 / alpha from the size of the on region given by the
    theta cuts.


    Parameters
    ----------
    events: astropy.table.QTable
        DL2 event list of background surviving event selection
        and inside ``background_radius`` from the center of the FOV
        Required columns for this function:
        - `reco_energy`,
        - `reco_source_fov_offset`.
    reco_energy_bins: astropy.units.Quantity[energy]
        Desired bin edges in reconstructed energy for the background rate
    theta_cuts: astropy.table.QTable
        The cuts table for the theta cut,
        e.g. as returned by ``~pyirf.cuts.calculate_percentile_cut``.
        Columns `center` and `cut` are required for this function.
    alpha: float
        size of the on region divided by the size of the off region.
    background_radius: astropy.units.Quantity[angle]
        Maximum distance from the fov center for background events to be taken into account
    '''
    bg = create_histogram_table(
        events[events['reco_source_fov_offset'] < background_radius],
        reco_energy_bins,
        key='reco_energy',
    )

    # scale number of background events according to the on region size
    # background radius and alpha
    center = bin_center(reco_energy_bins)
    # interpolate the theta cut to the bins used here
    theta_cuts_bg_bins = np.interp(
        center,
        theta_cuts['center'],
        theta_cuts['cut']
    )
    size_ratio = (
        cone_solid_angle(theta_cuts_bg_bins)
        / cone_solid_angle(background_radius)
    ).to_value(u.one)

    for key in filter(lambda col: col.startswith('n'), bg.colnames):
        # *= not possible due to upcast from int to float
        bg[key] = bg[key] * size_ratio / alpha

    return bg
