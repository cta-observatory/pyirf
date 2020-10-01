"""
Functions to calculate sensitivity
"""
import astropy.units as u
import numpy as np
from scipy.optimize import brentq
from astropy.table import QTable
import logging

from .statistics import li_ma_significance
from .utils import check_histograms, cone_solid_angle
from .binning import create_histogram_table, bin_center


__all__ = ["relative_sensitivity", "calculate_sensitivity"]


log = logging.getLogger(__name__)


def relative_sensitivity(
    n_on,
    n_off,
    alpha,
    target_significance=5,
    significance_function=li_ma_significance,
    initial_guess=0.01,
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
    before calculating ``n_on`` and ``n_off``

    Parameters
    ----------
    n_on: int or array-like
        Number of signal-like events for the on observations
    n_off: int or array-like
        Number of signal-like events for the off observations
    alpha: float
        Scaling factor between on and off observations.
        1 / number of off regions for wobble observations.
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
    """
    n_background = n_off * alpha
    n_signal = n_on - n_background

    if np.isnan(n_on) or np.isnan(n_off):
        return np.nan

    if n_on < 1 or n_off < 1:
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
        upper_bound = 10 * target_significance * np.sqrt(n_background) / n_signal
        result = brentq(equation, 0, upper_bound,)
    except (RuntimeError, ValueError):
        log.warn(
            "Could not calculate relative significance for"
            f" n_signal={n_signal:.1f}, n_off={n_off:.1f}, returning nan"
        )
        return np.nan

    return result


def calculate_sensitivity(
    signal_hist,
    background_hist,
    alpha,
    target_significance=5,
    significance_function=li_ma_significance,
):
    """
    Calculate sensitivity for DL2 event lists in bins of reconstructed energy.

    Sensitivity is defined as the minimum flux detectable with ``target_significance``
    sigma significance in a certain time.

    This time must be incorporated into the event weights.

    Two conditions are required for the sensitivity:
    - At least ten weighted signal events
    - the weighted signal events must be larger than 5 % of the weighted background

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
    target_significance: float
        Required significance
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
    assert len(signal_hist) == len(background_hist)

    check_histograms(signal_hist, background_hist)
    sensitivity = QTable()
    for key in ("low", "high", "center"):
        k = "reco_energy_" + key
        sensitivity[k] = signal_hist[k]

    # add event number information
    sensitivity["n_signal"] = signal_hist["n"]
    sensitivity["n_signal_weighted"] = signal_hist["n_weighted"]
    sensitivity["n_background"] = background_hist["n"]
    sensitivity["n_background_weighted"] = background_hist["n_weighted"]

    sensitivity["relative_sensitivity"] = [
        relative_sensitivity(
            n_on=n_signal + alpha * n_background,
            n_off=n_background,
            alpha=alpha,
        )
        for n_signal, n_background in zip(
            signal_hist["n_weighted"], background_hist["n_weighted"]
        )
    ]

    # safety checks according to the IRF document
    # at least ten signal events and the number of signal events
    # must be larger then five percent of the remaining background
    invalid = (
        (sensitivity["n_signal_weighted"] < 10)
        | (
            sensitivity["n_signal_weighted"]
            < (0.05 * alpha * sensitivity["n_background_weighted"])
        )
    )
    sensitivity["relative_sensitivity"][invalid] = np.nan

    return sensitivity


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
        - `source_fov_offset`.
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
        events[events['source_fov_offset'] < background_radius],
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

    for key in ['n', 'n_weighted']:
        # *= not possible due to upcast from int to float
        bg[key] = bg[key] * size_ratio / alpha

    return bg
