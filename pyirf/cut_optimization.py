import numpy as np
from astropy.table import QTable
import astropy.units as u
from tqdm import tqdm
import operator

from .cuts import evaluate_binned_cut, calculate_percentile_cut
from .sensitivity import calculate_sensitivity, estimate_background
from .binning import create_histogram_table, bin_center


__all__ = [
    "optimize_gh_cut",
]


def optimize_gh_cut(
    signal,
    background,
    reco_energy_bins,
    gh_cut_efficiencies,
    theta_cuts,
    op=operator.ge,
    background_radius=1 * u.deg,
    alpha=1.0,
    progress=True,
    **kwargs
):
    """
    Optimize the gh-score cut in every energy bin of reconstructed energy
    for best sensitivity.

    This procedure is EventDisplay-like, since it only applies a
    pre-computed theta cut and then optimizes only the gamma/hadron separation
    cut.

    Parameters
    ----------
    signal: astropy.table.QTable
        event list of simulated signal events.
        Required columns are `theta`, `reco_energy`, 'weight', `gh_score`
        No directional (theta) or gamma/hadron cut should already be applied.
    background: astropy.table.QTable
        event list of simulated background events.
        Required columns are `reco_source_fov_offset`, `reco_energy`,
        'weight', `gh_score`.
        No directional (theta) or gamma/hadron cut should already be applied.
    reco_energy_bins: astropy.units.Quantity[energy]
        Bins in reconstructed energy to use for sensitivity computation
    gh_cut_efficiencies: np.ndarray[float, ndim=1]
        The cut efficiencies to scan for best sensitivity.
    theta_cuts: astropy.table.QTable
        cut definition of the energy dependent theta cut,
        e.g. as created by ``calculate_percentile_cut``
    op: comparison function with signature f(a, b) -> bool
        The comparison function to use for the gamma hadron score.
        Returning true means an event passes the cut, so is not discarded.
        E.g. for gammaness-like score, use `operator.ge` (>=) and for a
        hadroness-like score use `operator.le` (<=).
    background_radius: astropy.units.Quantity[angle]
        Radius around the field of view center used for background rate
        estimation.
    alpha: float
        Size ratio of off region / on region. Will be used to
        scale the background rate.
    progress: bool
        If True, show a progress bar during cut optimization
    **kwargs are passed to ``calculate_sensitivity``
    """

    # we apply each cut for all reco_energy_bins globally, calculate the
    # sensitivity and then lookup the best sensitivity for each
    # bin independently

    signal_selected_theta = evaluate_binned_cut(
        signal['theta'], signal['reco_energy'], theta_cuts,
        op=operator.le,
    )

    sensitivities = []
    gh_cuts = []
    for efficiency in tqdm(gh_cut_efficiencies, disable=not progress):

        # calculate necessary percentile needed for
        # ``calculate_percentile_cut`` with the correct efficiency.
        # Depends on the operator, since we need to invert the
        # efficiency if we compare using >=, since percentile is
        # defines as <=.
        if op(-1, 1): # if operator behaves like "<=", "<" etc:
            percentile = 100 * efficiency
            fill_value = signal['gh_score'].min()
        else: # operator behaves like ">=", ">"
            percentile = 100 * (1 - efficiency)
            fill_value = signal['gh_score'].max()

        gh_cut = calculate_percentile_cut(
            signal['gh_score'], signal['reco_energy'],
            bins=reco_energy_bins,
            fill_value=fill_value, percentile=percentile,
        )
        gh_cuts.append(gh_cut)

        # apply the current cut
        signal_selected = evaluate_binned_cut(
            signal["gh_score"], signal["reco_energy"], gh_cut, op,
        ) & signal_selected_theta

        background_selected = evaluate_binned_cut(
            background["gh_score"], background["reco_energy"], gh_cut, op,
        )

        # create the histograms
        signal_hist = create_histogram_table(
            signal[signal_selected], reco_energy_bins, "reco_energy"
        )

        background_hist = estimate_background(
            events=background[background_selected],
            reco_energy_bins=reco_energy_bins,
            theta_cuts=theta_cuts,
            alpha=alpha,
            background_radius=background_radius
        )

        sensitivity = calculate_sensitivity(
            signal_hist, background_hist, alpha=alpha,
            **kwargs,
        )
        sensitivities.append(sensitivity)

    best_cut_table = QTable()
    best_cut_table["low"] = reco_energy_bins[0:-1]
    best_cut_table["center"] = bin_center(reco_energy_bins)
    best_cut_table["high"] = reco_energy_bins[1:]
    best_cut_table["cut"] = np.nan

    best_sensitivity = sensitivities[0].copy()
    for bin_id in range(len(reco_energy_bins) - 1):
        sensitivities_bin = [s["relative_sensitivity"][bin_id] for s in sensitivities]

        if not np.all(np.isnan(sensitivities_bin)):
            # nanargmin won't return the index of nan entries
            best = np.nanargmin(sensitivities_bin)
        else:
            # if all are invalid, just use the first one
            best = 0

        best_sensitivity[bin_id] = sensitivities[best][bin_id]
        best_cut_table["cut"][bin_id] = gh_cuts[best]["cut"][bin_id]

    return best_sensitivity, best_cut_table
