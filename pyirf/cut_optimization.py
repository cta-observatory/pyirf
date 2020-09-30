import numpy as np
from astropy.table import QTable
import astropy.units as u
from tqdm import tqdm

from .cuts import evaluate_binned_cut
from .sensitivity import calculate_sensitivity, estimate_background
from .binning import create_histogram_table


__all__ = [
    "optimize_gh_cut",
]


def optimize_gh_cut(
    signal,
    background,
    reco_energy_bins,
    gh_cut_values,
    theta_cuts,
    op,
    background_radius=1 * u.deg,
    alpha=1.0,
    progress=True,
):
    """
    Optimize the gh-score in every energy bin.
    Theta Squared Cut  should already be applied on the input tables.
    """

    # we apply each cut for all reco_energy_bins globally, calculate the
    # sensitivity and then lookup the best sensitivity for each
    # bin independently

    sensitivities = []
    for cut_value in tqdm(gh_cut_values, disable=not progress):

        # create appropriate table for ``evaluate_binned_cut``
        cut_table = QTable()
        cut_table["low"] = reco_energy_bins[:-1]
        cut_table["high"] = reco_energy_bins[1:]
        cut_table["cut"] = cut_value

        # apply the current cut
        signal_selected = evaluate_binned_cut(
            signal["gh_score"], signal["reco_energy"], cut_table, op,
        )

        background_selected = evaluate_binned_cut(
            background["gh_score"], background["reco_energy"], cut_table, op,
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
        )
        sensitivities.append(sensitivity)

    best_cut_table = QTable()
    best_cut_table["low"] = reco_energy_bins[0:-1]
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
        best_cut_table["cut"][bin_id] = gh_cut_values[best]

    return best_sensitivity, best_cut_table
