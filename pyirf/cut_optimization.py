import logging
import numpy as np
from astropy.table import QTable
import astropy.units as u
from tqdm import tqdm
import operator

from .cuts import evaluate_binned_cut, calculate_percentile_cut
from .sensitivity import calculate_sensitivity, estimate_background
from .binning import create_histogram_table


__all__ = [
    "optimize_gh_cut",
]


log = logging.getLogger(__name__)


def optimize_gh_cut(
    signal,
    background,
    reco_energy_bins,
    gh_efficiency_values,
    theta_cuts,
    op,
    background_radius=1 * u.deg,
    alpha=1.0,
    progress=True,
):
    """
    Optimize the gh-score in every energy bin.
    """

    # we apply each cut for all reco_energy_bins globally, calculate the
    # sensitivity and then lookup the best sensitivity for each
    # bin independently

    signal_theta_mask = evaluate_binned_cut(
        signal['theta'], signal['reco_energy'], theta_cuts, operator.le,
    )

    sensitivities = []
    cuts = []
    for target_efficiency in tqdm(gh_efficiency_values, disable=not progress):

        # create appropriate table for ``evaluate_binned_cut``
        gh_cuts = calculate_percentile_cut(
            signal['gh_score'], signal['reco_energy'], reco_energy_bins,
            percentile=100 * (1 - target_efficiency),
            fill_value=-1.0,
        )
        cuts.append(gh_cuts)

        signal_gh_mask = evaluate_binned_cut(
            signal["gh_score"], signal["reco_energy"], gh_cuts, op,
        )

        # apply the current cut
        signal_selected = signal_theta_mask & signal_gh_mask
        background_selected = evaluate_binned_cut(
            background["gh_score"], background["reco_energy"], gh_cuts, op,
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

    best_gh_cuts = QTable()
    best_gh_cuts["low"] = reco_energy_bins[0:-1]
    best_gh_cuts["high"] = reco_energy_bins[1:]
    best_gh_cuts["cut"] = np.nan

    copy_cols = ['reco_energy_' + k for k in ('low', 'high', 'center')]
    best_sensitivity = QTable()
    for col in copy_cols:
        best_sensitivity[col] = sensitivities[0][col]

    nan_cols = [
        'n_signal',
        'n_signal_weighted',
        'n_background',
        'n_background_weighted',
        'relative_sensitivity'
    ]
    for col in nan_cols:
        best_sensitivity[col] = np.nan

    for bin_id in range(len(reco_energy_bins) - 1):
        sensitivities_bin = [s["relative_sensitivity"][bin_id] for s in sensitivities]
        if not np.all(np.isnan(sensitivities_bin)):
            # nanargmin won't return the index of nan entries
            best = np.nanargmin(sensitivities_bin)
            best_sensitivity[bin_id] = sensitivities[best][bin_id]
            best_gh_cuts["cut"][bin_id] = cuts[best]["cut"][bin_id]

    return best_sensitivity, best_gh_cuts
