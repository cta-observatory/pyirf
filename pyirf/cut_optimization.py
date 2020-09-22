import numpy as np
from astropy.table import Table
import astropy.units as u
from tqdm import tqdm

from .cuts import evaluate_binned_cut
from .sensitivity import calculate_sensitivity
from .binning import create_histogram_table


def optimize_gh_cut(signal, background, bins, cut_values, op, progress=True):
    '''
    Optimize the gh-score in every energy bin.
    Theta Squared Cut  should already be applied on the input tables.
    '''

    # we apply each cut for all bins globally, calculate the 
    # sensitivity and then lookup the best sensitivity for each 
    # bin independently

    sensitivities = []
    for cut_value in tqdm(cut_values, disable=not progress):

        # create appropriate table for ``evaluate_binned_cut``
        cut_table = Table()
        cut_table['low'] = bins[0:-1]
        cut_table['high'] = bins[1:]
        cut_table['cut'] = cut_value

        # apply the current cut
        signal_selected = evaluate_binned_cut(
            signal['gh_score'],
            signal['reco_energy'],
            cut_table,
            op,
        )

        background_selected = evaluate_binned_cut(
            background['gh_score'],
            background['reco_energy'],
            cut_table,
            op,
        )

        # create the histograms
        signal_hist = create_histogram_table(
            signal[signal_selected], bins, 'reco_energy'
        )
        background_hist = create_histogram_table(
            background[background_selected], bins, 'reco_energy'
        )

        sensitivity = calculate_sensitivity(
            signal_hist,
            background_hist,
            alpha=1,
            t_obs=50 * u.hour,
        )
        sensitivities.append(sensitivity)

    best_cut_table = Table()
    best_cut_table['low'] = bins[0:-1]
    best_cut_table['high'] = bins[1:]
    best_cut_table['cut'] = np.nan

    best_sensitivity = sensitivities[0].copy()
    for bin_id in range(len(bins) - 1):
        sensitivities_bin = [s['relative_sensitivity'][bin_id] for s in sensitivities]

        if not np.all(np.isnan(sensitivities_bin)):
            # nanargmin won't return the index of nan entries
            best = np.nanargmin(sensitivities_bin)
        else:
            # if all are invalid, just use the first one
            best = 0

        best_sensitivity[bin_id] = sensitivities[best][bin_id]
        best_cut_table['cut'][bin_id] = cut_values[best]

    return best_sensitivity, best_cut_table
