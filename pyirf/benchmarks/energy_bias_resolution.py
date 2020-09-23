from astropy.table import Table
import numpy as np
from scipy.stats import norm

from ..binning import calculate_bin_indices


NORM_UPPER_SIGMA = norm(0, 1).cdf(1)
NORM_LOWER_SIGMA = norm(0, 1).cdf(-1)


def resolution_abelardo(rel_error):
    return np.percentile(np.abs(rel_error), 68)


def inter_quantile_distance(rel_error):
    upper_sigma = np.percentile(rel_error, 100 * NORM_UPPER_SIGMA)
    lower_sigma = np.percentile(rel_error, 100 * NORM_LOWER_SIGMA)
    return 0.5 * (upper_sigma - lower_sigma)


def energy_bias_resolution(
    events,
    true_energy_bins,
    bias_function=np.median,
    resolution_function=inter_quantile_distance
):

    # create a table to make use of groupby operations
    table = Table(events[['true_energy', 'reco_energy']])
    table['rel_error'] = (events['reco_energy'] / events['true_energy']) - 1

    table['bin_index'] = calculate_bin_indices(
        table['true_energy'].quantity, true_energy_bins
    )

    result = Table()
    result['true_energy_low'] = true_energy_bins[:-1]
    result['true_energy_high'] = true_energy_bins[1:]
    result['true_energy_center'] = 0.5 * (true_energy_bins[:-1] + true_energy_bins[1:])

    result['bias'] = np.nan
    result['resolution'] = np.nan

    # use groupby operations to calculate the percentile in each bin
    by_bin = table.group_by('bin_index')

    index = by_bin.groups.keys['bin_index']
    result['bias'][index] = by_bin['rel_error'].groups.aggregate(bias_function)
    result['resolution'][index] = by_bin['rel_error'].groups.aggregate(resolution_function)
    return result
