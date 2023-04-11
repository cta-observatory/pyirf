import numpy as np
from scipy.stats import norm
from astropy.table import QTable
import astropy.units as u

from ..binning import calculate_bin_indices, UNDERFLOW_INDEX, OVERFLOW_INDEX


NORM_LOWER_SIGMA, NORM_UPPER_SIGMA = norm(0, 1).cdf([-1, 1])
ONE_SIGMA_COVERAGE = NORM_UPPER_SIGMA - NORM_LOWER_SIGMA
MEDIAN = 0.5


def energy_resolution_absolute_68(rel_error):
    """Calculate the energy resolution as the central 68% interval.

    Utility function for pyirf.benchmarks.energy_bias_resolution

    Parameters
    ----------
    rel_error : numpy.ndarray(dtype=float, ndim=1)
        Array of float on which the quantile is calculated.

    Returns
    -------
    resolution: numpy.ndarray(dtype=float, ndim=1)
        Array containing the 68% intervals
    """
    return np.nanquantile(np.abs(rel_error), ONE_SIGMA_COVERAGE)


def inter_quantile_distance(rel_error):
    """Calculate the energy resolution as the half of the 68% containment.

    Percentile equivalent of the standard deviation.
    Utility function for pyirf.benchmarks.energy_bias_resolution

    Parameters
    ----------
    rel_error : numpy.ndarray(dtype=float, ndim=1)
        Array of float on which the quantile is calculated.

    Returns
    -------
    resolution: numpy.ndarray(dtype=float, ndim=1)
        Array containing the resolution values.
    """
    upper_sigma = np.nanquantile(rel_error, NORM_UPPER_SIGMA)
    lower_sigma = np.nanquantile(rel_error, NORM_LOWER_SIGMA)
    resolution = 0.5 * (upper_sigma - lower_sigma)
    return resolution


def energy_bias_resolution(
    events,
    energy_bins,
    energy_type="true",
    bias_function=np.nanmedian,
    resolution_function=inter_quantile_distance,
):
    """
    Calculate bias and energy resolution.

    Parameters
    ----------
    events: astropy.table.QTable
        Astropy Table object containing the reconstructed events information.
    energy_bins: numpy.ndarray(dtype=float, ndim=1)
        Bin edges in energy.
    energy_type: str
        Either "true" or "reco" energy.
        Default is "true".
    bias_function: callable
        Function used to calculate the energy bias
    resolution_function: callable
        Function used to calculate the energy resolution

    Returns
    -------
    result : astropy.table.QTable
        QTable containing the energy bias and resolution
        per each bin in true energy.
    """

    # create a table to make use of groupby operations
    table = QTable(events[["true_energy", "reco_energy"]], copy=False)
    table["rel_error"] = (events["reco_energy"] / events["true_energy"]).to_value(u.one) - 1

    energy_key = f"{energy_type}_energy"

    result = QTable()
    result[f"{energy_key}_low"] = energy_bins[:-1]
    result[f"{energy_key}_high"] = energy_bins[1:]
    result[f"{energy_key}_center"] = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    result["n_events"] = 0
    result["bias"] = np.nan
    result["resolution"] = np.nan

    if not len(events):
        # if we get an empty input (no selected events available)
        # we return the table filled with NaNs
        return result


    # use groupby operations to calculate the percentile in each bin
    bin_index, valid = calculate_bin_indices(table[energy_key], energy_bins)
    by_bin = table.group_by(bin_index)

    # use groupby operations to calculate the percentile in each bin
    by_bin = table[valid].group_by(bin_index[valid])
    for bin_idx, group in zip(by_bin.groups.keys, by_bin.groups):
        result["n_events"][bin_idx] = len(group)
        result["bias"][bin_idx] = bias_function(group["rel_error"])
        result["resolution"][bin_idx] = resolution_function(group["rel_error"])
    return result

def energy_bias_resolution_from_energy_dispersion(
    energy_dispersion,
    migration_bins,
):
    """
    Calculate bias and energy resolution.

    Parameters
    ----------
    edisp:
        Energy dispersion matrix of shape
        (n_energy_bins, n_migra_bins, n_source_offset_bins)
    migration_bins: numpy.ndarray
        Bin edges for the relative energy migration (``reco_energy / true_energy``)
    """

    cdf = np.cumsum(energy_dispersion, axis=1)

    n_energy_bins, _, n_fov_bins = energy_dispersion.shape

    bias = np.zeros((n_energy_bins, n_fov_bins))
    resolution = np.zeros((n_energy_bins, n_fov_bins))

    for energy_bin in range(n_energy_bins):
        for fov_bin in range(n_fov_bins):

            low, median, high = np.interp(
                [NORM_LOWER_SIGMA, MEDIAN, NORM_UPPER_SIGMA],
                cdf[energy_bin, :, fov_bin],
                migration_bins[1:] # cdf is defined at upper bin edge
            )
            bias[energy_bin, fov_bin] = median - 1
            resolution[energy_bin, fov_bin] = 0.5 * (high - low)

    return bias, resolution
