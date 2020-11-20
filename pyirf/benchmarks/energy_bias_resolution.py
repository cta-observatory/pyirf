from astropy.table import Table
import numpy as np
from scipy.stats import norm

from ..binning import calculate_bin_indices


NORM_UPPER_SIGMA = norm(0, 1).cdf(1)
NORM_LOWER_SIGMA = norm(0, 1).cdf(-1)


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
    resolution = np.percentile(np.abs(rel_error), 68)
    return resolution


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
    upper_sigma = np.percentile(rel_error, 100 * NORM_UPPER_SIGMA)
    lower_sigma = np.percentile(rel_error, 100 * NORM_LOWER_SIGMA)
    resolution = 0.5 * (upper_sigma - lower_sigma)
    return resolution


def energy_bias_resolution(
    events,
    energy_bins,
    energy_type="true",
    bias_function=np.median,
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
    result : astropy.table.Table
        Table containing the energy bias and resolution
        per each bin in true energy.
    """

    # create a table to make use of groupby operations
    table = Table(events[["true_energy", "reco_energy"]])
    table["rel_error"] = (events["reco_energy"] / events["true_energy"]) - 1

    table["bin_index"] = calculate_bin_indices(
        table[f"{energy_type}_energy"].quantity, energy_bins
    )

    result = Table()
    result[f"{energy_type}_energy_low"] = energy_bins[:-1]
    result[f"{energy_type}_energy_high"] = energy_bins[1:]
    result[f"{energy_type}_energy_center"] = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    result["bias"] = np.nan
    result["resolution"] = np.nan

    # use groupby operations to calculate the percentile in each bin
    by_bin = table.group_by("bin_index")

    index = by_bin.groups.keys["bin_index"]
    result["bias"][index] = by_bin["rel_error"].groups.aggregate(bias_function)
    result["resolution"][index] = by_bin["rel_error"].groups.aggregate(
        resolution_function
    )
    return result
