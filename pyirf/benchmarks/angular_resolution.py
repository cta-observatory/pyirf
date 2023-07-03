import numpy as np
from astropy.table import QTable
from scipy.stats import norm
import astropy.units as u

from ..binning import calculate_bin_indices


ONE_SIGMA_QUANTILE = norm.cdf(1) - norm.cdf(-1)


def angular_resolution(
    events, energy_bins,
    energy_type="true",
    quantile=ONE_SIGMA_QUANTILE,
):
    """
    Calculate the angular resolution.

    This implementation corresponds to the 68% containment of the angular
    distance distribution.

    Parameters
    ----------
    events : astropy.table.QTable
        Astropy Table object containing the reconstructed events information.
    energy_bins: numpy.ndarray(dtype=float, ndim=1)
        Bin edges in energy.
    energy_type: str
        Either "true" or "reco" energy.
        Default is "true".
    quantile : float
        Which quantile to use for the angular resolution,
        by default, the containment of the 1-sigma region
        of the normal distribution (~68%) is used.

    Returns
    -------
    result : astropy.table.QTable
        QTable containing the 68% containment of the angular
        distance distribution per each reconstructed energy bin.
    """
    # create a table to make use of groupby operations
    energy_key = f"{energy_type}_energy"
    table = QTable(events[[energy_key, "theta"]])

    bin_index, valid = calculate_bin_indices(table[energy_key], energy_bins)

    result = QTable()
    result[f"{energy_key}_low"] = energy_bins[:-1]
    result[f"{energy_key}_high"] = energy_bins[1:]
    result[f"{energy_key}_center"] = 0.5 * (energy_bins[:-1] + energy_bins[1:])
    result["n_events"] = 0

    key = "angular_resolution"
    result[key] = u.Quantity(np.nan, table["theta"].unit)

    # if we get an empty input (no selected events available)
    # we return the table filled with NaNs
    if len(events) == 0:
        return result

    # use groupby operations to calculate the percentile in each bin
    by_bin = table[valid].group_by(bin_index[valid])
    for bin_idx, group in zip(by_bin.groups.keys, by_bin.groups):
        result[key][bin_idx] = np.nanquantile(group["theta"], quantile)
        result["n_events"][bin_idx] = len(group)
    return result
