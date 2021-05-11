import numpy as np
from astropy.table import Table
from scipy.stats import norm
import astropy.units as u

from ..binning import calculate_bin_indices


ONE_SIGMA_QUANTILE = norm.cdf(1) - norm.cdf(-1)


def angular_resolution(
    events, energy_bins, energy_type="true",
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

    Returns
    -------
    result : astropy.table.Table
        Table containing the 68% containment of the angular
        distance distribution per each reconstructed energy bin.
    """
    # create a table to make use of groupby operations
    table = Table(events[[f"{energy_type}_energy", "theta"]])

    table["bin_index"] = calculate_bin_indices(
        table[f"{energy_type}_energy"].quantity, energy_bins
    )

    n_bins =  len(energy_bins) - 1
    mask = (table["bin_index"] >= 0) & (table["bin_index"] < n_bins)

    result = Table()
    result[f"{energy_type}_energy_low"] = energy_bins[:-1]
    result[f"{energy_type}_energy_high"] = energy_bins[1:]
    result[f"{energy_type}_energy_center"] = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    result["angular_resolution"] = np.nan * u.deg

    if not len(events):
        # if we get an empty input (no selected events available)
        # we return the table filled with NaNs
        return result

    # use groupby operations to calculate the percentile in each bin
    by_bin = table[mask].group_by("bin_index")

    index = by_bin.groups.keys["bin_index"]
    result["angular_resolution"][index] = by_bin["theta"].groups.aggregate(
        lambda x: np.quantile(x, ONE_SIGMA_QUANTILE)
    )
    return result
