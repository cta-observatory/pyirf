import numpy as np
from astropy.table import Table
from scipy.stats import norm
import astropy.units as u

from ..binning import calculate_bin_indices


ONE_SIGMA_PERCENTILE = norm.cdf(1) - norm.cdf(-1)


def angular_resolution(
    events, true_energy_bins,
):
    """
    Calculate the angular resolution.

    This implementation corresponds to the 68% containment of the angular
    distance distribution.

    Parameters
    ----------
    events : astropy.table.QTable
        Astropy Table object containing the reconstructed events information.
    true_energy_bins: numpy.ndarray(dtype=float, ndim=1)
        Bin edges in true energy.

    Returns
    -------
    result : astropy.table.Table
        Table containing the 68% containment of the angular
        distance distribution per each true energy bin.
    """

    # create a table to make use of groupby operations
    table = Table(events[["true_energy", "theta"]])

    table["bin_index"] = calculate_bin_indices(
        table["true_energy"].quantity, true_energy_bins
    )

    result = Table()
    result["true_energy_low"] = true_energy_bins[:-1]
    result["true_energy_high"] = true_energy_bins[1:]
    result["true_energy_center"] = 0.5 * (true_energy_bins[:-1] + true_energy_bins[1:])

    result["angular_resolution"] = np.nan * u.deg

    # use groupby operations to calculate the percentile in each bin
    by_bin = table.group_by("bin_index")

    index = by_bin.groups.keys["bin_index"]
    result["angular_resolution"][index] = by_bin["theta"].groups.aggregate(
        lambda x: np.percentile(x, 100 * ONE_SIGMA_PERCENTILE)
    )
    return result
