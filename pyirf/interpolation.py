"""Functions for performing interpolation of IRF to the values read from the data."""

import numpy as np
import astropy.units as u
from scipy.interpolate import griddata, interp1d
from pyirf.utils import cone_solid_angle

__all__ = [
    "interpolate_effective_area_per_energy_and_fov",
    "interpolate_energy_dispersion",
]


def lookup(x, len_hist):
    """
    Function to look up the bin-value corresponding to a bin number.

    Parameters
    ----------
    x: numpy.ndarray, shape=(M+L)
    Concatenated array in which the first M values are the M bin-values for a histogram consisting of M bins and
    the following L values are L bin-numbers, where the histogram value should be looked up.

    len_hist: int
    Number (M) of histogram bins

    Returns
    -------
    looked_up_values: numpy.ndarray, shape=(M)
    Looked up bin-values
    """
    hist = x[:len_hist]
    binnr = x[len_hist:]

    mask = (binnr == 0) | (binnr == len_hist + 1)

    looked_up_values = np.zeros(len(x) - len_hist)
    looked_up_values[~mask] = hist[(binnr[~mask] - 1).astype(int)]

    return looked_up_values


def numerical_quantile(pdf, mids, percentile):
    """
    Approximative quantile computation for histograms

    Parameters
    ----------
    pdf: numpy.ndarray, shape=(N)
    Input histogram

    mids: numpy.ndarray, shape=(M)
    Bin mids for the input histogram

    percentile: numpy.ndarray, shape=(L)
    Percentages where the quantile is needed

    Returns
    -------
    quantiles: numpy.ndarray, shape=(L)
    Computed quantiles

    """
    if np.sum(pdf) == 0:
        return np.full(len(percentile), np.nan)

    # Compute cdf and interpolate for continuity, Normalize to one first, since the sum of a histogram does not
    # need to be one
    cdf = np.cumsum(pdf)
    interp = interp1d(mids, cdf / cdf.max(), fill_value=(0, 1))

    # Create more continious version of the cdf
    x_range = np.linspace(mids.min(), mids.max(), 10000)
    y_new = interp(x_range)

    # Lookup when the cdf first lies above a certain percentile
    return [x_range[y_new >= perc][0] for perc in percentile]


def rebin(entries, mids, width):
    """
    Sort L pdf-values into M bins by computing the mean of all entries falling into a bin

    Parameters
    ----------
    entries: numpy.ndarray, shape=(2*L)
    Concatenated array in which the first L values are the x-values and
    the following L values pdf(x). This is purely done to make this function usable with np.apply_along_axis.

    mids: numpy.ndarray, shape=(M)
    Bin mids for the desired histogram

    width: float
    Bin width for the desired histogram, all bins are assumed to have the same bin-width

    Returns
    -------
    hist: numpy.ndarray, shape=(M)
    Histogram
    """

    # Decompose input
    x = entries[: int(len(entries) / 2)]
    y = entries[int(len(entries) / 2) :]

    # Put into bins, replace nans by 0 that arise in those bins where no entries lie (np.mean([]) = np.nan)
    return np.nan_to_num(
        [
            np.mean(y[(x >= low) & (x < up)])
            for low, up in zip(mids - width / 2, mids + width / 2)
        ]
    )


def interp_hist_quantile(edges, hists, m, m_prime, axis, normalize):
    """
    Function that wraps up the quantile PDF interpolation procedure [1] adopted for histograms.

    Parameters
    ----------
    edges: numpy.ndarray, shape=(M+1)
    Common array of bin-edges (along the abscissal ("x") axis) for the M bins of the input templates

    hists: numpy.ndarray, shape=(N,...,M,...)
    Array of M bin-heights (along the ordinate ("y") axis) for each of the N input templates.
    The distributions to be interpolated (e.g. MigraEnerg for the IRFs Energy Dispersion) is expected to
    be given at the dimension specified by axis.

    m: numpy.ndarray, shape=(N)
    Array of the N morphing parameter values corresponding to the N input templates

    m_prime: float
    Value at which the histogram should be interpolated

    axis: int
    Axis along which the pdfs used for interpolation are located

    normalize: string or None
    Mode of normalisation to account for the approximative nature of the interpolation. "sum" normalizes
    the interpolated histogram to a sum of 1, "weighted_sum" to an integral of 1. None does not apply any
    normalization.

    Returns
    -------
    f_new: numpy.ndarray, shape=(N,...,M,...)
    Interpolated histograms

    References
    ----------
    .. [1] B. E. Hollister and A. T. Pang (2013). Interpolation of Non-Gaussian Probability Distributions
           for Ensemble Visualization
           https://engineering.ucsc.edu/sites/default/files/technical-reports/UCSC-SOE-13-13.pdf
    """
    # determine quantiles step
    percentages = np.linspace(0, 1, 1000)
    mids = (edges[:-1] + edges[1:]) / 2
    quantiles = np.apply_along_axis(
        numerical_quantile, axis, hists, *[mids, percentages]
    )

    # interpolate quantiles step
    # First: compute alpha from eq. (6), the euclidean norm between the two grid points has to be one,
    # so normalize to ||m1-m0|| first
    dist = np.linalg.norm(m[1] - m[0])
    g = m / (m[1] - m[0])
    p = m_prime / (m[1] - m[0])
    alpha = np.linalg.norm(p - g[0])

    # Second: Interpolate quantiles as in eq. (10)
    q_bar = (1 - alpha) * quantiles[0] + alpha * quantiles[1]

    # evaluate interpolant PDF values step
    # The original PDF (only given as histogram) has to be re-evaluated at the quantiles determined above
    binnr = np.digitize(quantiles, edges)
    helper = np.concatenate((hists, binnr), axis=axis)
    V = np.apply_along_axis(lookup, axis, helper, *[hists.shape[axis]])

    # Compute the interpolated histogram at positions q_bar
    V_bar = V[0] * V[1] / ((1 - alpha) * V[1] + alpha * V[0])

    # Create temporal axis to imitate the former shape, as one dimension was lost through the interpolation and
    # therefore axis might not longer be correct
    q_bar = q_bar[None, :]
    V_bar = V_bar[None, :]

    # Shift interpolated pdf back into the original histogram binsm as V_bar is by construction given at
    # positions q_bar.
    width = np.diff(edges)[0]
    helper = np.concatenate((q_bar, V_bar), axis=axis)
    interpolated_histogram = np.apply_along_axis(rebin, axis, helper, *[mids, width])

    # Re-Normalize, as the normalisation is lost due to approximate nature of this method
    if normalize == "sum":
        norm = np.sum(interpolated_histogram, axis=axis)
        norm = np.repeat(
            np.expand_dims(norm, axis), interpolated_histogram.shape[axis], axis=axis
        )
    elif normalize == "weighted_sum":
        norm = np.sum(interpolated_histogram * width, axis=axis)
        norm = np.repeat(
            np.expand_dims(norm, axis), interpolated_histogram.shape[axis], axis=axis
        )
    elif normalize is None:
        norm = 1

    # Normalize and squeeze the temporal axis from the result
    return np.nan_to_num(interpolated_histogram / norm).squeeze()


@u.quantity_input(effective_area=u.m ** 2)
def interpolate_effective_area_per_energy_and_fov(
    effective_area,
    grid_points,
    target_point,
    min_effective_area=1.0 * u.Unit("m2"),
    method="linear",
):
    """
    Takes a grid of effective areas for a bunch of different parameters
    and interpolates (log) effective areas to given value of those parameters

    Parameters
    ----------
    effective_area: np.array of astropy.units.Quantity[area]
        grid of effective area, of shape (n_grid_points, n_fov_offset_bins, n_energy_bins)
    grid_points: np.array
        list of parameters corresponding to effective_area, of shape (n_grid_points, n_interp_dim)
    target_point: np.array
        values of parameters for which the interpolation is performed, of shape (n_interp_dim)
    min_effective_area: astropy.units.Quantity[area]
        Minimum value of effective area to be considered for interpolation
    method: 'linear’, ‘nearest’, ‘cubic’
        Interpolation method

    Returns
    -------
    aeff_interp: astropy.units.Quantity[area]
        Interpolated Effective area array with shape (n_energy_bins, n_fov_offset_bins)
    """

    # get rid of units
    effective_area = effective_area.to_value(u.m ** 2)
    min_effective_area = min_effective_area.to_value(u.m ** 2)

    # remove zeros and log it
    effective_area[effective_area < min_effective_area] = min_effective_area
    effective_area = np.log(effective_area)

    # interpolation
    aeff_interp = griddata(grid_points, effective_area, target_point, method=method).T
    # exp it and set to zero too low values
    aeff_interp = np.exp(aeff_interp)
    aeff_interp[
        aeff_interp < min_effective_area * 1.1
    ] = 0  # 1.1 to correct for numerical uncertainty and interpolation
    return u.Quantity(aeff_interp, u.m ** 2, copy=False)


def interpolate_energy_dispersion(
    energy_dispersions, bin_edges, grid_points, target_point, axis, normalize
):
    """
    Takes a grid of dispersion matrixes for a bunch of different parameters
    and interpolates it to given value of those parameters

    Parameters
    ----------
    energy_dispersions: np.ndarray
        grid of energy migrations
    bin_edges: np.ndarray
        bin edges of the energy migrations, have to be equidistant
    grid_points: np.ndarray
        array of parameters corresponding to energy_dispersions, of shape (n_grid_points, n_interp_dim)
    target_point: np.ndarray
        values of parameters for which the interpolation is performed, of shape (n_interp_dim)
    axis: int
        axis along which the energy-migration pdfs used for interpolation are located
    normalize: str
        Mode of normalisation to account for the approximative nature of the interpolation. "sum" normalizes
        the interpolated histogram to a sum of 1, "weighted_sum" to an integral of 1. None does not apply any
        normalization.

    Returns
    -------
    matrix_interp: np.ndarray
        Interpolated dispersion matrix 3D array with shape (n_energy_bins, n_migration_bins, n_fov_offset_bins)
    """

    matrix_interp = interp_hist_quantile(
        bin_edges, energy_dispersions, grid_points, target_point, axis, normalize
    )
    return matrix_interp


def interpolate_psf_table(
    psfs,
    grid_points,
    target_point,
    source_offset_bins,
    cumulative=False,
    method="linear",
):
    """
    Takes a grid of PSF tables for a bunch of different parameters
    and interpolates it to given value of those parameters

    Parameters
    ----------
    psfs: np.ndarray of astropy.units.Quantity
        grid of PSF tables, of shape (n_grid_points, n_energy_bins, n_fov_offset_bins, n_source_offset_bins)
    grid_points: np.ndarray
        array of parameters corresponding to energy_dispersions, of shape (n_grid_points, n_interp_dim)
    target_point: np.ndarray
        values of parameters for which the interpolation is performed, of shape (n_interp_dim)
    source_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the source offset (used for normalization)
    cumulative: bool
        If false interpolation is done directly on PSF bins, if true first cumulative distribution is computed
    method: 'linear’, ‘nearest’, ‘cubic’
        Interpolation method

    Returns
    -------
    psf_interp: np.ndarray
        Interpolated PSF table with shape (n_energy_bins,  n_fov_offset_bins, n_source_offset_bins)
    """

    # interpolation (stripping units first)
    if cumulative:
        psfs = np.cumsum(psfs, axis=3)

    psf_interp = griddata(
        grid_points, psfs.to_value("sr-1"), target_point, method=method
    ) * u.Unit("sr-1")

    if cumulative:
        psf_interp = np.concatenate(
            (psf_interp[..., :1], np.diff(psf_interp, axis=2)), axis=2
        )

    # now we need to renormalize along the source offset axis
    omegas = np.diff(cone_solid_angle(source_offset_bins))
    norm = np.sum(psf_interp * omegas, axis=2, keepdims=True)
    # By using out and where, it is ensured that columns with norm = 0 will have 0 values without raising an invalid value warning
    psf_norm = np.divide(
        psf_interp, norm, out=np.zeros_like(psf_interp), where=norm != 0
    )
    return psf_norm
