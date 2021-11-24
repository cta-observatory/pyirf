"""Functions for performing interpolation of IRF to the values read from the data."""

import numpy as np
import astropy.units as u
from scipy.interpolate import griddata, interp1d
from scipy.spatial import Delaunay
from pyirf.utils import cone_solid_angle
from pyirf.binning import bin_center

import warnings

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
    pdf: numpy.ndarray, shape=(2)
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

    # Compute cdf and interpolate for ppf, Normalize to one first, since the sum of a histogram does not
    # need to be one
    cdf = np.cumsum(pdf)
    interp = interp1d(
        cdf / cdf.max(), mids, fill_value=(mids.min(), mids.max()), bounds_error=False
    )

    # Lookup ppf value
    return interp(percentile)


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
    # Catch both warnings that arise when np.mean([]) is called, as this is anticipated.

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        warnings.filterwarnings(
            action="ignore", message="invalid value encountered in double_scalars"
        )
        rebinned_histogram = [
            np.mean(y[(x >= low) & (x < up)])
            for low, up in zip(mids - width / 2, mids + width / 2)
        ]

    return np.nan_to_num(rebinned_histogram)


def interp_hist_quantile(
    edges, hists, m, m_prime, axis, normalize, quantile_resolution=1e-3
):
    """
    Function that wraps up the quantile PDF interpolation procedure [1] adopted for histograms.

    Parameters
    ----------
    edges: numpy.ndarray, shape=(M+1)
    Common array of bin-edges (along the abscissal ("x") axis) for the M bins of the input templates

    hists: numpy.ndarray, shape=(2,...,M,...)
    Array of M bin-heights (along the ordinate ("y") axis) for each of the 2 input templates.
    The distributions to be interpolated (e.g. MigraEnerg for the IRFs Energy Dispersion) is expected to
    be given at the dimension specified by axis.

    m: numpy.ndarray, shape=(2)
    Array of the 2 morphing parameter values corresponding to the 2 input templates. The pdf's qunatiles
    are expected to vary linearly between these two reference points.

    m_prime: float
    Value for which the interpolation is performed (target point)

    axis: int
    Axis along which the pdfs used for interpolation are located

    normalize: string or None
    Mode of normalisation to account for the approximative nature of the interpolation. "sum" normalizes
    the interpolated histogram to a sum of 1, "weighted_sum" to an integral of 1. None does not apply any
    normalization.

    quantile_resolution: float
    Interpolated quantile spacing, defaults to 1/1000

    Returns
    -------
    f_new: numpy.ndarray, shape=(...,M,...)
    Interpolated histograms

    References
    ----------
    .. [1] B. E. Hollister and A. T. Pang (2013). Interpolation of Non-Gaussian Probability Distributions
           for Ensemble Visualization
           https://engineering.ucsc.edu/sites/default/files/technical-reports/UCSC-SOE-13-13.pdf
    """
    # determine quantiles step
    percentages = np.arange(0, 1 + 0.5 * quantile_resolution, quantile_resolution)
    mids = bin_center(edges)

    quantiles = np.apply_along_axis(numerical_quantile, axis, hists, mids, percentages)

    # interpolate quantiles step
    # First: compute alpha from eq. (6), the euclidean norm between the two grid points has to be one,
    # so normalize to ||m1-m0|| first
    dist = np.linalg.norm(m[1] - m[0])
    g = m / dist
    p = m_prime / dist
    alpha = np.linalg.norm(p - g[0])

    # Second: Interpolate quantiles as in eq. (10)
    q_bar = (1 - alpha) * quantiles[0] + alpha * quantiles[1]

    # evaluate interpolant PDF values step
    # The original PDF (only given as histogram) has to be re-evaluated at the quantiles determined above
    binnr = np.digitize(quantiles, edges)
    helper = np.concatenate((hists, binnr), axis=axis)
    V = np.apply_along_axis(lookup, axis, helper, hists.shape[axis])

    # Compute the interpolated histogram at positions q_bar as in eq. (12), set V_bar to
    # zero when both template PDFs are zero
    # V_bar = V[0] * V[1] / ((1 - alpha) * V[1] + alpha * V[0])
    a = V[0] * V[1]
    b = (1 - alpha) * V[1] + alpha * V[0]
    V_bar = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    # Create temporary axis to imitate the former shape, as one dimension was lost through the interpolation and
    # therefore axis might not longer be correct
    q_bar = q_bar[np.newaxis, :]
    V_bar = V_bar[np.newaxis, :]

    # Shift interpolated pdf back into the original histogram bins as V_bar is by construction given at
    # positions q_bar.
    width = np.diff(edges)
    helper = np.concatenate((q_bar, V_bar), axis=axis)
    interpolated_histogram = np.apply_along_axis(rebin, axis, helper, mids, width)

    # Re-Normalize, as the normalisation is lost due to approximate nature of this method
    # Set norm to nan for empty histograms to avoid division through 0
    if normalize == "sum":
        norm = np.sum(interpolated_histogram, axis=axis)
        norm = np.repeat(
            np.expand_dims(norm, axis), interpolated_histogram.shape[axis], axis=axis
        )
        norm[norm == 0] = np.nan
    elif normalize == "weighted_sum":
        norm = np.sum(interpolated_histogram * width, axis=axis)
        norm = np.repeat(
            np.expand_dims(norm, axis), interpolated_histogram.shape[axis], axis=axis
        )
        norm[norm == 0] = np.nan
    elif normalize is None:
        norm = 1

    # Normalize and squeeze the temporary axis from the result
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

    Raises
    ------
    ValueError if number of grid- and template-points is not matching

    ValueError if target_point is outside grid

    ValueError if grid dimension is > 2
    """
    # Test grid and energy_dispersions for matching dimensions
    if len(grid_points) != energy_dispersions.shape[0]:
        raise ValueError("Number of grid- and template-points not matching.")

    # If grid is nD with n>2: Raise Error.
    if (not np.isscalar(grid_points[0])) and (len(grid_points[0]) > 2):
        raise ValueError("Grid Dimension > 2")

    # If data is 1D: directly interpolate between next neighbors
    if np.isscalar(grid_points[0]):
        # Sort arrays to find the pair of next neighbors to the target_point
        sorting_indizes = np.argsort(grid_points)
        sorted_grid = grid_points[sorting_indizes]
        sorted_template = energy_dispersions[sorting_indizes]
        input_pos = np.digitize(target_point, sorted_grid)

        if (input_pos == len(grid_points)) or (input_pos == -1):
            raise ValueError("The target point lies outside the specified grid.")

        neighbors = np.array([input_pos - 1, input_pos])

        # Get matching pdfs and interpolate
        grid_point_subset = sorted_grid[neighbors]

        template_subset = sorted_template[neighbors]
        return interp_hist_quantile(
            bin_edges, template_subset, grid_point_subset, target_point, axis, normalize
        )

    # Else for 2D: Chain 1D interpolations to arive at the desired point

    # Find simplex (triangle in this 2D case) of grid-points in which the target point lies.
    # Use Delaunay-Transformation to costruct a triangular grid from grid_points.
    triangular_grid = Delaunay(grid_points)
    target_simplex = triangular_grid.find_simplex(target_point)
    if target_simplex == -1:
        raise ValueError("The target point lies outside the specified grid.")

    target_simplex_indices = triangular_grid.simplices[target_simplex]
    target_simplex_vertices = grid_points[target_simplex_indices]

    # Use the segment between the two vertices of the triangle that are closest to the target_point to construct
    # the intermediate point m_tilde. m_tilde will be the point where the line between
    # the most distant vertex and the target point crosses the remaining triangle side. This construction assures
    # minimal distance between m_tilde and the target point and should thus minimize interpolation error.
    distances = np.linalg.norm(target_point - target_simplex_vertices, axis=1)

    sorting_indizes = np.argsort(distances)
    sorted_vertices = target_simplex_vertices[sorting_indizes]
    sorted_indices = target_simplex_indices[sorting_indizes]

    # Construct m_tilde. This needs one to solve a problem of the form Ax = b
    A = np.array(
        [target_point - sorted_vertices[-1], -sorted_vertices[1] + sorted_vertices[0]]
    ).T
    b = sorted_vertices[0] - sorted_vertices[-1]
    m_tilde = sorted_vertices[-1] + np.linalg.solve(A, b)[0] * (
        target_point - sorted_vertices[-1]
    )

    # Interpolate to m_tilde
    template_subset = energy_dispersions[[sorted_indices[0], sorted_indices[1]], :]
    grid_point_subset = grid_points[[sorted_indices[0], sorted_indices[1]], :]
    interpolated_hist_tilde = interp_hist_quantile(
        bin_edges, template_subset, grid_point_subset, m_tilde, axis, normalize
    )

    # Interpolate to target_point, reshape as axis with length 1 are lost in the computation and the shapes would not be matching
    template_subset = np.array(
        [
            interpolated_hist_tilde.reshape(energy_dispersions.shape[1:]),
            energy_dispersions[sorted_indices[-1]],
        ]
    )
    grid_point_subset = np.array([m_tilde, grid_points[sorted_indices[-1]]])

    return interp_hist_quantile(
        bin_edges, template_subset, grid_point_subset, target_point, axis, normalize
    )


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
