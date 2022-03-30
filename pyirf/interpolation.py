"""Functions for performing interpolation of IRF to the values read from the data."""

import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d, griddata
from pyirf.utils import cone_solid_angle
from pyirf.binning import bin_center

__all__ = [
    "interpolate_effective_area_per_energy_and_fov",
    "interpolate_energy_dispersion",
]


def cdf_values(binned_pdf):
    """
    Compute cdf values and assure they are normed to unity
    """
    cdfs = np.cumsum(binned_pdf, axis=-1)

    # assure the last cdf value is 1, ignore errors for empty pdfs as they are reset to 0 by nan_to_num
    with np.errstate(invalid="ignore"):
        cdfs = np.nan_to_num(cdfs / np.max(cdfs, axis=-1)[..., np.newaxis])

    return cdfs


def ppf_values(cdfs, bin_mids, m, mprime, quantiles):
    """
    Compute ppfs from cdfs and interpolate them to the desired interpolation point

    Parameters
    ----------
    cdfs: numpy.ndarray, shape=(N,...,M)
        Corresponding cdf-values for all quantiles

    bin_mids: numpy.ndarray, shape=(M)
        M bin-midss for which the cdf-values are known

    m: numpy.ndarray, shape=(N, O)
        Array of the N O-dimensional morphing parameter values corresponding to the N input templates. The pdf's quantiles
        are expected to vary linearly between these two reference points.

    m_prime: numpy.ndarray, shape=(O)
        Value for which the interpolation is performed (target point)

    quantiles: numpy.ndarray, shape=(L)
        L quantiles for which the ppf_values are known

    Returns
    -------
    ppfs: numpy.ndarray, shape=(1,...,L)
        Corresponding ppf-values for all quantiles at the target interpolation point
    """

    def cropped_interp(cdf):
        """
        create ppf-values through inverse interpolation of the cdf, avoiding repeating 0 and 1 entries
        around the first and last bins as well as repeating values due to zero bins
        in between filled bins. Both cases would result in division by zero errors when computing
        the interpolation polynom.
        """
        # Find last 0 and first 1 entry
        last_0 = np.max(np.arange(len(cdf))[cdf == 0]) if cdf[0] == 0 else 0
        first_1 = np.min(np.arange(len(cdf))[cdf == 1])

        # Predefine selection mask
        selection = np.ones(len(cdf), dtype=bool)

        # Keep only the first of subsequently matching values
        selection[1:] = cdf[1:] != cdf[:-1]

        # Keep only the last 0 and first 1 entry
        selection[:last_0] = False
        selection[last_0] = True
        selection[first_1 + 1 :] = False
        selection[first_1] = True

        # create ppf values from selected bins
        return interp1d(
            cdf[selection],
            bin_mids[selection],
            bounds_error=False,
            fill_value="extrapolate",
        )(quantiles)

    # create ppf values from cdf samples via interpolation of the cdfs
    # return nan for empty pdfs
    ppfs = np.apply_along_axis(
        lambda cdf: cropped_interp(cdf)
        if np.sum(cdf) > 0
        else np.full_like(quantiles, np.nan),
        -1,
        cdfs,
    )
    # nD interpolation of ppf values
    return griddata(m, ppfs, mprime)


def reconstruct_pdf_values(quantiles, ppfs, edges):
    """
    Reconstruct pdf from ppf and evaluate at desired points.

    Parameters
    ----------
    quantiles: numpy.ndarray, shape=(L)
        L quantiles for which the ppf_values are known

    ppfs: numpy.ndarray, shape=(1,...,L)
        Corresponding ppf-values for all quantiles

    edges: numpy.ndarray, shape=(M+1)
        Binning of the desired binned pdf

    Returns
    -------
    pdf_values: numpy.ndarray, shape=(1,...,M)
        Recomputed, binned pdf
    """
    # recalculate pdf values through numerical differentiation
    pdf_interpolant = np.nan_to_num(np.diff(quantiles) / np.diff(ppfs, axis=-1))

    # Unconventional solution to make this usable with np.apply_along_axis for readability
    # The ppf bin-mids are computed since the pdf-values are derived through derivation
    # from the ppf-values
    xyconcat = np.concatenate(
        (ppfs[..., :-1] + np.diff(ppfs) / 2, pdf_interpolant), axis=-1
    )

    # Interpolate pdf samples and evaluate at bin edges, weight with the bin_width to estimate
    # correct bin height via the midpoint rule formulation of the trapezoidal rule
    pdf_values = np.apply_along_axis(
        lambda xy: np.diff(edges)
        * np.nan_to_num(
            interp1d(
                xy[: int(len(xy) / 2)],
                xy[int(len(xy) / 2) :],
                bounds_error=False,
                fill_value=(0, 0),
            )(edges[:-1])
        ),
        -1,
        xyconcat,
    )

    return pdf_values


def norm_pdf(pdf_values):
    """
    Normalize binned_pdf to a sum of 1
    """
    norm = np.sum(pdf_values, axis=-1)

    # Ignore errors from rows without errors as they are set to 0 through nan_to_num
    with np.errstate(invalid="ignore"):
        normed_pdf_values = pdf_values / norm[..., np.newaxis]
    return np.nan_to_num(normed_pdf_values)


def interpolate_binned_pdf(edges, binned_pdf, m, mprime, axis, quantile_resolution):
    """
    Takes a grid of binned pdfs for a bunch of different parameters
    and interpolates it to given value of those parameters.
    This function provides an adapted version of the quantile interpolation introduced
    in [1].
    Instead of following this method closely, it implements different approaches to the
    steps shown in Fig. 5 of [1].

    Parameters
    ----------
    edges: numpy.ndarray, shape=(M+1)
        Common array of bin-edges (along the abscissal ("x") axis) for the M bins of the input templates

    binned_pdf: numpy.ndarray, shape=(N,...,M,...)
        Array of M bin-heights (along the ordinate ("y") axis) for each of the N input templates.
        The distributions to be interpolated (e.g. MigraEnerg for the IRFs Energy Dispersion) is expected to
        be given at the dimension specified by axis.

    m: numpy.ndarray, shape=(N, O)
        Array of the N O-dimensional morphing parameter values corresponding to the N input templates. The pdf's quantiles
        are expected to vary linearly between these two reference points.

    m_prime: numpy.ndarray, shape=(O)
        Value for which the interpolation is performed (target point)

    axis: int
        Axis along which the pdfs used for interpolation are located

    quantile_resolution: float
        Spacing between the quantiles that are computed in the interpolation. Defaults to 1e-3.

    Returns
    -------
    f_new: numpy.ndarray, shape=(1,...,M,...)
        Interpolated and binned pdf

    References
    ----------
    .. [1] B. E. Hollister and A. T. Pang (2013). Interpolation of Non-Gaussian Probability Distributions
           for Ensemble Visualization
           https://engineering.ucsc.edu/sites/default/files/technical-reports/UCSC-SOE-13-13.pdf
    """
    # To have the needed axis always at the last index, as the number of indices is
    # not safely propageted through the interpolation but the last element remains the last element
    binned_pdf = np.swapaxes(binned_pdf, axis, -1)

    # compute quantiles from quantile_resolution
    quantiles = np.linspace(0, 1, int(np.round(1 / quantile_resolution, decimals=0)))

    # compute bin-mids from bin-edges
    bin_mids = bin_center(edges)

    # compute cdf values
    cdfs = cdf_values(binned_pdf)

    # compute ppf values at interpolation point, determine quantile and interpolate quantiles steps of [1]
    ppfs = ppf_values(cdfs, bin_mids, m, mprime, quantiles)

    # compute pdf values for all bins, evaluate interpolant PDF values step of [1]
    pdf_values = reconstruct_pdf_values(quantiles, ppfs, edges)

    # Renormalize pdf
    normed_pdf_values = norm_pdf(pdf_values)

    # Re-swap axes
    return np.swapaxes(normed_pdf_values, axis, -1)


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
    edges, binned_pdf, m, mprime, axis, quantile_resolution=1e-3
):
    """
    Takes a grid of energy dispersions for a bunch of different parameters
    and interpolates it to given value of those parameters.
    """
    return interpolate_binned_pdf(
        edges, binned_pdf, m, mprime, axis, quantile_resolution
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
