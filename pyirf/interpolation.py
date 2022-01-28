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
    edges, hists, m, mprime, axis, quantile_resolution=1e-3
):
    """
    Takes a grid of dispersion matrixes for a bunch of different parameters
    and interpolates it to given value of those parameters.
    This function provides an adapted version of the quantile Interpolation introduced
    in [1].
    Instead of following this method closely, it implements different approaches to the
    steps shown in Fig. 5 of [1].

    Parameters
    ----------
    edges: numpy.ndarray, shape=(M+1)
        Common array of bin-edges (along the abscissal ("x") axis) for the M bins of the input templates

    hists: numpy.ndarray, shape=(N,...,M,...)
        Array of M bin-heights (along the ordinate ("y") axis) for each of the 2 input templates.
        The distributions to be interpolated (e.g. MigraEnerg for the IRFs Energy Dispersion) is expected to
        be given at the dimension specified by axis.

    m: numpy.ndarray, shape=(N, O)
        Array of the N O-dimensional morphing parameter values corresponding to the N input templates. The pdf's qunatiles
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
        Interpolated histograms

    References
    ----------
    .. [1] B. E. Hollister and A. T. Pang (2013). Interpolation of Non-Gaussian Probability Distributions
           for Ensemble Visualization
           https://engineering.ucsc.edu/sites/default/files/technical-reports/UCSC-SOE-13-13.pdf
    """
    # To have the needed axis always at the last index, as the number of indices is
    # not safely propageted through the interpolation but the last element remains the last element
    hists = np.swapaxes(hists, axis, -1)

    # Compute CDFs
    cdfs = np.cumsum(hists, axis=-1)

    cdfs = cdfs / np.expand_dims(np.max(cdfs, axis=-1), axis=-1)

    quantiles = np.linspace(0, 1, int(np.round(1 / quantile_resolution, decimals=0)))

    bin_mids = bin_center(edges)

    # create ppf values from cdf samples via interpolation of the cdfs, determine quantile steps of [1]
    ppfs_resampled = np.apply_along_axis(
        lambda cdf: interp1d(
            cdf, bin_mids, bounds_error=False, fill_value="extrapolate"
        )(quantiles),
        -1,
        cdfs,
    )

    # nD interpolation of ppf values, interpolate quantiles step of [1]
    ppf_interpolant = griddata(m, ppfs_resampled, mprime)

    # recalculate pdf values, evaluate interpolant PDF values step of [1]
    pdf_interpolant = np.diff(quantiles) / np.diff(ppf_interpolant, axis=-1)

    # Unconventional solution to make this usable with np.apply_along_axis for readability
    # The ppf bin-mids are computed since the pdf-values are derived through derivation
    # from the ppf-values
    xyconcat = np.concatenate(
        (
            ppf_interpolant[..., :-1] + np.diff(ppf_interpolant) / 2,
            np.nan_to_num(pdf_interpolant),
        ),
        axis=-1,
    )

    # Interpolate pdf samples and evaluate at bin edges, weight with the bin_width to estimate
    # correct bin height via the midpoint rule formulation of the trapezoidal rule
    result = np.apply_along_axis(
        lambda xy: np.diff(edges)
        * np.nan_to_num(
            interp1d(
                xy[: int(len(xy) / 2)],
                xy[int(len(xy) / 2) :],
                bounds_error=False,
                fill_value=(0, 0),
            )(bin_mids)
        ),
        -1,
        xyconcat,
    )

    # Renormalize histogram to a sum of 1
    norm = np.sum(result, axis=-1)
    norm = np.repeat(np.expand_dims(norm, -1), result.shape[-1], axis=-1)
    norm[norm == 0] = np.nan

    return np.swapaxes(np.nan_to_num(result / norm), axis, -1)


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
