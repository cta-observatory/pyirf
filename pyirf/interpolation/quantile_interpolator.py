import numpy as np
from scipy.interpolate import griddata, interp1d

from .base_interpolators import DiscretePDFInterpolator, PDFNormalization
from .utils import get_bin_width

__all__ = ["QuantileInterpolator"]


def cdf_values(binned_pdf, bin_edges, normalization):
    """
    compute cdf values and assure they are normed to unity
    """
    bin_widths = get_bin_width(bin_edges, normalization)
    cdfs = np.cumsum(binned_pdf * bin_widths, axis=-1)

    # assure the last cdf value is 1, ignore errors for empty pdfs as they are reset to 0 by nan_to_num
    with np.errstate(invalid="ignore"):
        cdfs = np.nan_to_num(cdfs / np.max(cdfs, axis=-1)[..., np.newaxis])

    return cdfs


def ppf_values(bin_mids, cdfs, quantiles):
    """
    Compute ppfs from cdfs and interpolate them to the desired interpolation point

    Parameters
    ----------
    bin_mids: numpy.ndarray, shape=(M)
        Bin-mids for each bin along interpolation axis

    cdfs: numpy.ndarray, shape=(N,...,M)
        Corresponding cdf-values for all quantiles

    quantiles: numpy.ndarray, shape=(L)
        Quantiles for which the ppf-values should be estimated

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
        last_0 = np.nonzero(cdf == 0)[0][-1] if cdf[0] == 0 else 0
        first_1 = np.nonzero(cdf == 1.0)[0][0]

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
    return ppfs


def pdf_from_ppf(bin_edges, interp_ppfs, quantiles):
    """
    Reconstruct pdf from ppf and evaluate at desired points.

    Parameters
    ----------
    bin_edges: numpy.ndarray, shape=(M+1)
        Edges of the bins in which the final pdf should be binned

    interp_ppfs: numpy.ndarray, shape=(1,...,L)
        Corresponding ppf-values for all quantiles at the target_point,
        not to be confused with QunatileInterpolators self.ppfs, the ppfs
        computed from the input distributions.

    quantiles: numpy.ndarray, shape=(L)
        Quantiles corresponding to the ppf-values in interp_ppfs

    Returns
    -------
    pdf_values: numpy.ndarray, shape=(1,...,M)
        Recomputed, binned pdf at target_point
    """
    # recalculate pdf values through numerical differentiation
    pdf_interpolant = np.nan_to_num(np.diff(quantiles) / np.diff(interp_ppfs, axis=-1))

    # Unconventional solution to make this usable with np.apply_along_axis for readability
    # The ppf bin-mids are computed since the pdf-values are derived through derivation
    # from the ppf-values
    xyconcat = np.concatenate(
        (interp_ppfs[..., :-1] + np.diff(interp_ppfs) / 2, pdf_interpolant), axis=-1
    )

    def interpolate_ppf(xy):
        ppf = xy[: len(xy) // 2]
        pdf = xy[len(xy) // 2 :]
        interpolate = interp1d(ppf, pdf, bounds_error=False, fill_value=(0, 0))
        result = np.nan_to_num(interpolate(bin_edges[:-1]))
        return result

    # Interpolate pdf samples and evaluate at bin edges, weight with the bin_width to estimate
    # correct bin height via the midpoint rule formulation of the trapezoidal rule
    pdf_values = np.apply_along_axis(interpolate_ppf, -1, xyconcat)

    return pdf_values


def norm_pdf(pdf_values, bin_edges, normalization):
    """
    Normalize binned_pdf to a sum of 1
    """
    norm = np.sum(pdf_values, axis=-1)
    width = get_bin_width(bin_edges, normalization)

    # Norm all binned_pdfs to unity that are not empty
    normed_pdf_values = np.divide(
        pdf_values,
        norm[..., np.newaxis] * width,
        out=np.zeros_like(pdf_values),
        where=norm[..., np.newaxis] != 0,
    )

    return normed_pdf_values


class QuantileInterpolator(DiscretePDFInterpolator):
    def __init__(
        self,
        grid_points,
        bin_edges,
        binned_pdf,
        quantile_resolution=1e-3,
        normalization=PDFNormalization.AREA,
    ):
        """BinnedInterpolator constructor

        Parameters
        ----------
        grid_points : np.ndarray
            Grid points at which interpolation templates exist
        bin_edges : np.ndarray
            Edges of the data binning
        binned_pdf : np.ndarray
            Content of each bin in bin_edges for
            each point in grid_points. First dimesion has to correspond to number
            of grid_points, the last axis has to correspond to number
            of bins for the quantity that should be interpolated
            (e.g. the Migra axis for EDisp)
        quantile_resolution : float
            Spacing between quantiles
        normalization : PDFNormalization
            How the discrete PDF is normalized

        Raises
        ------
        ValueError:
            When last axis in binned_pdf and number of bins are not equal.

        Note
        ----
            Also calls __init__ of pyirf.interpolation.BaseInterpolator and
            DiscretePDFInterpolator
        """
        # Remember input shape
        self.input_shape = binned_pdf.shape

        if self.input_shape[-1] != len(bin_edges) - 1:
            raise ValueError(
                "Number of bins along last axis and those specified by bin_edges not matching."
            )

        # include 0-bin at first position in each pdf to avoid edge-effects where the CDF would otherwise
        # start at a value != 0, also extend edges with one bin to the left
        fill_zeros = np.zeros(shape=binned_pdf.shape[:-1])[..., np.newaxis]
        binned_pdf = np.concatenate((fill_zeros, binned_pdf), axis=-1)
        bin_edges = np.append(bin_edges[0] - np.diff(bin_edges)[0], bin_edges)

        # compute quantiles from quantile_resolution
        self.quantiles = np.linspace(
            0, 1, int(np.round(1 / quantile_resolution, decimals=0))
        )

        super().__init__(
            grid_points=grid_points,
            bin_edges=bin_edges,
            binned_pdf=binned_pdf,
            normalization=normalization,
        )

        # Compute CDF values
        self.cdfs = cdf_values(self.binned_pdf, self.bin_edges, self.normalization)

        # compute ppf values at quantiles, determine quantile step of [1]
        self.ppfs = ppf_values(self.bin_mids, self.cdfs, self.quantiles)

    def interpolate(self, target_point):
        """
        Takes a grid of binned pdfs for a bunch of different parameters
        and interpolates it to given value of those parameters.
        This function provides an adapted version of the quantile interpolation introduced
        in [1].
        Instead of following this method closely, it implements different approaches to the
        steps shown in Fig. 5 of [1].

        Parameters
        ----------
        target_point: numpy.ndarray, shape=(O)
            Value for which the interpolation is performed (target point)

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

        # interpolate ppfs to target point, interpolate quantiles step of [1]
        interpolated_ppfs = griddata(self.grid_points, self.ppfs, target_point)

        # compute pdf values for all bins, evaluate interpolant PDF values step of [1], drop the earlier
        # introduced extra bin
        interpolated_pdfs = pdf_from_ppf(
            self.bin_edges, interpolated_ppfs, self.quantiles
        )[..., 1:]

        # Renormalize pdf to sum of 1
        normed_interpolated_pdfs = norm_pdf(interpolated_pdfs, self.bin_edges[1:], self.normalization)

        # Re-swap axes and set all nans to zero
        return np.nan_to_num(normed_interpolated_pdfs).reshape(1, *self.input_shape[1:])
