import numpy as np
from pyirf.binning import bin_center, calculate_bin_indices
from scipy.spatial import Delaunay

from .base_interpolators import DiscretePDFInterpolator, PDFNormalization
from .utils import get_bin_width

__all__ = [
    "MomentMorphInterpolator",
]


def _estimate_mean_std(bin_edges, binned_pdf, normalization):
    """
    Function to roughly estimate mean and standard deviation from a histogram.

    Parameters
    ----------
    bin_edges: np.ndarray, shape=(M+1)
        Array of common bin-edges for binned_pdf
    binned_pdf: np.ndarray, shape=(N, ..., M)
        PDF values from which to compute mean and std
    normalization : PDFNormalization
        How the PDF is normalized

    Returns
    -------
    mean: np.ndarray, shape=(N, ...)
        Estimated mean for each input template
    std: np.ndarray, shape=(N, ...)
        Estimated standard deviation for each input template. Set to width/2 if only one bin in
        the input template is =/= 0
    """
    # Create an 2darray where the 1darray mids is repeated n_template times
    mids = np.broadcast_to(bin_center(bin_edges), binned_pdf.shape)

    width = get_bin_width(bin_edges, normalization)

    # integrate pdf to get probability in each bin
    probability = binned_pdf * width
    # Weighted averages to compute mean and std
    mean = np.average(mids, weights=probability, axis=-1)
    std = np.sqrt(
        np.average((mids - mean[..., np.newaxis]) ** 2, weights=probability, axis=-1)
    )

    # Set std to 0.5*width for all those templates that have only one bin =/= 0. In those
    # cases mids-mean = 0 and therefore std = 0. Uses the width of the one bin with
    # binned_pdf!=0
    mask = std == 0
    if np.any(mask):
        width = np.diff(bin_edges)
        # std of a uniform distribution inside the bin
        uniform_std = np.broadcast_to(np.sqrt(1/12) * width, binned_pdf[mask].shape)
        std[mask] = uniform_std[binned_pdf[mask, :] != 0]

    return mean, std


def _lookup(bin_edges, binned_pdf, x):
    """
    Function to return the bin-height at a desired point.

    Parameters
    ----------
    bin_edges: np.ndarray, shape=(M+1)
        Array of common bin-edges for binned_pdf
    binned_pdf: np.ndarray, shape=(N, ..., M)
        Array of bin-entries, actual
    x: numpy.ndarray, shape=(N, ..., M)
        Array of M points for each input template, where the histogram-value (bin-height) should be found

    Returns
    -------
    y: numpy.ndarray, shape=(N, ..., M)
        Array of the bin-heights at the M points x, set to 0 at each point outside the histogram

    """
    # Find the bin where each point x is located in
    binnr, valid = calculate_bin_indices(x, bin_edges)

    # Set under/overflow-bins (invalid bins) to 0 to avoid errors below
    binnr[~valid] = 0

    # Loop over every combination of flattend input histograms and flattend binning
    lu = np.array(
        [
            cont[binnr_row]
            for cont, binnr_row in zip(
                binned_pdf.reshape(-1, binned_pdf.shape[-1]),
                binnr.reshape(-1, binnr.shape[-1]),
            )
        ]
    ).reshape(binned_pdf.shape)

    # Set all invalid bins to 0
    lu[~valid] = 0

    # Set under-/ overflowbins to 0, reshape to original shape
    return lu


def linesegment_1D_interpolation_coefficients(grid_points, target_point):
    """
    Compute 1D interpolation coefficients for moment morph interpolation,
    as in eq. (6) of [1]

    Parameters
    ----------
    grid_points: np.ndarray, shape=(2, 1)
        Points spanning a triangle in which
    target_point: numpy.ndarray, shape=(1, 1)
        Value at which the histogram should be interpolated

    Returns
    -------
    coefficients: numpy.ndarray, shape=(2,)
        Interpolation coefficients for all three interpolation simplex vertices
        to interpolate to the target_point

    References
    ----------
    .. [1] M. Baak, S. Gadatsch, R. Harrington and W. Verkerke (2015). Interpolation between
           multi-dimensional histograms using a new non-linear moment morphing method
           Nucl. Instrum. Methods Phys. Res. A 771, 39-48. https://doi.org/10.1016/j.nima.2014.10.033
    """
    # Set zeroth grid point as reference value
    m0 = grid_points[0, :]

    # Compute matrix M as in eq. (2) of [1]
    j = np.arange(0, grid_points.shape[0], 1)
    m_ij = (grid_points - m0) ** j

    # Compute coefficients, eq. (6) from [1]
    return np.einsum("...j, ji -> ...i", (target_point - m0) ** j, np.linalg.inv(m_ij))


def barycentric_2D_interpolation_coefficients(grid_points, target_point):
    """
    Compute barycentric 2D interpolation coefficients for triangular
    interpolation, see e.g. [1]

    Parameters
    ----------
    grid_points: np.ndarray, shape=(3, 2)
        Points spanning a triangle in which
    target_point: np.ndarray, shape=(1, 2)
        Value at which barycentric interpolation is needed

    Returns
    -------
    coefficients: numpy.ndarray, shape=(3,)
        Interpolation coefficients for all three interpolation simplex vertices
        to interpolate to the target_point

    References
    ----------
    .. [1] https://codeplea.com/triangular-interpolation
    """
    # Compute distance vectors between the grid points
    d13 = grid_points[0, :] - grid_points[2, :]
    d23 = grid_points[1, :] - grid_points[2, :]

    # Compute distance vector between target and third grid point
    dp3 = target_point.squeeze() - grid_points[2, :]

    # Compute first and second weight
    w1 = ((d23[1] * dp3[0]) + (-d23[0] * dp3[1])) / (
        (d23[1] * d13[0]) + (-d23[0] * d13[1])
    )
    w2 = ((-d13[1] * dp3[0]) + (d13[0] * dp3[1])) / (
        (d23[1] * d13[0]) + (-d23[0] * d13[1])
    )

    # Use w1+w2+w3 = 1 for third weight
    w3 = 1 - w1 - w2

    coefficients = np.array([w1, w2, w3])

    return coefficients


def moment_morph_estimation(bin_edges, binned_pdf, coefficients, normalization):
    """
    Function that wraps up the moment morph procedure [1] adopted for histograms.

    Parameters
    ----------
    bin_edges: np.ndarray, shape=(M+1)
        Array of common bin-edges for binned_pdf
    binned_pdf: np.ndarray, shape=(N, ..., M)
        Array of bin-entries, actual
    coefficients: np.ndarray, shape=(N)
        Estimation coefficients for each entry in binned_pdf
    normalization : PDFNormalization
        How the PDF is normalized

    Returns
    -------
    f_new: numpy.ndarray, shape=(1, M)
        Interpolated histogram

    References
    ----------
    .. [1] M. Baak, S. Gadatsch, R. Harrington and W. Verkerke (2015). Interpolation between
           multi-dimensional histograms using a new non-linear moment morphing method
           Nucl. Instrum. Methods Phys. Res. A 771, 39-48. https://doi.org/10.1016/j.nima.2014.10.033
    """
    bin_mids = bin_center(bin_edges)

    # Catch all those templates, where at least one template histogram is all zeros.
    zero_templates = ~np.all(~np.isclose(np.sum(binned_pdf, axis=-1), 0), 0)

    # Manipulate those templates so that computations pass without error
    binned_pdf[:, zero_templates] = np.full(len(bin_mids), 1 / len(bin_mids))

    # Estimate mean and std for each input template histogram. First adaption needed to extend
    # the moment morph procedure to histograms
    mus, sigs = _estimate_mean_std(bin_edges=bin_edges, binned_pdf=binned_pdf, normalization=normalization)
    coefficients = coefficients.reshape(
        binned_pdf.shape[0], *np.ones(mus.ndim - 1, "int")
    )

    # Transform mean and std as in eq. (11) and (12) in [1]
    # cs = np.broadcast_to(cs, mus.shape)
    mu_prime = np.sum(coefficients * mus, axis=0)
    sig_prime = np.sum(coefficients * sigs, axis=0)

    # Compute slope and offset as in eq. (14) and (15) in [1]
    aij = sigs / sig_prime
    bij = mus - mu_prime * aij

    # Transformation as in eq. (13) in [1]
    mids = np.broadcast_to(bin_mids, binned_pdf.shape)
    transf_mids = aij[..., np.newaxis] * mids + bij[..., np.newaxis]

    # Compute the morphed historgram according to eq. (18) in [1]. The function "lookup" "resamples"
    # the histogram at the transformed bin-mids by using the templates historgam value at the transformed
    # bin-mid as new value for a whole transformed bin. Second adaption needed to extend
    # the moment morph procedure to histograms, adaptes the behaviour of eq. (16)

    transf_hist = _lookup(bin_edges=bin_edges, binned_pdf=binned_pdf, x=transf_mids)

    f_new = np.sum(
        np.expand_dims(coefficients, -1) * transf_hist * np.expand_dims(aij, -1), axis=0
    )

    # Reset interpolation resolts for those templates with partially zero entries from above to 0
    f_new[zero_templates] = np.zeros(len(bin_mids))

    # Re-Normalize, needed, as the estimation of the std used above is not exact but the result is scaled with
    # the estimated std
    bin_widths = get_bin_width(bin_edges, normalization)
    norm = np.expand_dims(np.sum(f_new * bin_widths, axis=-1), -1)

    return np.divide(f_new, norm, out=np.zeros_like(f_new), where=norm != 0).reshape(
        1, *binned_pdf.shape[1:]
    )


class MomentMorphInterpolator(DiscretePDFInterpolator):
    def __init__(
        self, grid_points, bin_edges, binned_pdf, normalization=PDFNormalization.AREA
    ):
        """
        Interpolator class using moment morphing.

        Parameters
        ----------
        grid_points: np.ndarray, shape=(N, ...)
            Grid points at which interpolation templates exist. May be one ot two dimensional.
        bin_edges: np.ndarray, shape=(M+1)
            Edges of the data binning
        binned_pdf: np.ndarray, shape=(N, ..., M)
            Content of each bin in bin_edges for
            each point in grid_points. First dimesion has to correspond to number
            of grid_points. Interpolation dimension, meaning the
            the quantity that should be interpolated (e.g. the Migra axis for EDisp)
            has to be at axis specified by axis-keyword as well as having entries
            corresponding to the number of bins given through bin_edges keyword.
        normalization : PDFNormalization
            How the PDF is normalized

        Note
        ----
            Also calls pyirf.interpolation.DiscretePDFInterpolator.__init__.
        """
        super().__init__(grid_points, bin_edges, binned_pdf, normalization)

        if self.grid_dim == 2:
            self.triangulation = Delaunay(self.grid_points)
        elif self.grid_dim > 2:
            raise NotImplementedError(
                "Interpolation in more then two dimension not impemented."
            )

    def _interpolate1D(self, target_point):
        """
        Function to find target inside 1D self.grid_points and interpolate
        on this subset.
        """
        target_bin = np.digitize(target_point.squeeze(), self.grid_points.squeeze())
        segment_inds = np.array([target_bin - 1, target_bin], "int")
        coefficients = linesegment_1D_interpolation_coefficients(
            grid_points=self.grid_points[segment_inds],
            target_point=target_point,
        )

        return moment_morph_estimation(
            bin_edges=self.bin_edges,
            binned_pdf=self.binned_pdf[segment_inds],
            coefficients=coefficients,
            normalization=self.normalization,
        )

    def _interpolate2D(self, target_point):
        """
        Function to find target inside 2D self.grid_points and interpolate
        on this subset.
        """
        simplex_inds = self.triangulation.simplices[
            self.triangulation.find_simplex(target_point)
        ].squeeze()
        coefficients = barycentric_2D_interpolation_coefficients(
            grid_points=self.grid_points[simplex_inds],
            target_point=target_point,
        )

        return moment_morph_estimation(
            bin_edges=self.bin_edges,
            binned_pdf=self.binned_pdf[simplex_inds],
            coefficients=coefficients,
            normalization=self.normalization,
        )

    def interpolate(self, target_point):
        """
        Takes a grid of binned pdfs for a bunch of different parameters
        and interpolates it to given value of those parameters.
        This function calls implementations of the moment morphing interpolation
        pocedure introduced in [1].

        Parameters
        ----------
        target_point: numpy.ndarray
            Value for which the interpolation is performed (target point)

        Returns
        -------
        f_new: numpy.ndarray, shape=(1,...,M,...)
            Interpolated and binned pdf

        References
        ----------
        .. [1] M. Baak, S. Gadatsch, R. Harrington and W. Verkerke (2015). Interpolation between
               multi-dimensional histograms using a new non-linear moment morphing method
               Nucl. Instrum. Methods Phys. Res. A 771, 39-48. https://doi.org/10.1016/j.nima.2014.10.033
        """
        if self.grid_dim == 1:
            interpolant = self._interpolate1D(target_point)
        elif self.grid_dim == 2:
            interpolant = self._interpolate2D(target_point)

        return interpolant
