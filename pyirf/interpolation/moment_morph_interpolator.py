from abc import ABCMeta, abstractmethod

import numpy as np

from .base_interpolators import BinnedInterpolator

__all__ = [
    "BaseMomentMorphInterpolator",
    "Base2DTriangularMomentMorphInterpolator",
    "Base1DMomentMorphInterpolator",
    "MomentMorphInterpolator",
]


class BaseMomentMorphInterpolator(BinnedInterpolator, metaclass=ABCMeta):
    def __init__(self, grid_points, bin_edges, bin_contents):
        super().__init__(grid_points, bin_edges, bin_contents)

    @abstractmethod
    def get_interpolation_coefficients(self, target_point):
        """Overridable function creating interpolation coefficients"""

    def _estimate_mean_std(self):
        """
        Function to roughly estimate mean and standart deviation from a histogram.

        Returns
        -------
        mean: numpy.ndarray, shape=(N)
            Estimated mean for each input template

        std: numpy.ndarray, shape=(N)
            Estimated standart deviation for each input template. Set to width/2 if only one bin in
            the input template is =/= 0
        """
        # Create an 2darray where the 1darray mids is repeated n_template times
        # mids = np.outer(np.ones(self.bin_contents.shape[0]), self.bin_mids)
        mids = np.broadcast_to(self.bin_mids, self.bin_contents.shape)
        # Weighted averages to compute mean and std
        # mean = np.average(mids, weights=self.bin_contents, axis=-1)
        # std = np.sqrt(np.average((mids.T - mean).T**2, weights=self.bin_contents, axis=-1))
        mean = np.average(mids, weights=self.bin_contents, axis=-1)
        std = np.sqrt(
            np.average(
                (mids - mean[..., np.newaxis]) ** 2, weights=self.bin_contents, axis=-1
            )
        )

        # Set std to 0.5*width for all those templates that have only one bin =/= 0. In those
        # cases mids-mean = 0 and therefore std = 0. Uses the width of the one bin with
        # bin_content!=0
        mask = std == 0
        if np.any(mask):
            width = np.diff(self.bin_edges) / 2
            std[mask] = width[self.bin_contents[mask, :] != 0]

        return mean, std

    def _lookup(self, x):
        """
        Function to return the bin-height at a desired point.

        Parameters
        ----------
        x: numpy.ndarray, shape=(N,M)
            Array of M points for each input template, where the histogram-value (bin-height) should be found

        Returns
        -------
        y: numpy.ndarray, shape=(N,M)
            Array of the bin-heights at the M points x, set to 0 at each point outside the histogram

        """
        # Create a flattend version of self.bin_contents to ease broadcasting
        intermediate_bin_contentents = self.bin_contents.reshape(
            -1, self.bin_contents.shape[-1]
        )

        # Find the bin where each point x is located in
        binnr = np.digitize(x, self.bin_edges).reshape(-1, self.bin_contents.shape[-1])

        # np.digitize returns 0 if x is below the first and len(bins) if x is above the last bin,
        # set these to 0 in a copy of the binnr to avoid errors later
        binnr_copy = np.copy(binnr)
        binnr_copy = np.where(
            (binnr == 0) | (binnr == len(self.bin_edges)), 0, binnr_copy
        )

        # Loop over every combination of input histograms and binning
        lu = np.array(
            [
                cont[binnr]
                for cont, binnr in zip(intermediate_bin_contentents, binnr_copy - 1)
            ]
        )

        # Set under-/ overflowbins to 0, reshape to original shape
        return np.where((binnr > 0) & (binnr < len(self.bin_edges)), lu, 0).reshape(
            self.bin_contents.shape
        )

    def interpolate(self, target_point, **kwargs):
        """
        Function that wraps up the moment morph procedure [1] adopted for histograms.

        Parameters
        ----------
        target_point: numpy.ndarray, shape=(N, 2)
        Value at which the histogram should be interpolated


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
        # Catch all those templates, where at least one template histogram is all zeros.
        zero_templates = ~np.all(~np.isclose(np.sum(self.bin_contents, axis=-1), 0), 0)

        # Manipulate those templates so that computations pass without error
        self.bin_contents[:, zero_templates] = np.full(
            len(self.bin_mids), 1 / len(self.bin_mids)
        )

        # Get interpolation coefficients as in eq. (6) in [1]
        cs = self.get_interpolation_coefficients(target_point)

        # Estimate mean and std for each input template histogram. First adaption needed to extend
        # the moment morph procedure to histograms
        mus, sigs = self._estimate_mean_std()
        cs = cs.reshape(self.n_points, *np.ones(mus.ndim - 1, "int"))

        # Transform mean and std as in eq. (11) and (12) in [1]
        # cs = np.broadcast_to(cs, mus.shape)
        mu_strich = np.sum(cs * mus, axis=0)
        sig_strich = np.sum(cs * sigs, axis=0)

        # Compute slope and offset as in eq. (14) and (15) in [1]
        aij = sigs / sig_strich
        bij = mus - mu_strich * aij

        # Transformation as in eq. (13) in [1]
        mids = np.broadcast_to(self.bin_mids, self.bin_contents.shape)
        transf_mids = aij[..., np.newaxis] * mids + bij[..., np.newaxis]

        # Compute the morphed historgram according to eq. (18) in [1]. The function "lookup" "resamples"
        # the histogram at the transformed bin-mids by using the templates historgam value at the transformed
        # bin-mid as new value for a whole transformed bin. Second adaption needed to extend
        # the moment morph procedure to histograms, adaptes the behaviour of eq. (16)

        transf_hist = self._lookup(transf_mids)

        f_new = np.sum(
            np.expand_dims(cs, -1) * transf_hist * np.expand_dims(cs, -1), axis=0
        )

        # Reset interpolation resolts for those templates with partially zero entries from above to 0
        f_new[zero_templates] = np.zeros(len(self.bin_mids))

        # If used normally, all values are stricktly positive, if used for extrapolation, some values can
        # become slightly negative as the mean/std estimation is not exact.
        f_new[f_new < 0] = 0

        # Re-Normalize, needed, as the estimation of the std used above is not exact but the result is scaled with
        # the estimated std
        norm = np.expand_dims(np.sum(f_new, axis=-1), -1)

        return np.divide(
            f_new, norm, out=np.zeros_like(f_new), where=norm != 0
        ).reshape(1, *self.bin_contents.shape[1:])


class Base2DTriangularMomentMorphInterpolator(BaseMomentMorphInterpolator):
    def __init__(self, grid_points, bin_edges, bin_contents):
        if grid_points.shape != (3, 2):
            raise ValueError("This base class can only interpolate in a triangle.")

        super().__init__(grid_points, bin_edges, bin_contents)

    def get_interpolation_coefficients(self, target_point):
        """
        Compute 2D interpolation coefficients for triangular interpolation,
        see e.g. [1]

        Parameters
        ----------
        target_point: numpy.ndarray, shape=(1, 2)
            Value at which the histogram should be interpolated

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
        d12 = self.grid_points[0, :] - self.grid_points[1, :]
        d13 = self.grid_points[0, :] - self.grid_points[2, :]
        d23 = self.grid_points[1, :] - self.grid_points[2, :]

        # Compute distance vector between target and third grid point
        dp3 = target_point.squeeze() - self.grid_points[2, :]

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


class Base1DMomentMorphInterpolator(BaseMomentMorphInterpolator):
    def __init__(self, grid_points, bin_edges, bin_contents):
        if grid_points.shape != (2, 1):
            raise ValueError("This base class can only interpolate between two points.")

        super().__init__(grid_points, bin_edges, bin_contents)

    def get_interpolation_coefficients(self, target_point):
        """
        Compute 1D interpolation coefficients for moment morph interpolation,
        as in eq. (6) of [1]

        Parameters
        ----------
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
        m0 = self.grid_points[0, :]

        # Compute matrix M as in eq. (2) of [1]
        j = np.arange(0, self.n_points, 1)
        m_ij = (self.grid_points - m0) ** j

        # Compute coefficients, eq. (6) from [1]
        return np.einsum(
            "...j, ji -> ...i", ((target_point - m0) ** j), np.linalg.inv(m_ij)
        )


class MomentMorphInterpolator(BinnedInterpolator):
    def __init__(self, grid_points, bin_edges, bin_contents):
        super().__init__(grid_points, bin_edges, bin_contents)

        if self.grid_dim > 2:
            raise NotImplementedError(
                "Interpolation in more then two dimension not impemented."
            )

    def _interpolate1D(self, target_point, **kwargs):
        target_bin = np.digitize(target_point.squeeze(), self.grid_points.squeeze())
        segment_inds = np.array([target_bin - 1, target_bin], "int")
        Interpolator = Base1DMomentMorphInterpolator(
            grid_points=self.grid_points[segment_inds],
            bin_edges=self.bin_edges,
            bin_contents=self.bin_contents[segment_inds],
        )

        return Interpolator(target_point, **kwargs)

    def _interpolate2D(self, target_point, **kwargs):
        simplex_inds = self.triangulation.simplices[
            self.triangulation.find_simplex(target_point)
        ].squeeze()
        Interpolator = Base2DTriangularMomentMorphInterpolator(
            grid_points=self.grid_points[simplex_inds],
            bin_edges=self.bin_edges,
            bin_contents=self.bin_contents[simplex_inds],
        )
        return Interpolator(target_point, **kwargs)

    def interpolate(self, target_point, **kwargs):
        if self.grid_dim == 1:
            return self._interpolate1D(target_point, **kwargs)
        elif self.grid_dim == 2:
            return self._interpolate2D(target_point, **kwargs)
