import numpy as np

from .base_interpolators import BaseInterpolator

__all__ = ["NearestNeighborSearcher"]


class NearestNeighborSearcher(BaseInterpolator):
    """
    Dummy NearestNeighbor approach usable instead of
    actual Interpolation/Extrapolation
    """

    def __init__(self, grid_points, contents, norm_ord=2):
        """
        NearestNeighborSearcher

        Parameters
        ----------
        grid_points: np.ndarray, shape=(n_points, n_dims)
            Grid points at which templates exist
        contents: np.ndarray, shape=(n_points, ...)
            Corresponding IRF contents at grid_points
        norm_ord: non-zero int
            Order of the norm which is used to compute the distances,
            passed to numpy.linalg.norm [1]. Defaults to 2,
            which uses the euclidean norm.

        Note
        ----
            Also calls pyirf.interpolation.BaseInterpolators.__init__
        """

        super().__init__(grid_points)

        self.contents = contents

        # Test wether norm_ord is a number
        try:
            norm_ord > 0
        except TypeError:
            raise ValueError(
                f"Only positiv integers allowed for norm_ord, got {norm_ord}."
            )

        # Test wether norm_ord is a finite, positive integer
        if (norm_ord <= 0) or ~np.isfinite(norm_ord) or (norm_ord != int(norm_ord)):
            raise ValueError(
                f"Only positiv integers allowed for norm_ord, got {norm_ord}."
            )

        self.norm_ord = norm_ord

    def interpolate(self, target_point):
        """
        Takes a grid of IRF contents for a bunch of different parameters
        and returns the contents at the nearest grid point
        as seen from the target point.

        Parameters
        ----------
        target_point: numpy.ndarray
            Value for which the nearest neighbor should be found (target point)

        Returns
        -------
        content_new: numpy.ndarray, shape=(1,...,M,...)
            Contents at nearest neighbor

        Note
        ----
            In case of multiple nearest neighbors, the contents corresponding
            to the first one are returned.
        """

        if target_point.ndim == 1:
            target_point = target_point.reshape(1, *target_point.shape)

        distances = np.linalg.norm(
            self.grid_points - target_point, ord=self.norm_ord, axis=1
        )

        index = np.argmin(distances)

        return self.contents[index, :]
