"""
Simple wrapper around scipy.interpolate.griddata to interpolate parametrized quantities
"""
from scipy.interpolate import griddata

from .base_interpolators import ParametrizedInterpolator

__all__ = ["GridDataInterpolator"]


class GridDataInterpolator(ParametrizedInterpolator):
    def __init__(self, grid_points, params, **griddata_kwargs):
        """Parametrized Interpolator using scipy.interpolate.griddata

        Parameters
        ----------
        grid_points: np.ndarray, shape=(N, O)
            Grid points at which interpolation templates exist
        params: np.ndarray, shape=(N, ...)
            Structured array of corresponding parameter values at each
            point in grid_points.
            First dimesion has to correspond to number of grid_points
        griddata_kwargs: dict
            Keyword-Arguments passed to scipy.griddata [1], e.g.
            interpolation method. Defaults to None, which uses scipy's
            defaults

        Raises
        ------
        TypeError:
            When params is not a np.ndarray
        ValueError:
            When number of points grid_points and params is not matching

        References
        ----------
        .. [1] Scipy Documentation, scipy.interpolate.griddata
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

        """
        super().__init__(grid_points, params)

        self.griddata_kwargs = griddata_kwargs

    def interpolate(self, target_point):
        """
        Wrapper around scipy.interpolate.griddata [1]

        Parameters
        ----------
        target_point: np.ndarray, shape=(1, O)
            Target point for interpolation

        Returns
        -------
        interpolant: np.ndarray

        References
        ----------
        .. [1] Scipy Documentation, scipy.interpolate.griddata
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
        """
        interpolant = griddata(
            self.grid_points, self.params, target_point, **self.griddata_kwargs
        ).squeeze()

        return interpolant.reshape(1, *self.params.shape[1:])
