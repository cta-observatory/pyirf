"""
Simple wrapper around scipy.interpolate.griddata to interpolate parametrized quantities
"""

from .base_interpolators import ParametrizedInterpolator
from scipy.interpolate import griddata
import numpy as np

__all__ = ["GridDataInterpolator"]


class GridDataInterpolator(ParametrizedInterpolator):
    def __init__(self, grid_points, params):
        """GridDataInterpolator constructor

        Parameters
        ----------
        grid_points: np.ndarray, shape=(N, O)
            Grid points at which interpolation templates exist
        params: np.ndarray, shape=(N, ...)
            Structured array of corresponding parameter values at each
            point in grid_points.
            First dimesion has to correspond to number of grid_points

        Raises
        ------
        TypeError:
            When params is not a np.ndarray
        ValueError:
            When number of points grid_points and params is not matching
        """
        super().__init__(grid_points, params)

    def interpolate(self, target_point, **kwargs):
        """
        Wrapper around scipy.interpolate.griddata [1]

        Parameters
        ----------
        target_point: np.ndarray, shape=(1, O)
            Target point for interpolation
        **kwargs:
            Passed to scipy.interpolate.griddata [1]

        Returns
        -------
        interpolant: np.ndarray

        References
        ----------
        .. [1] Scipy Documentation, scipy.interpolate.griddata
               https://docs.scipy.org/doc/scipy/reference/generated
        """
        # Initiate return value as structured array by copying one input array over
        return_array = np.copy(self.params[0])

        # Intependently interpolate all parameters
        for param_name in self.params.dtype.names:
            return_array[param_name] = griddata(
                self.grid_points, self.params[param_name], target_point, **kwargs
            )

        return return_array
