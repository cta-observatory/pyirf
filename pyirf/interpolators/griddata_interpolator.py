"""
Simple wrapper around scipy.interpolate.griddata to interpolate parametrized quantities
"""

from .base_interpolators import ParametrizedInterpolator
from scipy.interpolate import griddata
import numpy as np


class GridDataInterpolator(ParametrizedInterpolator):
    def __init__(self, grid_points, params):
        """GridDataInterpolator constructor

        Args:
            grid_points (np.ndarray): Grid points at which interpolation templates exist
            params (np.ndarray): Corresponding parameter values at each point in
                grid_points. First dimesion has to correspond to number of grid_points

        Raises:
            TypeError: When params is not a np.ndarray
            ValueError: When number of points grid_points and params is not matching
        """
        super().__init__(grid_points, params)

    def _interpolate(self, target_point, **kwargs):
        # Initiate return value as structured array by copying one input array over
        return_array = np.copy(self.params[0])

        for param_name in self.params.dtype.names:
            return_array[param_name] = griddata(
                self.grid_points, self.params[param_name], target_point, **kwargs
            )

        return return_array
