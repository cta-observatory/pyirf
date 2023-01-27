"""
Collection of interpolation methods
"""

from .griddata_interpolator import GridDataInterpolator
from .quantile_interpolator import QuantileInterpolator

__all__ = ["GridDataInterpolator", "QuantileInterpolator"]
