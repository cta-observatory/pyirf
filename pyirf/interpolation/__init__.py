"""
Collection of interpolation methods
"""

from .base_interpolators import (
    BaseInterpolator,
    BinnedInterpolator,
    ParametrizedInterpolator,
)
from .griddata_interpolator import GridDataInterpolator
from .interpolate_irfs import (
    interpolate_effective_area_per_energy_and_fov,
    interpolate_energy_dispersion,
    interpolate_psf_table,
    interpolate_rad_max,
)
from .quantile_interpolator import QuantileInterpolator

__all__ = [
    "BaseInterpolator",
    "BinnedInterpolator",
    "GridDataInterpolator",
    "ParametrizedInterpolator",
    "QuantileInterpolator",
    "interpolate_effective_area_per_energy_and_fov",
    "interpolate_energy_dispersion",
    "interpolate_psf_table",
    "interpolate_rad_max",
]
