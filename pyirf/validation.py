import numpy as np
from .utils import cone_solid_angle

class InvalidIRF(Exception):
    pass

class InvalidNormalisation(InvalidIRF):
    pass

def validate_psf(psf,source_offset_bins):
    
    radbin_solidangle = np.diff(cone_solid_angle(source_offset_bins))
    integral = np.sum(psf*radbin_solidangle,axis=2)
    
    if not np.allclose(integral,1):
        nbad = np.count_nonzero(~np.isclose(integral,1))
        raise InvalidNormalisation(f"PSF not correctly normalised: {nbad} affected bins")
    