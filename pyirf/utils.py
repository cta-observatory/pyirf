import numpy as np
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation


def is_scalar(val):
    '''Workaround that also supports astropy quantities'''
    return np.array(val, copy=False).shape == tuple()


def calculate_theta(events):
    theta = angular_separation(
        events['true_az'], events['true_alt'],
        events['reco_az'], events['reco_alt'],
    )

    return theta.to(u.deg)
