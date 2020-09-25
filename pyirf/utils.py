import numpy as np
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation


def is_scalar(val):
    '''Workaround that also supports astropy quantities'''
    return np.array(val, copy=False).shape == tuple()


def calculate_theta(events, assumed_source_az, assumed_source_alt):
    theta = angular_separation(
        assumed_source_az, assumed_source_alt,
        events['reco_az'], events['reco_alt'],
    )

    return theta.to(u.deg)


def calculate_source_fov_offset(events):
    theta = angular_separation(
        events['true_az'], events['true_alt'],
        events['pointing_az'], events['pointing_alt'],
    )

    return theta.to(u.deg)


def check_histograms(hist1, hist2, key='reco_energy'):
    '''
    Check if two histogram tables have the same binning

    Parameters
    ----------
    hist1: ``~astropy.table.Table``
        First histogram table, as created by ``~pyirf.binning.create_histogram_table``
    hist2: ``~astropy.table.Table``
        Second histogram table
    '''

    # check binning information and add to output
    for k in ('low', 'center', 'high'):
        k = key + '_' + k
        if not np.all(hist1[k] == hist2[k]):
            raise ValueError(
                'Binning for signal_hist and background_hist must be equal'
            )


def cone_solid_angle(angle):
    '''Calculate the solid angle of a view cone with opening angle ``angle``.'''
    return 2 * np.pi * (1 - np.cos(angle)) * u.sr
