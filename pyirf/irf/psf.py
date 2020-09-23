import numpy as np
import astropy.units as u
from astropy.table import QTable

from astropy.coordinates.angle_utilities import angular_separation


def psf_table(events, true_energy_bins, source_offset_bins, fov_offset_bins):
    '''
    Calculate the table based PSF (radially symmetrical bins around the true source)
    '''

    source_fov_offset = angular_separation(
        events['true_az'], events['true_alt'],
        events['pointing_az'], events['pointing_alt'],
    )

    array = np.column_stack([
        events['true_energy'].to_value(u.TeV),
        source_fov_offset.to_value(u.deg),
        events['theta'].to_value(u.deg)
    ])

    hist, edges = np.histogramdd(
        array,
        [
            true_energy_bins.to_value(u.TeV),
            fov_offset_bins.to_value(u.deg),
            source_offset_bins.to_value(u.deg),
        ]
    )

    solid_angle = np.diff(
        2 * np.pi
        * (1 - np.cos(source_offset_bins.to_value(u.rad))),
    ) * u.sr

    # ignore numpy zero division warning
    with np.errstate(invalid='ignore'):
        # normalize and replace nans with 0
        psf = np.nan_to_num(hist / hist.sum(axis=2) / solid_angle)

    result = QTable({
        'true_energy_low': [true_energy_bins[:-1]],
        'true_energy_high': [true_energy_bins[1:]],
        'source_offset_low': [source_offset_bins[:-1]],
        'source_offset_high': [source_offset_bins[1:]],
        'fov_offset_low': [fov_offset_bins[:-1]],
        'fov_offset_high': [fov_offset_bins[1:]],
        'psf': [psf],
    })

    return result
