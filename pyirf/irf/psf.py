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

    psf = _normalize_psf(hist, source_offset_bins)

    result = QTable({
        'true_energy_low': u.Quantity(true_energy_bins[:-1], ndmin=2),
        'true_energy_high': u.Quantity(true_energy_bins[1:], ndmin=2),
        'source_offset_low': u.Quantity(source_offset_bins[:-1], ndmin=2),
        'source_offset_high': u.Quantity(source_offset_bins[1:], ndmin=2),
        'fov_offset_low': u.Quantity(fov_offset_bins[:-1], ndmin=2),
        'fov_offset_high': u.Quantity(fov_offset_bins[1:], ndmin=2),
        'psf': [psf],
    })

    return result


def _normalize_psf(hist, source_offset_bins):
    '''Normalize the psf histogram to a probability densitity over solid angle'''
    solid_angle = np.diff(
        2 * np.pi
        * (1 - np.cos(source_offset_bins.to_value(u.rad))),
    ) * u.sr

    # ignore numpy zero division warning
    with np.errstate(invalid='ignore'):

        # to correctly divide by using broadcasting here,
        # we need to swap the axis order
        n_events = hist.sum(axis=2).T
        hist = np.swapaxes(hist, 0, 2)

        # normalize and replace nans with 0
        psf = np.nan_to_num(hist / n_events)

        # swap axes back to order required by GADF
        psf = np.swapaxes(psf, 0, 2)

    return psf / solid_angle
