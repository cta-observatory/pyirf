import numpy as np
import astropy.units as u

from ..utils import cone_solid_angle


def psf_table(events, true_energy_bins, source_offset_bins, fov_offset_bins):
    """
    Calculate the table based PSF (radially symmetrical bins around the true source)
    """

    array = np.column_stack(
        [
            events["true_energy"].to_value(u.TeV),
            events["source_fov_offset"].to_value(u.deg),
            events["theta"].to_value(u.deg),
        ]
    )

    hist, edges = np.histogramdd(
        array,
        [
            true_energy_bins.to_value(u.TeV),
            fov_offset_bins.to_value(u.deg),
            source_offset_bins.to_value(u.deg),
        ],
    )

    psf = _normalize_psf(hist, source_offset_bins)
    return psf


def _normalize_psf(hist, source_offset_bins):
    """Normalize the psf histogram to a probability densitity over solid angle"""
    solid_angle = np.diff(cone_solid_angle(source_offset_bins))

    # ignore numpy zero division warning
    with np.errstate(invalid="ignore"):

        # to correctly divide by using broadcasting here,
        # we need to swap the axis order
        n_events = hist.sum(axis=2).T
        hist = np.swapaxes(hist, 0, 2)

        # normalize and replace nans with 0
        psf = np.nan_to_num(hist / n_events)

        # swap axes back to order required by GADF
        psf = np.swapaxes(psf, 0, 2)

    return psf / solid_angle
