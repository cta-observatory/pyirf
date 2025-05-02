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
            events["true_source_fov_offset"].to_value(u.deg),
            events["theta"].to_value(u.deg),
        ]
    )

    hist, _ = np.histogramdd(
        array,
        [
            true_energy_bins.to_value(u.TeV),
            fov_offset_bins.to_value(u.deg),
            source_offset_bins.to_value(u.deg),
        ],
    )

    psf = _normalize_psf(hist, source_offset_bins)
    return psf


def psf_table_3d_polar(
    events,
    true_energy_bins,
    source_offset_bins,
    fov_offset_bins,
    fov_position_angle_bins,
):
    """
    Calculate the table based PSF (radially symmetrical bins around the true source, offset and position angle bins in the FoV)
    """

    array = np.column_stack(
        [
            events["true_energy"].to_value(u.TeV),
            events["true_source_fov_offset"].to_value(u.deg),
            events["true_source_fov_position_angle"].to_value(u.deg),
            events["theta"].to_value(u.deg),
        ]
    )

    hist, _ = np.histogramdd(
        array,
        [
            true_energy_bins.to_value(u.TeV),
            fov_offset_bins.to_value(u.deg),
            fov_position_angle_bins.to_value(u.deg),
            source_offset_bins.to_value(u.deg),
        ],
    )

    psf = _normalize_psf(hist, source_offset_bins)
    return psf


def psf_table_3d_lonlat(
    events,
    true_energy_bins,
    source_offset_bins,
    fov_lon_bins,
    fov_lat_bins,
):
    """
    Calculate the table based PSF (radially symmetrical bins around the true source, lon/lat bins in the FoV)
    """

    array = np.column_stack(
        [
            events["true_energy"].to_value(u.TeV),
            events["true_source_fov_lon"].to_value(u.deg),
            events["true_source_fov_lat"].to_value(u.deg),
            events["theta"].to_value(u.deg),
        ]
    )

    hist, _ = np.histogramdd(
        array,
        [
            true_energy_bins.to_value(u.TeV),
            fov_lon_bins.to_value(u.deg),
            fov_lat_bins.to_value(u.deg),
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

        # normalize over the theta axis
        n_events = hist.sum(axis=-1)
        # normalize and replace nans with 0
        psf = np.nan_to_num(hist / n_events[..., np.newaxis])

    return psf / solid_angle
