import numpy as np
import astropy.units as u
from gammapy.irf import PSF3D
from gammapy.maps import MapAxis

from ..utils import cone_solid_angle


def psf_table(
    events,
    true_energy_axis: MapAxis,
    source_offset_axis: MapAxis,
    fov_offset_axis: MapAxis
):
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
            true_energy_axis.edges.to_value(u.TeV),
            fov_offset_axis.edges.to_value(u.deg),
            source_offset_axis.edges.to_value(u.deg),
        ],
    )

    psf = _normalize_psf(hist, source_offset_axis.edges)

    return PSF3D(
        axes=[
            true_energy_axis,
            fov_offset_axis,
            source_offset_axis,
        ],
        data=psf
    )


def _normalize_psf(hist, source_offset_bins):
    """Normalize the psf histogram to a probability densitity over solid angle"""
    solid_angle = np.diff(cone_solid_angle(source_offset_bins))

    # ignore numpy zero division warning
    with np.errstate(invalid="ignore"):

        # normalize over the theta axis
        n_events = hist.sum(axis=2)
        # normalize and replace nans with 0
        psf = np.nan_to_num(hist / n_events[:, :, np.newaxis])

    return psf / solid_angle
