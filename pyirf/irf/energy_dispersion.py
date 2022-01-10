import warnings
import numpy as np
import astropy.units as u

from gammapy.irf import EnergyDispersion2D
from gammapy.maps import MapAxis


__all__ = [
    "energy_dispersion",
]


def _normalize_hist(hist):
    # make sure we do not mutate the input array
    hist = hist.copy()

    # calculate number of events along the N_MIGRA axis to get events
    # per energy per fov
    norm = hist.sum(axis=1)

    with np.errstate(invalid="ignore"):
        # hist shape is (N_E, N_MIGRA, N_FOV), norm shape is (N_E, N_FOV)
        # so we need to add a new axis in the middle to get (N_E, 1, N_FOV)
        # for broadcasting
        hist = hist / norm[:, np.newaxis, :]

    return np.nan_to_num(hist)


def energy_dispersion(
    selected_events,
    true_energy_axis: MapAxis,
    fov_offset_axis: MapAxis,
    migration_axis: MapAxis,
):
    """
    Calculate energy dispersion for the given DL2 event list.
    Energy dispersion is defined as the probability of finding an event
    at a given relative deviation ``(reco_energy / true_energy)`` for a given
    true energy.

    Parameters
    ----------
    selected_events: astropy.table.QTable
        Table of the DL2 events.
        Required columns: ``reco_energy``, ``true_energy``, ``true_source_fov_offset``.
    true_energy_aix: MapAxis[energy]
        Bin edges in true energy
    migration_axis: MapAxis[dimensionless]
        Bin edges in relative deviation, recommended range: [0.2, 5]
    fov_offset_axis: MapAxis[angle]
        Field of view offset axis.
        For Point-Like IRFs, only giving a single bin is appropriate.

    Returns
    -------
    energy_dispersion: numpy.ndarray
        Energy dispersion matrix
        with shape (n_true_energy_bins, n_migration_bins, n_fov_ofset_bins)
    """
    mu = (selected_events["reco_energy"] / selected_events["true_energy"]).to_value(
        u.one
    )

    energy_dispersion, _ = np.histogramdd(
        np.column_stack(
            [
                selected_events["true_energy"].to_value(u.TeV),
                mu,
                selected_events["true_source_fov_offset"].to_value(u.deg),
            ]
        ),
        bins=[
            true_energy_axis.edges.to_value(u.TeV),
            migration_axis.edges.to_value(u.one),
            fov_offset_axis.edges.to_value(u.deg),
        ],
    )

    energy_dispersion = _normalize_hist(energy_dispersion)

    return EnergyDispersion2D(
        axes=[
            true_energy_axis,
            migration_axis,
            fov_offset_axis,
        ],
        data=energy_dispersion
    )
