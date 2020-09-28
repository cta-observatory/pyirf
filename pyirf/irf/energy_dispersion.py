import numpy as np
import astropy.units as u


__all__ = [
    "energy_dispersion",
    "energy_dispersion_to_migration"
]


def _normalize_hist(hist):
    # (N_E, N_MIGRA, N_FOV)
    # (N_E, N_FOV)

    norm = hist.sum(axis=1)
    h = np.swapaxes(hist, 0, 1)

    with np.errstate(invalid="ignore"):
        h /= norm

    h = np.swapaxes(h, 0, 1)
    return np.nan_to_num(h)


def energy_dispersion(
    selected_events, true_energy_bins, fov_offset_bins, migration_bins,
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
        Required columns: ``reco_energy``, ``true_energy``, ``source_fov_offset``.
    true_energy_bins: astropy.units.Quantity[energy]
        Bin edges in true energy
    migration_bins: astropy.units.Quantity[energy]
        Bin edges in relative deviation, recommended range: [0.2, 5]
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
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
                selected_events["source_fov_offset"].to_value(u.deg),
            ]
        ),
        bins=[
            true_energy_bins.to_value(u.TeV),
            migration_bins,
            fov_offset_bins.to_value(u.deg),
        ],
    )

    n_events_per_energy = energy_dispersion.sum(axis=1)
    assert len(n_events_per_energy) == len(true_energy_bins) - 1
    energy_dispersion = _normalize_hist(energy_dispersion)

    return energy_dispersion


def energy_dispersion_to_migration(dispersion_matrix, true_energy_bins):
    assert len(true_energy_bins) - 1 == dispersion_matrix.shape[0]
    n_true_energy_bins = dispersion_matrix.shape[0]
    n_dispersion_bins = dispersion_matrix.shape[1]
    n_offset_bins = dispersion_matrix.shape[2]

    # additional true energy bins?
    migration_matrix = np.zeros((
        n_true_energy_bins,
        n_true_energy_bins * n_dispersion_bins,
        n_offset_bins,
    ))

    # probably can be done with numpy sparse matrices, but lets start like this
    for idx in range(n_true_energy_bins):
        migration_matrix[idx, idx*n_dispersion_bins + np.arange(n_dispersion_bins), :] = dispersion_matrix[idx, :, :]


    return migration_matrix

