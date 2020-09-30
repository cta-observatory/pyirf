import warnings
import numpy as np
import astropy.units as u
from ..binning import resample_histogram1d


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


def energy_dispersion_to_migration(
    dispersion_matrix,
    disp_true_energy_edges,
    disp_migration_edges,
    new_true_energy_edges,
    new_reco_energy_edges,
):
    """
    Construct a energy migration matrix from a energy
    dispersion matrix.
    Depending on the new energy ranges, the sum over the first axis
    can be smaller than 1.
    The new true energy bins need to be a subset of the old range,
    extrapolation is not supported.
    New reconstruction bins outside of the old migration range are filled with
    zeros.

    Parameters
    ----------
    dispersion_matrix: numpy.ndarray
        Energy dispersion_matrix
    disp_true_energy_edges: astropy.units.Quantity[energy]
        True energy edges matching the first dimension of the dispersion matrix
    disp_migration_edges: numpy.ndarray
        Migration edges matching the second dimension of the dispersion matrix
    new_true_energy_edges: astropy.units.Quantity[energy]
        True energy edges matching the first dimension of the output
    new_reco_energy_edges: astropy.units.Quantity[energy]
        Reco energy edges matching the second dimension of the output

    Returns:
    --------
    migration_matrix: numpy.ndarray
        Three-dimensional energy migration matrix. The third dimension
        equals the fov offset dimension of the energy dispersion matrix.
    """

    migration_matrix = np.zeros((
        len(new_true_energy_edges) - 1,
        len(new_reco_energy_edges) - 1,
        dispersion_matrix.shape[2],
    ))

    true_energy_interpolation = resample_histogram1d(
        dispersion_matrix,
        disp_true_energy_edges,
        new_true_energy_edges,
        axis=0,
    )

    norm = np.sum(true_energy_interpolation, axis=1, keepdims=True)
    norm[norm == 0] = 1
    true_energy_interpolation /= norm

    for idx, e_true in enumerate(
        (new_true_energy_edges[1:] + new_true_energy_edges[:-1]) / 2
    ):

        # get migration for the new true energy bin
        e_true_dispersion = true_energy_interpolation[idx]

        with warnings.catch_warnings():
            # silence inf/inf division warning
            warnings.filterwarnings('ignore', 'invalid value encountered in true_divide')
            interpolation_edges = new_reco_energy_edges / e_true

        y = resample_histogram1d(
            e_true_dispersion,
            disp_migration_edges,
            interpolation_edges,
            axis=0,
        )

        migration_matrix[idx, :, :] = y

    return migration_matrix
