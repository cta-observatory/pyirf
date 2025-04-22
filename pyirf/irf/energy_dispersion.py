import warnings
import numpy as np
import astropy.units as u
from ..binning import resample_histogram1d


__all__ = [
    "energy_dispersion",
    "energy_dispersion_asymmetric_polar",
    "energy_dispersion_asymmetric_lonlat",
    "energy_migration_matrix",
    "energy_migration_matrix_asymmetric_polar",
    "energy_migration_matrix_asymmetric_lonlat",
    "energy_dispersion_to_migration",
    "energy_dispersion_to_migration_asymmetric",
]


def _normalize_hist(hist, migration_bins):
    # make sure we do not mutate the input array
    hist = hist.copy()
    bin_width = np.diff(migration_bins)

    # calculate number of events along the N_MIGRA axis to get events
    # per energy per fov
    norm = hist.sum(axis=1)

    with np.errstate(invalid="ignore"):
        if hist.ndim == 3:
            # hist shape is (N_E, N_MIGRA, N_FOV), norm shape is (N_E, N_FOV)
            # so we need to add a new axis in the middle to get (N_E, 1, N_FOV)
            # bin_width is 1d, so we need newaxis, use the values, newaxis
            hist = hist / norm[:, np.newaxis, :] / bin_width[np.newaxis, :, np.newaxis]
        else:
            # this handles the asymmetric case where hist shape is (N_E, N_MIGRA, N_FOV_1, N_FOV_2)
            hist /= norm[:, np.newaxis, ...]
            hist /= bin_width[np.newaxis, :, np.newaxis, np.newaxis]
    return np.nan_to_num(hist)


def energy_dispersion(
    selected_events,
    true_energy_bins,
    fov_offset_bins,
    migration_bins,
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
                selected_events["true_source_fov_offset"].to_value(u.deg),
            ]
        ),
        bins=[
            true_energy_bins.to_value(u.TeV),
            migration_bins,
            fov_offset_bins.to_value(u.deg),
        ],
    )

    energy_dispersion = _normalize_hist(energy_dispersion, migration_bins)

    return energy_dispersion


def energy_dispersion_asymmetric_polar(
    selected_events,
    true_energy_bins,
    fov_offset_bins,
    fov_position_angle_bins,
    migration_bins,
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
        Required columns: ``reco_energy``, ``true_energy``, ``true_source_fov_offset``,
        ``true_source_fov_position_angle``.
    true_energy_bins: astropy.units.Quantity[energy]
        Bin edges in true energy
    migration_bins: astropy.units.Quantity[energy]
        Bin edges in relative deviation, recommended range: [0.2, 5]
    fov_offset_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view offset.
        For Point-Like IRFs, only giving a single bin is appropriate.
    fov_position_angle_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view position angle.
        For Point-Like IRFs or when only considering offset, only giving a single bin 
        is apporpriate.

    Returns
    -------
    energy_dispersion: numpy.ndarray
        Energy dispersion matrix
        with shape (n_true_energy_bins, n_migration_bins, n_fov_offset_bins, 
        n_fov_position_angle_bins)
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
                selected_events["true_source_fov_position_angle"].to_value(u.deg),
            ]
        ),
        bins=[
            true_energy_bins.to_value(u.TeV),
            migration_bins,
            fov_offset_bins.to_value(u.deg),
            fov_position_angle_bins.to_value(u.deg),
        ],
    )

    energy_dispersion = _normalize_hist(energy_dispersion, migration_bins)

    return energy_dispersion


def energy_dispersion_asymmetric_lonlat(
    selected_events,
    true_energy_bins,
    fov_longitude_bins,
    fov_latitude_bins,
    migration_bins,
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
        Required columns: ``reco_energy``, ``true_energy``, ``true_source_fov_lon``, 
        ``true_source_fov_lat``.
    true_energy_bins: astropy.units.Quantity[energy]
        Bin edges in true energy
    migration_bins: astropy.units.Quantity[energy]
        Bin edges in relative deviation, recommended range: [0.2, 5]
    fov_longitude_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view longitude.
        For Point-Like IRFs, only giving a single bin is appropriate.
    fov_latitude_bins: astropy.units.Quantity[angle]
        Bin edges in the field of view latitude.
        For Point-Like IRFs, only giving a single bin is apporpriate.

    Returns
    -------
    energy_dispersion: numpy.ndarray
        Energy dispersion matrix
        with shape (n_true_energy_bins, n_migration_bins, n_fov_longitude_bins, 
        n_fov_latitude_bins)
    """
    mu = (selected_events["reco_energy"] / selected_events["true_energy"]).to_value(
        u.one
    )

    energy_dispersion, _ = np.histogramdd(
        np.column_stack(
            [
                selected_events["true_energy"].to_value(u.TeV),
                mu,
                selected_events["true_source_fov_lon"].to_value(u.deg),
                selected_events["true_source_fov_lat"].to_value(u.deg),
            ]
        ),
        bins=[
            true_energy_bins.to_value(u.TeV),
            migration_bins,
            fov_longitude_bins.to_value(u.deg),
            fov_latitude_bins.to_value(u.deg),
        ],
    )

    energy_dispersion = _normalize_hist(energy_dispersion, migration_bins)

    return energy_dispersion


@u.quantity_input(true_energy_bins=u.TeV, reco_energy_bins=u.TeV, fov_offset_bins=u.deg)
def energy_migration_matrix(
    events, true_energy_bins, reco_energy_bins, fov_offset_bins
):
    """Compute the energy migration matrix directly from the events.

    Parameters
    ----------
    events : `~astropy.table.QTable`
        Table of the DL2 events.
        Required columns: ``reco_energy``, ``true_energy``, ``true_source_fov_offset``.
    true_energy_bins : `~astropy.units.Quantity`
        Bin edges in true energy.
    reco_energy_bins : `~astropy.units.Quantity`
        Bin edges in reconstructed energy.

    Returns
    -------
    matrix : array-like
        Migration matrix as probabilities along the reconstructed energy axis.
        energy axis with shape
        (n_true_energy_bins, n_reco_energy_bins, n_fov_offset_bins)
        containing energies in TeV.
    """

    hist, _ = np.histogramdd(
        np.column_stack(
            [
                events["true_energy"].to_value(u.TeV),
                events["reco_energy"].to_value(u.TeV),
                events["true_source_fov_offset"].to_value(u.deg),
            ]
        ),
        bins=[
            true_energy_bins.to_value(u.TeV),
            reco_energy_bins.to_value(u.TeV),
            fov_offset_bins.to_value(u.deg),
        ],
    )

    with np.errstate(invalid="ignore"):
        hist /= hist.sum(axis=1)[:, np.newaxis, :]
        # the nans come from the fact that the sum along the reconstructed energy axis
        # might sometimes be 0 when there are no events in that given true energy bin
        # and fov offset bin
        hist[np.isnan(hist)] = 0

    return hist

@u.quantity_input(
    true_energy_bins=u.TeV,
    reco_energy_bins=u.TeV,
    fov_offset_bins=u.deg,
    fov_position_angle_bins=u.deg,
)
def energy_migration_matrix_asymmetric_polar(
    events, true_energy_bins, reco_energy_bins, fov_offset_bins, fov_position_angle_bins
):
    """Compute the energy migration matrix directly from the events in
    offset and position angle binning.

    Parameters
    ----------
    events : `~astropy.table.QTable`
        Table of the DL2 events.
        Required columns: ``reco_energy``, ``true_energy``, ``true_source_fov_offset``,
        ``true_source_fov_position_angle``.
    true_energy_bins : `~astropy.units.Quantity`
        Bin edges in true energy.
    reco_energy_bins : `~astropy.units.Quantity`
        Bin edges in reconstructed energy.

    Returns
    -------
    matrix : array-like
        Migration matrix as probabilities along the reconstructed energy axis.
        energy axis with shape
        (n_true_energy_bins, n_reco_energy_bins, n_fov_offset_bins,
        n_fov_position_angle_bins)
        containing energies in TeV.
    """

    hist, _ = np.histogramdd(
        np.column_stack(
            [
                events["true_energy"].to_value(u.TeV),
                events["reco_energy"].to_value(u.TeV),
                events["true_source_fov_offset"].to_value(u.deg),
                events["true_source_fov_position_angle"].to_value(u.deg),
            ]
        ),
        bins=[
            true_energy_bins.to_value(u.TeV),
            reco_energy_bins.to_value(u.TeV),
            fov_offset_bins.to_value(u.deg),
            fov_position_angle_bins.to_value(u.deg),
        ],
    )

    with np.errstate(invalid="ignore"):
        hist /= hist.sum(axis=1)[:, np.newaxis, ...]
        # the nans come from the fact that the sum along the reconstructed energy axis
        # might sometimes be 0 when there are no events in that given true energy bin
        # and fov offset bin
        hist[np.isnan(hist)] = 0

    return hist


@u.quantity_input(
    true_energy_bins=u.TeV,
    reco_energy_bins=u.TeV,
    fov_longitude_bins=u.deg,
    fov_latitude_bins=u.deg,
)
def energy_migration_matrix_asymmetric_lonlat(
    events, true_energy_bins, reco_energy_bins, fov_longitude_bins, fov_latitude_bins
):
    """Compute the energy migration matrix directly from the events in
    longitude and latitude binning.

    Parameters
    ----------
    events : `~astropy.table.QTable`
        Table of the DL2 events.
        Required columns: ``reco_energy``, ``true_energy``, ``true_source_fov_lon``,
        ``true_source_fov_lat``.
    true_energy_bins : `~astropy.units.Quantity`
        Bin edges in true energy.
    reco_energy_bins : `~astropy.units.Quantity`
        Bin edges in reconstructed energy.

    Returns
    -------
    matrix : array-like
        Migration matrix as probabilities along the reconstructed energy axis.
        energy axis with shape
        (n_true_energy_bins, n_reco_energy_bins, n_fov_longitude_bins,
        n_fov_latitude_bins)
        containing energies in TeV.
    """

    hist, _ = np.histogramdd(
        np.column_stack(
            [
                events["true_energy"].to_value(u.TeV),
                events["reco_energy"].to_value(u.TeV),
                events["true_source_fov_lon"].to_value(u.deg),
                events["true_source_fov_lat"].to_value(u.deg),
            ]
        ),
        bins=[
            true_energy_bins.to_value(u.TeV),
            reco_energy_bins.to_value(u.TeV),
            fov_longitude_bins.to_value(u.deg),
            fov_latitude_bins.to_value(u.deg),
        ],
    )

    with np.errstate(invalid="ignore"):
        hist /= hist.sum(axis=1)[:, np.newaxis, ...]
        # the nans come from the fact that the sum along the reconstructed energy axis
        # might sometimes be 0 when there are no events in that given true energy bin
        # and fov offset bin
        hist[np.isnan(hist)] = 0

    return hist


def energy_dispersion_to_migration(
    dispersion_matrix,
    disp_true_energy_edges,
    disp_migration_edges,
    new_true_energy_edges,
    new_reco_energy_edges,
):
    """
    Construct a energy migration matrix from an energy dispersion matrix.

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

    Returns
    -------
    migration_matrix: numpy.ndarray
        Three-dimensional energy migration matrix. The third dimension
        equals the fov offset dimension of the energy dispersion matrix.
    """
    migration_matrix = np.zeros(
        (
            len(new_true_energy_edges) - 1,
            len(new_reco_energy_edges) - 1,
            dispersion_matrix.shape[2],
        )
    )

    migra_width = np.diff(disp_migration_edges)
    probability = dispersion_matrix * migra_width[np.newaxis, :, np.newaxis]

    true_energy_interpolation = resample_histogram1d(
        probability,
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
            warnings.filterwarnings(
                "ignore", "invalid value encountered in true_divide"
            )
            interpolation_edges = new_reco_energy_edges / e_true

        y = resample_histogram1d(
            e_true_dispersion,
            disp_migration_edges,
            interpolation_edges,
            axis=0,
        )

        migration_matrix[idx, :, :] = y

    return migration_matrix


def energy_dispersion_to_migration_asymmetric(
    dispersion_matrix,
    disp_true_energy_edges,
    disp_migration_edges,
    new_true_energy_edges,
    new_reco_energy_edges,
):
    """
    Construct a energy migration matrix from an energy dispersion matrix.

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

    Returns
    -------
    migration_matrix: numpy.ndarray
        Four-dimensional energy migration matrix. The third and fourth
        dimension equal the fov dimensions of the energy dispersion matrix.
    """
    migration_matrix = np.zeros(
        (
            len(new_true_energy_edges) - 1,
            len(new_reco_energy_edges) - 1,
            *dispersion_matrix.shape[2:],
        )
    )

    migra_width = np.diff(disp_migration_edges)
    probability = dispersion_matrix * migra_width[np.newaxis, :, np.newaxis, np.newaxis]

    true_energy_interpolation = resample_histogram1d(
        probability,
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
            warnings.filterwarnings(
                "ignore", "invalid value encountered in true_divide"
            )
            interpolation_edges = new_reco_energy_edges / e_true

        y = resample_histogram1d(
            e_true_dispersion,
            disp_migration_edges,
            interpolation_edges,
            axis=0,
        )

        migration_matrix[idx, ...] = y

    return migration_matrix
