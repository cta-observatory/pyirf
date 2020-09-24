import numpy as np
import astropy.units as u


def _normalize_hist(hist):
    # (N_E, N_MIGRA, N_FOV)
    # (N_E, N_FOV)

    norm = hist.sum(axis=1)
    h = np.swapaxes(hist, 0, 1)

    with np.errstate(invalid='ignore'):
        h /= norm

    h = np.swapaxes(h, 0, 1)
    return np.nan_to_num(h)


def energy_dispersion(
    selected_events,
    true_energy_bins,
    fov_offset_bins,
    migration_bins,
):
    mu = (selected_events['reco_energy'] / selected_events['true_energy']).to_value(u.one)

    energy_dispersion, _ = np.histogramdd(
        np.column_stack([
            selected_events['true_energy'].to_value(u.TeV),
            mu,
            selected_events['source_fov_offset'].to_value(u.deg),
        ]),
        bins=[
            true_energy_bins.to_value(u.TeV),
            migration_bins,
            fov_offset_bins.to_value(u.deg),
        ]
    )

    n_events_per_energy = energy_dispersion.sum(axis=1)
    assert len(n_events_per_energy) == len(true_energy_bins) - 1
    energy_dispersion = _normalize_hist(energy_dispersion)

    return energy_dispersion
