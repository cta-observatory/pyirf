import numpy as np
import astropy.units as u
from astropy.table import QTable


def _normalize_hist(hist):
    with np.errstate(invalid='ignore'):
        h = hist.T
        h = h / h.sum(axis=0)
        return np.nan_to_num(h).T


def point_like_energy_dispersion(
    selected_events,
    true_energy_bins,
    migration_bins,
    max_theta,
):
    mu = (selected_events['reco_energy'] / selected_events['true_energy']).to_value(u.one)

    energy_dispersion, _, _ = np.histogram2d(
        selected_events['true_energy'].to_value(u.TeV),
        mu,
        bins=[
            true_energy_bins.to_value(u.TeV),
            migration_bins,
        ]
    )

    n_events_per_energy = energy_dispersion.sum(axis=1)
    assert len(n_events_per_energy) == len(true_energy_bins) - 1
    energy_dispersion = _normalize_hist(energy_dispersion)

    edisp = QTable({
        'true_energy_low': [true_energy_bins[:-1]],
        'true_energy_high': [true_energy_bins[1:]],
        'migration_low': [migration_bins[:-1]],
        'migration_high': [migration_bins[1:]],
        'theta_low': [0 * u.deg],
        'theta_high': [max_theta],
        'energy_dispersion': [energy_dispersion[:, :, np.newaxis]],
    })

    return edisp
