import astropy.units as u
import numpy as np
from astropy.table import QTable


def test_energy_dispersion():
    from pyirf.irf import energy_dispersion
    selected_events = QTable({
        'reco_energy': np.concatenate([
            np.random.uniform(0.081, 0.099, size=3),
            np.random.uniform(0.1, 0.119, size=7),
            np.random.uniform(0.81, 0.99, size=5),
            np.random.uniform(1.0, 1.19, size=5),
            np.random.uniform(8.1, 9.9, size=8),
            np.random.uniform(10.0, 10.9, size=2),
        ])*u.TeV,
        'true_energy': np.concatenate([
            np.full(10, 0.1),
            np.full(10, 1.0),
            np.full(10, 10.0)
        ])*u.TeV,
        'source_fov_offset': np.concatenate([
            np.full(20, 0.2),
            np.full(10, 1.5)
        ])*u.deg
    })

    true_energy_bins = np.array([0.1, 1.0, 10.0]) * u.TeV
    fov_offset_bins = np.array([0, 1, 2]) * u.deg
    migration_bins = np.array([0.8, 1.0, 1.2])

    result = energy_dispersion(
            selected_events,
            true_energy_bins,
            fov_offset_bins,
            migration_bins)

    assert result.sum() == 3.0
    assert (result == np.array([
        [[0.3, 0.0], [0.7, 0.0]],
        [[0.5, 0.8], [0.5, 0.2]]
    ])).all()
