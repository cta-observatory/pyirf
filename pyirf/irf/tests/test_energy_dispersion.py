import astropy.units as u
import numpy as np
from astropy.table import QTable


def test_energy_dispersion():
    from pyirf.irf import energy_dispersion

    np.random.seed(0)

    N = 10000
    TRUE_SIGMA_1 = 0.20
    TRUE_SIGMA_2 = 0.10
    TRUE_SIGMA_3 = 0.05

    selected_events = QTable({
        'reco_energy': np.concatenate([
            np.random.normal(1.0, TRUE_SIGMA_1, size=N)*0.5,
            np.random.normal(1.0, TRUE_SIGMA_2, size=N)*5,
            np.random.normal(1.0, TRUE_SIGMA_3, size=N)*50,
        ])*u.TeV,
        'true_energy': np.concatenate([
            np.full(N, 0.5),
            np.full(N, 5.0),
            np.full(N, 50.0)
        ])*u.TeV,
        'source_fov_offset': np.concatenate([
            np.full(N // 2, 0.2),
            np.full(N // 2, 1.5),
            np.full(N // 2, 0.2),
            np.full(N // 2, 1.5),
            np.full(N // 2, 0.2),
            np.full(N // 2, 1.5),
        ])*u.deg
    })

    true_energy_bins = np.array([0.1, 1.0, 10.0, 100]) * u.TeV
    fov_offset_bins = np.array([0, 1, 2]) * u.deg
    migration_bins = np.linspace(0, 2, 1001)

    result = energy_dispersion(
            selected_events,
            true_energy_bins,
            fov_offset_bins,
            migration_bins)

    assert result.shape == (3, 1000, 2)
    assert np.isclose(result.sum(),  6.0)

    cumulated = np.cumsum(result, axis=1)
    bin_centers = 0.5 * (migration_bins[1:] + migration_bins[:-1])
    assert np.isclose(
        TRUE_SIGMA_1,
        (bin_centers[np.where(cumulated[0, :, :] >= 0.84)[0][0]]
         - bin_centers[np.where(cumulated[0, :, :] >= 0.16)[0][0]])/2,
        rtol=0.1
    )
    assert np.isclose(
        TRUE_SIGMA_2,
        (bin_centers[np.where(cumulated[1, :, :] >= 0.84)[0][0]]
         - bin_centers[np.where(cumulated[1, :, :] >= 0.16)[0][0]])/2,
        rtol=0.1
    )
    assert np.isclose(
        TRUE_SIGMA_3,
        (bin_centers[np.where(cumulated[2, :, :] >= 0.84)[0][0]]
         - bin_centers[np.where(cumulated[2, :, :] >= 0.16)[0][0]])/2,
        rtol=0.1
    )
