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

    selected_events = QTable(
        {
            "reco_energy": np.concatenate(
                [
                    np.random.normal(1.0, TRUE_SIGMA_1, size=N) * 0.5,
                    np.random.normal(1.0, TRUE_SIGMA_2, size=N) * 5,
                    np.random.normal(1.0, TRUE_SIGMA_3, size=N) * 50,
                ]
            )
            * u.TeV,
            "true_energy": np.concatenate(
                [np.full(N, 0.5), np.full(N, 5.0), np.full(N, 50.0)]
            )
            * u.TeV,
            "source_fov_offset": np.concatenate(
                [
                    np.full(N // 2, 0.2),
                    np.full(N // 2, 1.5),
                    np.full(N // 2, 0.2),
                    np.full(N // 2, 1.5),
                    np.full(N // 2, 0.2),
                    np.full(N // 2, 1.5),
                ]
            )
            * u.deg,
        }
    )

    true_energy_bins = np.array([0.1, 1.0, 10.0, 100]) * u.TeV
    fov_offset_bins = np.array([0, 1, 2]) * u.deg
    migration_bins = np.linspace(0, 2, 1001)

    result = energy_dispersion(
        selected_events, true_energy_bins, fov_offset_bins, migration_bins
    )

    assert result.shape == (3, 1000, 2)
    assert np.isclose(result.sum(), 6.0)

    cumulative_sum = np.cumsum(result, axis=1)
    bin_centers = 0.5 * (migration_bins[1:] + migration_bins[:-1])
    assert np.isclose(
        TRUE_SIGMA_1,
        (
            bin_centers[np.where(cumulative_sum[0, :, :] >= 0.84)[0][0]]
            - bin_centers[np.where(cumulative_sum[0, :, :] >= 0.16)[0][0]]
        )
        / 2,
        rtol=0.1,
    )
    assert np.isclose(
        TRUE_SIGMA_2,
        (
            bin_centers[np.where(cumulative_sum[1, :, :] >= 0.84)[0][0]]
            - bin_centers[np.where(cumulative_sum[1, :, :] >= 0.16)[0][0]]
        )
        / 2,
        rtol=0.1,
    )
    assert np.isclose(
        TRUE_SIGMA_3,
        (
            bin_centers[np.where(cumulative_sum[2, :, :] >= 0.84)[0][0]]
            - bin_centers[np.where(cumulative_sum[2, :, :] >= 0.16)[0][0]]
        )
        / 2,
        rtol=0.1,
    )


def test_energy_dispersion_to_migration():
    from pyirf.irf import energy_dispersion
    from pyirf.irf.energy_dispersion import energy_dispersion_to_migration

    np.random.seed(0)
    N = 10000
    true_energy_bins = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100]) * u.TeV
    fov_offset_bins = np.array([0, 1, 2]) * u.deg
    migration_bins = np.linspace(0, 2, 101)

    true_energy = np.random.uniform(
        true_energy_bins[0].value,
        true_energy_bins[-1].value,
        size=N
    ) * u.TeV
    reco_energy = true_energy * np.random.uniform(0.5, 1.5, size=N)

    selected_events = QTable(
        {
            "reco_energy": reco_energy,
            "true_energy": true_energy,
            "source_fov_offset": np.concatenate(
                [
                    np.full(N // 2, 0.2),
                    np.full(N // 2, 1.5),
                ]
            )
            * u.deg,
        }
    )

    dispersion_matrix = energy_dispersion(
        selected_events, true_energy_bins, fov_offset_bins, migration_bins
    )

    migration_matrix = energy_dispersion_to_migration(
        dispersion_matrix,
    )

    # first axis (true_energy) should not change
    assert migration_matrix.shape[0] == dispersion_matrix.shape[0]

    # second axis (reconstructed energy) contains more bins now
    assert migration_matrix.shape[1] == (
        dispersion_matrix.shape[0] * dispersion_matrix.shape[1]
    )

    # third axis (fov offset) should not change
    assert migration_matrix.shape[2] == dispersion_matrix.shape[2]

    # PDF should sum to one for each true energy and fov offset bin
    # (by construction of the example)
    assert np.isclose(
        migration_matrix.sum(axis=1),
        np.ones((migration_matrix.shape[0], migration_matrix.shape[2]))
    ).all()
