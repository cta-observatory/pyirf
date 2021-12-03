import astropy.units as u
import numpy as np
from gammapy.maps import MapAxis
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
            "true_source_fov_offset": np.concatenate(
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

    true_energy_axis = MapAxis.from_edges([0.1, 1.0, 10.0, 100], unit=u.TeV, name='energy_true')
    fov_offset_axis = MapAxis.from_edges([0, 1, 2], unit=u.deg, name='offset')
    migration_axis = MapAxis.from_bounds(0.2, 5, 1000, interp='log', name='migra')

    edisp2d = energy_dispersion(
        selected_events, true_energy_axis, fov_offset_axis, migration_axis
    )

    edisp = edisp2d.quantity
    assert edisp.shape == (3, 1000, 2)
    assert np.isclose(edisp.sum(), 6.0)

    cumulative_sum = np.cumsum(edisp, axis=1)
    bin_centers = migration_axis.center
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
