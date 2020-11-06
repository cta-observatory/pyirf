import astropy.units as u
import numpy as np
from astropy.table import QTable


def test_effective_area():
    from pyirf.irf import effective_area

    n_selected = np.array([10, 20, 30])
    n_simulated = np.array([100, 2000, 15000])

    area = 1e5 * u.m ** 2

    assert u.allclose(
        effective_area(n_selected, n_simulated, area), [1e4, 1e3, 200] * u.m ** 2
    )


def test_effective_area_per_energy():
    from pyirf.irf import effective_area_per_energy
    from pyirf.simulations import SimulatedEventsInfo

    true_energy_bins = [0.1, 1.0, 10.0] * u.TeV
    selected_events = QTable(
        {
            "true_energy": np.append(np.full(1000, 0.5), np.full(10, 5)),
        }
    )

    # this should give 100000 events in the first bin and 10000 in the second
    simulation_info = SimulatedEventsInfo(
        n_showers=110000,
        energy_min=true_energy_bins[0],
        energy_max=true_energy_bins[-1],
        max_impact=100 / np.sqrt(np.pi) * u.m,  # this should give a nice round area
        spectral_index=-2,
        viewcone=0 * u.deg,
    )

    area = effective_area_per_energy(selected_events, simulation_info, true_energy_bins)

    assert area.shape == (len(true_energy_bins) - 1,)
    assert area.unit == u.m ** 2
    assert u.allclose(area, [100, 10] * u.m ** 2)


def test_effective_area_energy_fov():
    from pyirf.irf import effective_area_per_energy_and_fov
    from pyirf.simulations import SimulatedEventsInfo

    true_energy_bins = [0.1, 1.0, 10.0] * u.TeV
    # choose edges so that half are in each bin in fov
    fov_offset_bins = [0, np.arccos(0.98), np.arccos(0.96)] * u.rad
    center_1, center_2 = 0.5 * (fov_offset_bins[:-1] + fov_offset_bins[1:]).to_value(
        u.deg
    )

    selected_events = QTable(
        {
            "true_energy": np.concatenate(
                [
                    np.full(1000, 0.5),
                    np.full(10, 5),
                    np.full(500, 0.5),
                    np.full(5, 5),
                ]
            )
            * u.TeV,
            "true_source_fov_offset": np.append(
                np.full(1010, center_1), np.full(505, center_2)
            )
            * u.deg,
        }
    )

    # this should give 100000 events in the first bin and 10000 in the second
    simulation_info = SimulatedEventsInfo(
        n_showers=110000,
        energy_min=true_energy_bins[0],
        energy_max=true_energy_bins[-1],
        max_impact=100 / np.sqrt(np.pi) * u.m,  # this should give a nice round area
        spectral_index=-2,
        viewcone=fov_offset_bins[-1],
    )

    area = effective_area_per_energy_and_fov(
        selected_events, simulation_info, true_energy_bins, fov_offset_bins
    )

    assert area.shape == (len(true_energy_bins) - 1, len(fov_offset_bins) - 1)
    assert area.unit == u.m ** 2
    assert u.allclose(area[:, 0], [200, 20] * u.m ** 2)
    assert u.allclose(area[:, 1], [100, 10] * u.m ** 2)
