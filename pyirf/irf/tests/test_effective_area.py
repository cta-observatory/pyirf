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
        viewcone_min=0 * u.deg,
        viewcone_max=0 * u.deg,
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
        viewcone_min=0 * u.deg,
        viewcone_max=fov_offset_bins[-1],
    )

    area = effective_area_per_energy_and_fov(
        selected_events, simulation_info, true_energy_bins, fov_offset_bins
    )

    assert area.shape == (len(true_energy_bins) - 1, len(fov_offset_bins) - 1)
    assert area.unit == u.m ** 2
    assert u.allclose(area[:, 0], [200, 20] * u.m ** 2)
    assert u.allclose(area[:, 1], [100, 10] * u.m ** 2)


def test_effective_area_3d_polar():
    from pyirf.irf import effective_area_3d_polar
    from pyirf.simulations import SimulatedEventsInfo

    true_energy_bins = [0.1, 1.0, 10.0] * u.TeV
    # choose edges so that half are in each bin in fov
    fov_offset_bins = [0, np.arccos(0.98), np.arccos(0.96)] * u.rad
    # choose edges so that they are equal size and cover the whole circle
    fov_pa_bins = [0, np.pi, 2*np.pi] * u.rad
    center_1_o, center_2_o = 0.5 * (fov_offset_bins[:-1] + fov_offset_bins[1:]).to_value(
        u.deg
    )
    center_1_pa, center_2_pa = 0.5 * (fov_pa_bins[:-1] + fov_pa_bins[1:]).to_value(
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
                    np.full(1000, 0.5),
                    np.full(10, 5),
                    np.full(500, 0.5),
                    np.full(5, 5),
                ]
            )
            * u.TeV,
            "true_source_fov_offset": np.concatenate(
                [
                    np.full(1010, center_1_o),
                    np.full(505, center_2_o),
                    np.full(1010, center_1_o),
                    np.full(505, center_2_o),
                ]
            )
            * u.deg,
            "true_source_fov_position_angle": np.append(
                np.full(1515, center_1_pa), np.full(1515, center_2_pa)
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
        viewcone_min=0 * u.deg,
        viewcone_max=fov_offset_bins[-1],
    )

    area = effective_area_3d_polar(
        selected_events, simulation_info, true_energy_bins, fov_offset_bins, fov_pa_bins
    )

    assert area.shape == (
        len(true_energy_bins) - 1, len(fov_offset_bins) - 1, len(fov_pa_bins) - 1
    )
    assert area.unit == u.m ** 2
    assert u.allclose(area[:, 0, :], [[400, 400],[40,40]] * u.m ** 2)
    assert u.allclose(area[:, 1, :], [[200, 200],[20,20]] * u.m ** 2)


def test_effective_area_3d_lonlat():
    from pyirf.irf import effective_area_3d_lonlat
    from pyirf.simulations import SimulatedEventsInfo
    from pyirf.utils import cone_solid_angle, rectangle_solid_angle

    true_energy_bins = [0.1, 1.0, 10.0] * u.TeV
    # non-square grid with a different number of bins on every axis, so the
    # result shape (n_energy, n_lon, n_lat) = (2, 3, 4) makes a lat/lon swap
    # directly visible
    fov_lon_bins = [-1.5, -0.5, 0.5, 1.5] * u.deg
    fov_lat_bins = [-2.0, -1.0, 0.0, 1.0, 2.0] * u.deg
    lon_centers = 0.5 * (fov_lon_bins[:-1] + fov_lon_bins[1:])
    lat_centers = 0.5 * (fov_lat_bins[:-1] + fov_lat_bins[1:])

    # selected events at the center of each cell, a distinct count per cell
    n_selected_e0 = np.array(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
        ]
    )
    n_selected_e1 = n_selected_e0 // 10

    true_energy, fov_lon, fov_lat = [], [], []
    for i in range(3):
        for j in range(4):
            for n, energy in [(n_selected_e0[i, j], 0.5), (n_selected_e1[i, j], 5.0)]:
                true_energy.append(np.full(n, energy))
                fov_lon.append(np.full(n, lon_centers[i].to_value(u.deg)))
                fov_lat.append(np.full(n, lat_centers[j].to_value(u.deg)))

    selected_events = QTable(
        {
            "true_energy": np.concatenate(true_energy) * u.TeV,
            "true_source_fov_lon": np.concatenate(fov_lon) * u.deg,
            "true_source_fov_lat": np.concatenate(fov_lat) * u.deg,
        }
    )

    # the viewcone covers the whole grid, so all bins are fully inside it and
    # the expected effective area of every cell is exact
    simulation_info = SimulatedEventsInfo(
        n_showers=110000,
        energy_min=true_energy_bins[0],
        energy_max=true_energy_bins[-1],
        max_impact=100 / np.sqrt(np.pi) * u.m,  # this should give a nice round area
        spectral_index=-2,
        viewcone_min=0 * u.deg,
        viewcone_max=3 * u.deg,
    )

    area = effective_area_3d_lonlat(
        selected_events,
        simulation_info,
        true_energy_bins,
        fov_lon_bins,
        fov_lat_bins,
        subpixels=20,
    )

    assert area.shape == (2, 3, 4)
    assert area.unit == u.m ** 2

    # expected area per cell: selected / simulated * area, with the
    # simulated showers per cell given by its solid angle fraction of the
    # viewcone
    e_integral = simulation_info.calculate_n_showers_per_energy(true_energy_bins)
    viewcone_area = cone_solid_angle(simulation_info.viewcone_max) - cone_solid_angle(
        simulation_info.viewcone_min
    )
    cell_frac = (
        np.array(
            [
                [
                    rectangle_solid_angle(
                        fov_lon_bins[i],
                        fov_lon_bins[i + 1],
                        fov_lat_bins[j],
                        fov_lat_bins[j + 1],
                    ).to_value(u.sr)
                    for j in range(4)
                ]
                for i in range(3)
            ]
        )
        / viewcone_area.to_value(u.sr)
    )
    n_selected = np.stack([n_selected_e0, n_selected_e1])
    expected = (
        n_selected
        / (e_integral[:, np.newaxis, np.newaxis] * cell_frac)
        * (np.pi * simulation_info.max_impact ** 2).to_value(u.m ** 2)
    ) * u.m ** 2

    assert u.allclose(area, expected, rtol=1e-8)