import numpy as np
import astropy.units as u


def test_integrate_energy():
    from pyirf.simulations import SimulatedEventsInfo

    proton_info = SimulatedEventsInfo(
        n_showers=int(1e6),
        energy_min=100 * u.GeV,
        energy_max=10 * u.TeV,
        max_impact=500 * u.m,
        spectral_index=-2,
        viewcone=10 * u.deg,
    )
    # simplest case, no bins  outside e_min, e_max
    energy_bins = np.geomspace(proton_info.energy_min, proton_info.energy_max, 20)
    assert np.isclose(
        proton_info.calculate_n_showers_per_energy(energy_bins).sum(),
        proton_info.n_showers,
    )

    # simple case, e_min and e_max on bin edges
    energy_bins = np.geomspace(proton_info.energy_min, proton_info.energy_max, 20)
    energy_bins = np.append(0.9 * proton_info.energy_min, energy_bins)
    energy_bins = np.append(energy_bins, 1.1 * proton_info.energy_max)

    events = proton_info.calculate_n_showers_per_energy(energy_bins)
    assert np.isclose(proton_info.n_showers, events.sum())

    assert events[0] == 0
    assert events[-1] == 0

    # complex case, e_min and e_max inside bins
    energy_bins = np.geomspace(
        0.5 * proton_info.energy_min, 2 * proton_info.energy_max, 5
    )
    events = proton_info.calculate_n_showers_per_energy(energy_bins)
    assert np.isclose(proton_info.n_showers, events.sum())
    assert events[0] > 0
    assert events[-1] > 0


def test_integrate_energy_fov():
    from pyirf.simulations import SimulatedEventsInfo

    # simple case, max viewcone on bin edge
    proton_info = SimulatedEventsInfo(
        n_showers=int(1e6),
        energy_min=100 * u.GeV,
        energy_max=10 * u.TeV,
        max_impact=500 * u.m,
        spectral_index=-2,
        viewcone=10 * u.deg,
    )

    fov_bins = [0, 10, 20] * u.deg
    energy_bins = np.geomspace(proton_info.energy_min, proton_info.energy_max, 20)

    n_events = proton_info.calculate_n_showers_per_energy_and_fov(energy_bins, fov_bins)

    assert np.all(n_events[:, 1:] == 0)
    assert np.isclose(np.sum(n_events), int(1e6))

    # viewcone inside of bin
    proton_info = SimulatedEventsInfo(
        n_showers=int(1e6),
        energy_min=100 * u.GeV,
        energy_max=10 * u.TeV,
        max_impact=500 * u.m,
        spectral_index=-2,
        viewcone=10 * u.deg,
    )

    fov_bins = [0, 9, 11, 20] * u.deg
    energy_bins = np.geomspace(proton_info.energy_min, proton_info.energy_max, 20)
    n_events = proton_info.calculate_n_showers_per_energy_and_fov(energy_bins, fov_bins)

    assert np.all(n_events[:, 1:2] > 0)
    assert np.all(n_events[:, 2:] == 0)
    assert np.isclose(np.sum(n_events), int(1e6))
