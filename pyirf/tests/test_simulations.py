import numpy as np
import astropy.units as u
import pytest


def test_integrate_energy():
    from pyirf.simulations import SimulatedEventsInfo

    info = SimulatedEventsInfo(
        n_showers=int(1e6),
        energy_min=100 * u.GeV,
        energy_max=10 * u.TeV,
        max_impact=500 * u.m,
        spectral_index=-2,
        viewcone_min=0 * u.deg,
        viewcone_max=10 * u.deg,
    )
    # simplest case, no bins  outside e_min, e_max
    energy_bins = np.geomspace(info.energy_min, info.energy_max, 20)
    assert np.isclose(
        info.calculate_n_showers_per_energy(energy_bins).sum(),
        info.n_showers,
    )

    # simple case, e_min and e_max on bin edges
    energy_bins = np.geomspace(info.energy_min, info.energy_max, 20)
    energy_bins = np.append(0.9 * info.energy_min, energy_bins)
    energy_bins = np.append(energy_bins, 1.1 * info.energy_max)

    events = info.calculate_n_showers_per_energy(energy_bins)
    assert np.isclose(info.n_showers, events.sum())

    assert events[0] == 0
    assert events[-1] == 0

    # complex case, e_min and e_max inside bins
    energy_bins = np.geomspace(
        0.5 * info.energy_min, 2 * info.energy_max, 5
    )
    events = info.calculate_n_showers_per_energy(energy_bins)
    assert np.isclose(info.n_showers, events.sum())
    assert events[0] > 0
    assert events[-1] > 0


def test_integrate_energy_fov():
    from pyirf.simulations import SimulatedEventsInfo

    # simple case, max viewcone on bin edge
    info = SimulatedEventsInfo(
        n_showers=int(1e6),
        energy_min=100 * u.GeV,
        energy_max=10 * u.TeV,
        max_impact=500 * u.m,
        spectral_index=-2,
        viewcone_min=0 * u.deg,
        viewcone_max=10 * u.deg,
    )

    fov_bins = [0, 10, 20] * u.deg
    energy_bins = np.geomspace(info.energy_min, info.energy_max, 20)

    n_events = info.calculate_n_showers_per_energy_and_fov(energy_bins, fov_bins)

    assert np.all(n_events[:, 1:] == 0)
    assert np.isclose(np.sum(n_events), int(1e6))

    # viewcone inside of bin
    info = SimulatedEventsInfo(
        n_showers=int(1e6),
        energy_min=100 * u.GeV,
        energy_max=10 * u.TeV,
        max_impact=500 * u.m,
        spectral_index=-2,
        viewcone_min=0 * u.deg,
        viewcone_max=10 * u.deg,
    )

    fov_bins = [0, 9, 11, 20] * u.deg
    energy_bins = np.geomspace(info.energy_min, info.energy_max, 20)
    n_events = info.calculate_n_showers_per_energy_and_fov(energy_bins, fov_bins)

    assert np.all(n_events[:, 1:2] > 0)
    np.testing.assert_equal(n_events[:, 2:], 0)
    assert np.isclose(np.sum(n_events), int(1e6))


def test_integrate_energy_fov_pointlike():
    from pyirf.simulations import SimulatedEventsInfo

    info = SimulatedEventsInfo(
        n_showers=int(1e6),
        energy_min=100 * u.GeV,
        energy_max=10 * u.TeV,
        max_impact=500 * u.m,
        spectral_index=-2,
        viewcone_min=0 * u.deg,
        viewcone_max=0 * u.deg,
    )

    fov_bins = [0, 9, 11, 20] * u.deg
    energy_bins = np.geomspace(info.energy_min, info.energy_max, 20)

    # make sure we raise an error on invalid input
    with pytest.raises(ValueError):
        info.calculate_n_showers_per_energy_and_fov(energy_bins, fov_bins)


def test_viewcone_integral():
    from pyirf.simulations import _viewcone_pdf_integral

    vmin = 1 * u.deg
    vmax = 5 * u.deg

    # completely outside viewcone range
    assert _viewcone_pdf_integral(vmin, vmax, 0 * u.deg, 0.5 * u.deg) == 0 * u.one
    assert _viewcone_pdf_integral(vmin, vmax, 5.0 * u.deg, 5.5 * u.deg) == 0 * u.one

    # half inside, half outside
    expected = _viewcone_pdf_integral(vmin, vmax, 1.0 * u.deg, 1.5 * u.deg)
    assert _viewcone_pdf_integral(vmin, vmax, 0.5 * u.deg, 1.5 * u.deg) == expected
    expected = _viewcone_pdf_integral(vmin, vmax, 4.5 * u.deg, 5.0 * u.deg)
    assert _viewcone_pdf_integral(vmin, vmax, 4.5 * u.deg, 5.5 * u.deg) == expected

    # whole region integrates to 1
    assert _viewcone_pdf_integral(vmin, vmax, 1 * u.deg, 5 * u.deg) == 1.0 * u.one
    assert _viewcone_pdf_integral(vmin, vmax, 0 * u.deg, 10 * u.deg) == 1.0 * u.one

    vmin = np.pi * u.rad
    vmax = 2 * np.pi * u.rad
    np.testing.assert_allclose(
        _viewcone_pdf_integral(vmin, vmax, np.pi * u.rad, 1.5 * np.pi * u.rad),
        0.5
    )

    # test scalar inputs result in scalar output
    assert _viewcone_pdf_integral(vmin, vmax, 0 * u.deg, 0.5 * u.deg).ndim == 0 * u.one
