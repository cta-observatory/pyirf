from astropy.table import QTable
import numpy as np
import astropy.units as u


def test_estimate_background():
    from pyirf.sensitivity import estimate_background
    N = 1000
    events = QTable({
        'source_fov_offset': np.append(np.full(N, 0.5), np.full(N, 1.5)) * u.deg,
        'reco_energy': np.tile([5, 50], N) * u.TeV,
        'weight': np.tile([1, 2], N),
    })
    reco_energy_bins = [1, 10, 100] * u.TeV
    theta_cuts = QTable({
        'low': [1, 10] * u.TeV,
        'high': [10, 100] * u.TeV,
        'center': [5.5, 55] * u.TeV,
        'cut': (np.arccos([0.9998, 0.9999]) * u.rad).to(u.deg),
    })
    background_radius = np.arccos(0.999) * u.rad

    bg = estimate_background(
        events,
        reco_energy_bins,
        theta_cuts,
        alpha=0.2,
        background_radius=background_radius
    )

    assert np.allclose(bg['n'], [1000, 500])
    assert np.allclose(bg['n_weighted'], [1000, 1000])
