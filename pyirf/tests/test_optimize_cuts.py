import numpy as np
from astropy.table import QTable
import astropy.units as u


def test_optimize_gh_cuts():
    from pyirf.cuts import calculate_percentile_cut
    from pyirf.cut_optimization import optimize_gh_cut

    rng = np.random.default_rng(0)

    n_signal = 1000
    n_background = 10000

    signal = QTable({
        "reco_energy": rng.uniform(1.0, 10.0, n_signal) * u.TeV,
        "theta": rng.uniform(0.0, 0.5, n_signal) * u.deg,
        "gh_score": np.clip(rng.normal(0.7, 0.4, n_signal), 0, 1),
    })

    background = QTable({
        "reco_energy": rng.uniform(1.0, 10.0, n_background) * u.TeV,
        "theta": rng.uniform(0.0, 0.5, n_background) * u.deg,
        "gh_score": np.clip(rng.normal(0.2, 0.3, n_background), 0, 1),
        "reco_source_fov_offset": rng.uniform(0, 1, n_background) * u.deg,
    })


    e_reco_bins = np.linspace(1.0, 10.0, 5) * u.TeV
    theta_cuts = calculate_percentile_cut(signal["theta"], signal["reco_energy"], e_reco_bins, fill_value=1 * u.deg)

    optimize_gh_cut(signal, background, e_reco_bins, [0.5, 0.8, 0.9], theta_cuts)


def test_optimize_cuts():
    from pyirf.cut_optimization import optimize_cuts

    rng = np.random.default_rng(0)

    n_signal = 3000
    n_background = 10000

    signal = QTable({
        "reco_energy": rng.uniform(1.0, 10.0, n_signal) * u.TeV,
        "theta": rng.uniform(0.0, 0.5, n_signal) * u.deg,
        "gh_score": np.clip(rng.normal(0.7, 0.4, n_signal), 0, 1),
        "multiplicity": rng.integers(2, 15, n_signal),
        "weight": rng.uniform(0.5, 2, n_signal),
    })

    background = QTable({
        "reco_energy": rng.uniform(1.0, 10.0, n_background) * u.TeV,
        "theta": rng.uniform(0.0, 0.5, n_background) * u.deg,
        "gh_score": np.clip(rng.normal(0.2, 0.3, n_background), 0, 1),
        "reco_source_fov_offset": rng.uniform(0, 1, n_background) * u.deg,
        "multiplicity": rng.integers(2, 15, n_background),
        "weight": rng.uniform(0.5, 2, n_background),
    })


    e_reco_bins = np.linspace(1.0, 10.0, 5) * u.TeV

    optimize_cuts(
        signal,
        background,
        e_reco_bins,
        multiplicity_cuts=[2, 3, 4, 5],
        gh_cut_efficiencies=[0.5, 0.8, 0.9],
        theta_cut_efficiencies=[0.68, 0.75],
        alpha=0.2,
    )
