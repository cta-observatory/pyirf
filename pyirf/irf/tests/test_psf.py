import astropy.units as u
from astropy.table import QTable
import numpy as np


def test_psf():
    from pyirf.irf import psf_table
    from pyirf.utils import cone_solid_angle

    N = 1000

    np.random.seed()
    TRUE_SIGMA_1 = 0.2
    TRUE_SIGMA_2 = 0.1
    TRUE_SIGMA = np.append(np.full(N, TRUE_SIGMA_1), np.full(N, TRUE_SIGMA_2))

    # toy event data set with just two energies
    # and a psf per energy bin, point-like
    events = QTable({
        'true_energy': np.append(np.full(N, 1), np.full(N, 2)) * u.TeV,
        'pointing_az': np.zeros(2 * N) * u.deg,
        'pointing_alt': np.full(2 * N, 70) * u.deg,
        'true_az': np.zeros(2 * N) * u.deg,
        'true_alt': np.full(2 * N, 70) * u.deg,
        'theta': np.random.normal(0, TRUE_SIGMA) * u.deg,
    })

    energy_bins = [0, 1.5, 3] * u.TeV
    fov_bins = [0, 1] * u.deg
    source_bins = np.linspace(0, 1, 201) * u.deg

    # We return a table with one row as needed for gadf
    psf = psf_table(events, energy_bins, source_bins, fov_bins)[0]

    # 2 energy bins, 1 fov bin, 200 source distance bins
    assert psf['psf'].shape == (2, 1, 200)
    assert psf['psf'].unit == u.Unit('sr-1')

    # check that psf is normalized
    bin_solid_angle = np.diff(cone_solid_angle(source_bins))
    assert np.allclose(np.sum(psf['psf'] * bin_solid_angle, axis=2), 1.0)

    cumulated = np.cumsum(psf['psf'] * bin_solid_angle, axis=2)

    # first energy and only fov bin
    bin_centers = 0.5 * (source_bins[1:] + source_bins[:-1])
    assert u.isclose(bin_centers[np.where(cumulated[0, 0, :] >= 0.68)[0][0]], TRUE_SIGMA_1 * u.deg, rtol=0.1)

    # second energy and only fov bin
    assert u.isclose(bin_centers[np.where(cumulated[1, 0, :] >= 0.68)[0][0]], TRUE_SIGMA_2 * u.deg, rtol=0.1)
