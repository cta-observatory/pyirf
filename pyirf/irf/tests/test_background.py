import astropy.units as u
from astropy.table import QTable
import numpy as np


def test_background():
    from pyirf.irf import background_2d
    from pyirf.utils import cone_solid_angle

    np.random.seed(0)

    N1 = 1000
    N2 = 100
    N = N1 + N2

    # toy event data set with just two energies
    # and a psf per energy bin, point-like
    events = QTable(
        {
            "reco_energy": np.append(np.full(N1, 1), np.full(N2, 2)) * u.TeV,
            "reco_source_fov_offset": np.zeros(N) * u.deg,
            "weight": np.ones(N),
        }
    )

    energy_bins = [0, 1.5, 3] * u.TeV
    fov_bins = [0, 1] * u.deg

    # We return a table with one row as needed for gadf
    bg = background_2d(events, energy_bins, fov_bins, t_obs=1 * u.s)

    # 2 energy bins, 1 fov bin, 200 source distance bins
    assert bg.shape == (2, 1)
    assert bg.unit == u.Unit("TeV-1 s-1 sr-1")

    # check that psf is normalized
    bin_solid_angle = np.diff(cone_solid_angle(fov_bins))
    e_width = np.diff(energy_bins)
    assert np.allclose(np.sum((bg.T * e_width).T * bin_solid_angle, axis=1), [1000, 100] / u.s)
