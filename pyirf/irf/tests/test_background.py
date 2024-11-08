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
    assert np.allclose(
        np.sum((bg.T * e_width).T * bin_solid_angle, axis=1), [1000, 100] / u.s
    )


def test_background_3d_lonlat():
    from pyirf.irf import background_3d_lonlat
    from pyirf.utils import rectangle_solid_angle
    from pyirf.irf.background import BACKGROUND_UNIT

    reco_energy_bins = [0.1, 1.1, 11.1, 111.1] * u.TeV
    fov_lon_bins = [-1.0, 0, 1.0] * u.deg
    fov_lat_bins = [-1.0, 0, 1.0] * u.deg

    N_low = 4000
    N_high = 40
    N_tot = N_low + N_high

    # Fill values
    E_low, E_hig = 0.5, 5
    Lon_low, Lon_hig = (-0.5, 0.5) * u.deg
    Lat_low, Lat_hig = (-0.5, 0.5) * u.deg

    t_obs = 100 * u.s
    bin_width_energy = np.diff(reco_energy_bins)
    bin_solid_angle = rectangle_solid_angle(
        fov_lon_bins[:-1], fov_lon_bins[1:], fov_lat_bins[:-1], fov_lat_bins[1:]
    )

    # Toy events with two energies and four different sky positions
    selected_events = QTable(
        {
            "reco_energy": np.concatenate(
                [
                    np.full(N_low // 4, E_low),
                    np.full(N_high // 4, E_hig),
                    np.full(N_low // 4, E_low),
                    np.full(N_high // 4, E_hig),
                    np.full(N_low // 4, E_low),
                    np.full(N_high // 4, E_hig),
                    np.full(N_low // 4, E_low),
                    np.full(N_high // 4, E_hig),
                ]
            )
            * u.TeV,
            "reco_fov_lon": np.concatenate(
                [
                    np.full(N_low // 4, Lon_low),
                    np.full(N_high // 4, Lon_hig),
                    np.full(N_low // 4, Lon_low),
                    np.full(N_high // 4, Lon_hig),
                    np.full(N_low // 4, Lon_low),
                    np.full(N_high // 4, Lon_hig),
                    np.full(N_low // 4, Lon_low),
                    np.full(N_high // 4, Lon_hig),
                ]
            )
            * u.deg,
            "reco_fov_lat": np.append(
                np.full(N_tot // 2, Lat_low),
                np.full(N_tot // 2, Lat_hig)
            )
            * u.deg,
            "weight": np.full(N_tot, 1.0),
        }
    )

    bkg_rate = background_3d_lonlat(
        selected_events,
        reco_energy_bins=reco_energy_bins,
        fov_lon_bins=fov_lon_bins,
        fov_lat_bins=fov_lat_bins,
        t_obs=t_obs,
    )
    assert bkg_rate.shape == (
        len(reco_energy_bins) - 1,
        len(fov_lon_bins) - 1,
        len(fov_lat_bins) - 1,
    )
    assert bkg_rate.unit == BACKGROUND_UNIT

    # Convert to counts, project to energy axis, and check counts round-trip correctly
    assert np.allclose(
        (bin_solid_angle * bkg_rate * bin_width_energy[:, np.newaxis, np.newaxis]).sum(axis=(1, 2)) * t_obs,
        [N_low, N_high, 0],
    )
    # Convert to counts, project to latitude axis, and check counts round-trip correctly
    assert np.allclose(
        (bin_solid_angle * bkg_rate * bin_width_energy[:, np.newaxis, np.newaxis]).sum(axis=(0, 1)) * t_obs,
        2 * [N_tot // 2],
    )
