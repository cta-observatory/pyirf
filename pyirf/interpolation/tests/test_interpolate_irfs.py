import astropy.units as u
import numpy as np
from scipy.stats import expon

import pyirf.interpolation as interp
from pyirf.utils import cone_solid_angle


def test_interpolate_energy_dispersion(prod5_irfs):
    from pyirf.interpolation import interpolate_energy_dispersion

    zen_pnt = np.array([key.value for key in prod5_irfs.keys()])
    edisps = np.array([irf["edisp"].data for irf in prod5_irfs.values()])
    bin_edges = list(prod5_irfs.values())[0]["edisp"].axes["migra"].edges

    interp = interpolate_energy_dispersion(
        migra_bins=bin_edges,
        edisps=edisps[[0, 2]],
        grid_points=zen_pnt[[0, 2]],
        target_point=zen_pnt[[1]],
        quantile_resolution=1e-3,
    )

    assert np.max(interp) <= 1
    assert np.min(interp) >= 0
    assert np.all(np.isfinite(interp))
    assert np.all(
        np.logical_or(
            np.isclose(np.sum(interp, axis=-2), 1),
            np.isclose(np.sum(interp, axis=-2), 0),
        )
    )
    assert interp.shape == edisps[[1]].shape


def test_interpolate_psf_table():
    from pyirf.interpolation import interpolate_psf_table

    # dummy psf_table with 30 bins of true energ and 6 bins of fov-offset, rad-axis
    # to be inflated later
    dummy_psf_template = np.ones((30, 6, 1))

    zen_pnt = np.array([20, 40, 60])
    bin_edges = np.linspace(0, 1, 31) * u.deg
    omegas = np.diff(cone_solid_angle(bin_edges))

    def hist(pnt):
        """Create dummy psf for given pointing"""
        histogram = np.diff(expon(scale=pnt / 400).cdf(bin_edges))
        normed_hist = histogram / np.sum(histogram)

        return normed_hist / omegas

    dummy_psfs = np.array(
        [np.apply_along_axis(hist, -1, dummy_psf_template * pnt) for pnt in zen_pnt]
    )

    interp = interpolate_psf_table(
        source_offset_bins=bin_edges,
        psfs=dummy_psfs[[0, 2]],
        grid_points=zen_pnt[[0, 2]],
        target_point=zen_pnt[[1]],
        quantile_resolution=1e-3,
    )

    interp *= omegas[np.newaxis, np.newaxis, np.newaxis, ...]

    assert np.max(interp) <= 1
    assert np.min(interp) >= 0
    assert np.all(np.isfinite(interp))
    assert np.all(
        np.logical_or(
            np.isclose(np.sum(interp, axis=-1), 1),
            np.isclose(np.sum(interp, axis=-1), 0),
        )
    )
    assert interp.shape == dummy_psfs[[1]].shape


def test_interpolate_effective_area_per_energy_and_fov():
    """Test of interpolating of effective area using dummy model files."""
    n_en = 20
    n_th = 1
    en = np.logspace(-2, 2, n_en)
    # applying a simple sigmoid function
    aeff0 = 1.0e4 / (1 + 1 / en**2) * u.m**2

    # assume that for parameters 'x' and 'y' the Aeff scales x*y*Aeff0
    x = [0.9, 1.1]
    y = [8.0, 11.5]
    n_grid = len(x) * len(y)
    aeff = np.empty((n_grid, n_th, n_en))
    pars = np.empty((n_grid, 2))
    i_grid = 0
    for xx in x:
        for yy in y:
            aeff[i_grid, 0, :] = aeff0 * xx * yy / 10
            pars[i_grid, :] = np.array([xx, yy])
            i_grid += 1
    aeff *= u.m**2
    pars0 = np.array([1, 10])
    min_aeff = 1 * u.m**2
    aeff_interp = interp.interpolate_effective_area_per_energy_and_fov(
        aeff, pars, pars0, min_effective_area=min_aeff, method="linear"
    )

    # allowing for 3% accuracy except of close to the minimum value of Aeff
    assert np.allclose(aeff_interp[:, 0], aeff0, rtol=0.03, atol=min_aeff)


def test_interpolate_effective_area_per_energy_and_fov_prod5(prod5_irfs):
    """Test of interpolation of effective are on prod5 irfs"""
    from pyirf.interpolation import interpolate_effective_area_per_energy_and_fov

    zen_pnt = np.array([key.value for key in prod5_irfs.keys()])
    aeffs = np.array([irf["aeff"].data for irf in prod5_irfs.values()])

    interp = interpolate_effective_area_per_energy_and_fov(
        effective_area=aeffs[[0, 2]] * u.m**2,
        grid_points=zen_pnt[[0, 2]],
        target_point=zen_pnt[[1]],
        min_effective_area=1 * u.m**2,
    ).value

    assert np.all(np.isfinite(interp))
    assert interp.shape == aeffs[[1]].shape
    assert np.all(interp >= 0)

    assert np.all(
        np.logical_or(
            np.logical_or(
                np.logical_and(aeffs[[0]] <= interp, interp <= aeffs[[2]]),
                np.logical_and(aeffs[[2]] <= interp, interp <= aeffs[[0]]),
            ),
            interp == 0,
        )
    )


def test_interpolate_rad_max():
    from pyirf.interpolation import interpolate_rad_max

    # linear test case
    rad_max_1 = np.array([[0, 0], [1, 0], [2, 1], [3, 2]])
    rad_max_2 = 2 * rad_max_1
    rad_max = np.array([rad_max_1, rad_max_2])

    grid_points = np.array([[0], [1]])
    target_point = np.array([0.5])

    interp = interpolate_rad_max(
        rad_max=rad_max,
        grid_points=grid_points,
        target_point=target_point,
        method="linear",
    )

    assert interp.shape == (1, *rad_max_1.shape)
    assert np.allclose(interp, 1.5 * rad_max_1)
