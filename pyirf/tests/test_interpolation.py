import pyirf.interpolation as interp
import numpy as np
import astropy.units as u


def test_interpolate_effective_area_per_energy_and_fov():
    """Test of interpolating of effective area using dummy model files."""
    n_en = 20
    n_th = 1
    en = np.logspace(-2, 2, n_en)
    # applying a simple sigmoid function
    aeff0 = 1.0e4 / (1 + 1 / en**2) * u.Unit("m2")

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
    aeff *= u.Unit("m2")
    pars0 = (1, 10)
    min_aeff = 1 * u.Unit("m2")
    aeff_interp = interp.interpolate_effective_area_per_energy_and_fov(
        aeff, pars, pars0, min_effective_area=min_aeff, method="linear"
    )
    # allowing for 3% accuracy except of close to the minimum value of Aeff
    assert np.allclose(aeff_interp[:, 0], aeff0, rtol=0.03, atol=min_aeff)


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
