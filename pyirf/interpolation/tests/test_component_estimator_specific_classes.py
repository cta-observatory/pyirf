import astropy.units as u
import numpy as np
from pyirf.utils import cone_solid_angle
from scipy.stats import expon


def test_EnergyDispersionEstimator(prod5_irfs):
    from pyirf.interpolation import EnergyDispersionEstimator, QuantileInterpolator

    zen_pnt = np.array([key.value for key in prod5_irfs.keys()])
    edisps = np.array([irf["edisp"].data for irf in prod5_irfs.values()])
    bin_edges = list(prod5_irfs.values())[0]["edisp"].axes["migra"].edges
    bin_width = np.diff(bin_edges)

    estimator = EnergyDispersionEstimator(
        grid_points=zen_pnt[[0, 2]],
        migra_bins=bin_edges,
        energy_dispersion=edisps[[0, 2]],
        interpolator_cls=QuantileInterpolator,
        interpolator_kwargs={"quantile_resolution": 1e-3},
        extrapolator_cls=None,
        extrapolator_kwargs=None,
        axis=-2,
    )

    interp = estimator(target_point=zen_pnt[[1]])

    assert np.min(interp) >= 0
    assert np.all(np.isfinite(interp))
    assert np.all(
        np.logical_or(
            np.isclose(np.sum(interp * bin_width[:, np.newaxis], axis=-2), 1),
            np.isclose(np.sum(interp * bin_width[:, np.newaxis], axis=-2), 0),
        )
    )
    assert interp.shape == edisps[[1]].shape


def test_PSFTableEstimator():
    from pyirf.interpolation import PSFTableEstimator, QuantileInterpolator

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

    dummy_psfs = (
        np.array(
            [np.apply_along_axis(hist, -1, dummy_psf_template * pnt) for pnt in zen_pnt]
        )
        * u.sr**-1
    )

    estimator = PSFTableEstimator(
        grid_points=zen_pnt[[0, 2]],
        source_offset_bins=bin_edges,
        psf=dummy_psfs[[0, 2]],
        interpolator_cls=QuantileInterpolator,
        interpolator_kwargs={"quantile_resolution": 1e-3},
        extrapolator_cls=None,
        extrapolator_kwargs=None,
        axis=-1,
    )

    interp = estimator(target_point=zen_pnt[[1]])

    probability = (interp * omegas[np.newaxis, np.newaxis, np.newaxis, ...]).to_value(u.one)

    assert np.max(probability) <= 1
    assert np.min(probability) >= 0
    assert np.all(np.isfinite(interp))
    assert np.all(
        np.logical_or(
            np.isclose(np.sum(probability, axis=-1), 1),
            np.isclose(np.sum(probability, axis=-1), 0),
        )
    )
    assert interp.shape == dummy_psfs[[1]].shape


def test_EffectiveAreaEstimator_sythetic_data():
    """Test of interpolating of effective area using dummy model files."""
    from pyirf.interpolation import EffectiveAreaEstimator, GridDataInterpolator

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

    estimator = EffectiveAreaEstimator(
        grid_points=pars,
        effective_area=aeff,
        interpolator_cls=GridDataInterpolator,
        interpolator_kwargs=None,
        extrapolator_cls=None,
        extrapolator_kwargs=None,
        min_effective_area=min_aeff,
    )
    aeff_interp = estimator(pars0)

    # allowing for 3% accuracy except of close to the minimum value of Aeff
    assert np.allclose(aeff_interp[:, 0], aeff0, rtol=0.03, atol=min_aeff)


def test_EffectiveAreaEstimator_prod5(prod5_irfs):
    """Test of interpolation of effective are on prod5 irfs"""
    from pyirf.interpolation import EffectiveAreaEstimator, GridDataInterpolator

    zen_pnt = np.array([key.value for key in prod5_irfs.keys()])
    aeffs = np.array([irf["aeff"].data for irf in prod5_irfs.values()])
    min_aeff = 1 * u.m**2

    estimator = EffectiveAreaEstimator(
        grid_points=zen_pnt[[0, 2]],
        effective_area=aeffs[[0, 2]] * u.m**2,
        interpolator_cls=GridDataInterpolator,
        interpolator_kwargs={"method": "linear"},
        extrapolator_cls=None,
        extrapolator_kwargs=None,
        min_effective_area=min_aeff,
    )
    interp = estimator(zen_pnt[[1]]).value

    assert np.all(np.isfinite(interp))
    assert interp.shape == aeffs[[1]].shape
    assert np.all(interp >= 0)

    assert np.all(
        np.logical_or(
            np.logical_or(
                np.logical_and(aeffs[[0]] <= interp, interp <= aeffs[[2]]),
                np.logical_and(aeffs[[2]] <= interp, interp <= aeffs[[0]]),
            ),
            np.logical_or(interp == 0, interp == min_aeff.value),
        )
    )


def test_RadMaxEstimator():
    from pyirf.interpolation import GridDataInterpolator, RadMaxEstimator

    # linear test case
    rad_max_1 = np.array([[0, 0], [1, 0], [2, 1], [3, 2]])
    rad_max_2 = 2 * rad_max_1
    rad_max = np.array([rad_max_1, rad_max_2])

    grid_points = np.array([[0], [1]])
    target_point = np.array([0.5])

    estimator = RadMaxEstimator(
        grid_points=grid_points,
        rad_max=rad_max,
        interpolator_cls=GridDataInterpolator,
        interpolator_kwargs=None,
        extrapolator_cls=None,
        extrapolator_kwargs=None,
    )
    interp = estimator(target_point)

    assert interp.shape == (1, *rad_max_1.shape)
    assert np.allclose(interp, 1.5 * rad_max_1)
