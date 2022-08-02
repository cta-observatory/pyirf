import pyirf.interpolation as interp
import numpy as np
import astropy.units as u
import pytest
from scipy.stats import norm


@pytest.fixture
def data():
    """Create common dataset containing binned Gaussians for interpolation testing. Binned PDFs sum to 1."""
    bin_edges = np.linspace(-5, 30, 101)
    distributions = [norm(5, 1), norm(10, 2), norm(15, 3)]

    # create binned pdfs by interpolation of bin content
    binned_pdfs = np.array([np.diff(dist.cdf(bin_edges)) for dist in distributions])

    dataset = {
        "bin_edges": bin_edges,
        "means": np.array([5, 10, 15]),
        "stds": np.array([1, 2, 3]),
        "distributions": distributions,
        "binned_pdfs": binned_pdfs,
        "grid_points": np.array([1, 2, 3]),
    }

    return dataset


def test_cdf_values(data):
    from pyirf.interpolation import cdf_values

    cdf_est = cdf_values(data["binned_pdfs"][0])

    # Assert empty histograms result in cdf containing zeros
    assert np.all(cdf_values(np.zeros(shape=5)) == 0)

    # Assert cdf is increasing or constant for actual pdfs
    assert np.all(np.diff(cdf_est) >= 0)

    # Assert cdf is capped at 1
    assert np.max(cdf_est) == 1

    # Assert estimated and true cdf are matching
    assert np.allclose(cdf_est, data["distributions"][0].cdf(data["bin_edges"][1:]))


def test_ppf_values(data):
    from pyirf.interpolation import ppf_values, cdf_values

    # Create quantiles, ignore the 0% and 100% quantile as they are analytically +- inf
    quantiles = np.linspace(0, 1, 10)[1:-2]

    # True ppf-values
    ppf_true = data["distributions"][0].ppf(quantiles)

    # Estimated ppf-values
    cdf_est = cdf_values(data["binned_pdfs"][0])
    ppf_est = ppf_values(cdf_est, data["bin_edges"], quantiles)

    # Assert truth and estimation match allowing for +- bin_width deviation
    assert np.allclose(ppf_true, ppf_est, atol=np.diff(data["bin_edges"])[0])


def test_pdf_from_ppf(data):
    from pyirf.interpolation import ppf_values, cdf_values, pdf_from_ppf

    # Create quantiles
    quantiles = np.linspace(0, 1, 1000)

    # Estimate ppf-values
    cdf_est = cdf_values(data["binned_pdfs"][0])
    ppf_est = ppf_values(cdf_est, data["bin_edges"], quantiles)

    # Compute pdf_values
    pdf_est = pdf_from_ppf(quantiles, ppf_est, data["bin_edges"])

    # Assert pdf-values matching true pdf within +-1%
    assert np.allclose(pdf_est, data["binned_pdfs"][0], atol=1e-2)


def test_norm_pdf(data):
    from pyirf.interpolation import norm_pdf

    assert np.allclose(norm_pdf(2 * data["binned_pdfs"][0]), data["binned_pdfs"][0])
    assert np.allclose(norm_pdf(np.zeros(5)), 0)


def test_interpolate_binned_pdf(data):
    from pyirf.interpolation import interpolate_binned_pdf
    from pyirf.binning import bin_center

    interp = interpolate_binned_pdf(
        edges=data["bin_edges"],
        binned_pdfs=data["binned_pdfs"][[0, 2], :],
        grid_points=data["grid_points"][[0, 2]],
        target_point=data["grid_points"][1],
        axis=-1,
        quantile_resolution=1e-3,
    )

    bin_mids = bin_center(data["bin_edges"])
    bin_width = np.diff(data["bin_edges"])[0]

    # Estimate mean and standart_deviation from interpolant
    interp_mean = np.average(bin_mids, weights=interp)
    interp_std = np.sqrt(np.average((bin_mids - interp_mean) ** 2, weights=interp))

    # Assert they match the truth within one bin of uncertainty
    assert np.isclose(interp_mean, data["means"][1], atol=bin_width)
    assert np.isclose(interp_std, data["stds"][1], atol=bin_width)


@pytest.mark.parametrize(
    "params, grid_points, target_point, expected",
    [
        (  # Model: param = 10 + 5 * (grid - 1)
            np.array([10, 20]),
            np.array([1, 3]),
            np.array([2]),
            15,
        ),
        (  # Model: param = 10 + 5 * (grid - 1), with extrapolation
            np.array([10, 20]),
            np.array([1, 3]),
            np.array([4]),
            25,
        ),
        (  # Model: param[0] = 10 + 5 * (grid - 1), param[1] = 5 + 2.5 * (grid - 1)
            np.array([[10, 5], [20, 10]]),
            np.array([1, 3]),
            np.array([2]),
            np.array([15, 7.5]),
        ),
        (  # Model: param[0] = 10 + 5 * (grid - 1), param[1] = 5 + 2.5 * (grid - 1),
            # with extrapolation
            np.array([[10, 5], [20, 10]]),
            np.array([1, 3]),
            np.array([0]),
            np.array([5, 2.5]),
        ),
        (  # Model: param = grid[0] + grid[1]
            np.array([0, 2, 2, 4]),
            np.array([[0, 0], [2, 0], [0, 2], [2, 2]]),
            np.array([-1, 1]),
            np.array([0]),
        ),
        (  # Model: param[0] = grid[0] + grid[1], param[1] = 5 + grid[1] * 2.5
            np.array([[0, 5], [2, 5], [2, 10], [4, 10]]),
            np.array([[0, 0], [2, 0], [0, 2], [2, 2]]),
            np.array([-1, 1]),
            np.array([0, 7.5]),
        ),
    ],
)
def test_interpolate_parametrized_pdf(params, grid_points, target_point, expected):
    from pyirf.interpolation import interpolate_parametrized_pdf

    assert np.allclose(
        interpolate_parametrized_pdf(params, grid_points, target_point), expected
    )


def test_interpolate_effective_area_per_energy_and_fov():
    """Test of interpolating of effective area using dummy model files."""
    n_en = 20
    n_th = 1
    en = np.logspace(-2, 2, n_en)
    # applying a simple sigmoid function
    aeff0 = 1.0e4 / (1 + 1 / en ** 2) * u.Unit("m2")

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
