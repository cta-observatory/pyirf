import numpy as np
import pytest
from pyirf.binning import bin_center
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
    from pyirf.interpolation.quantile_interpolator import cdf_values

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
    from pyirf.interpolation.quantile_interpolator import cdf_values, ppf_values

    # Create quantiles, ignore the 0% and 100% quantile as they are analytically +- inf
    quantiles = np.linspace(0, 1, 10)[1:-2]

    # True ppf-values
    ppf_true = data["distributions"][0].ppf(quantiles)
    bin_mids = bin_center(data["bin_edges"])

    # Estimated ppf-values
    cdf_est = cdf_values(data["binned_pdfs"][0])
    ppf_est = ppf_values(bin_mids, cdf_est, quantiles)

    # Assert truth and estimation match allowing for +- bin_width deviation
    assert np.allclose(ppf_true, ppf_est, atol=np.diff(data["bin_edges"])[0])


def test_pdf_from_ppf(data):
    from pyirf.interpolation.quantile_interpolator import (
        cdf_values,
        pdf_from_ppf,
        ppf_values,
    )

    # Create quantiles
    quantiles = np.linspace(0, 1, 1000)
    bin_mids = bin_center(data["bin_edges"])

    # Estimate ppf-values
    cdf_est = cdf_values(data["binned_pdfs"][0])
    ppf_est = ppf_values(bin_mids, cdf_est, quantiles)

    # Compute pdf_values
    pdf_est = pdf_from_ppf(data["bin_edges"], ppf_est, quantiles)

    # Assert pdf-values matching true pdf within +-1%
    assert np.allclose(pdf_est, data["binned_pdfs"][0], atol=1e-2)


def test_norm_pdf(data):
    from pyirf.interpolation.quantile_interpolator import norm_pdf

    assert np.allclose(norm_pdf(2 * data["binned_pdfs"][0]), data["binned_pdfs"][0])
    assert np.allclose(norm_pdf(np.zeros(5)), 0)


def test_interpolate_binned_pdf(data):
    from pyirf.interpolation import QuantileInterpolator

    interpolator = QuantileInterpolator(
        grid_points=data["grid_points"][[0, 2]],
        bin_edges=data["bin_edges"],
        bin_contents=data["binned_pdfs"][[0, 2], :],
        quantile_resolution=1e-3,
    )

    interp = interpolator(
        target_point=np.array([data["grid_points"][1]]),
    ).squeeze()

    bin_mids = bin_center(data["bin_edges"])
    bin_width = np.diff(data["bin_edges"])[0]

    # Estimate mean and standart_deviation from interpolant
    interp_mean = np.average(bin_mids, weights=interp)
    interp_std = np.sqrt(np.average((bin_mids - interp_mean) ** 2, weights=interp))

    # Assert they match the truth within one bin of uncertainty
    assert np.isclose(interp_mean, data["means"][1], atol=bin_width)
    assert np.isclose(interp_std, data["stds"][1], atol=bin_width)
