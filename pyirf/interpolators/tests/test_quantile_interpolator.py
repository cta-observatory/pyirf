import numpy as np
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


def test_interpolate_binned_pdf(data):
    from pyirf.interpolators import QuantileInterpolator
    from pyirf.binning import bin_center

    interpolator = QuantileInterpolator(
        grid_points=data["grid_points"][[0, 2]],
        bin_edges=data["bin_edges"],
        bin_contents=data["binned_pdfs"][[0, 2], :],
        quantile_resolution=1e-3,
        axis=-1,
    )

    interp = interpolator(
        target_point=np.array([data["grid_points"][1]]),
    )

    bin_mids = bin_center(data["bin_edges"])
    bin_width = np.diff(data["bin_edges"])[0]

    # Estimate mean and standart_deviation from interpolant
    interp_mean = np.average(bin_mids, weights=interp)
    interp_std = np.sqrt(np.average((bin_mids - interp_mean) ** 2, weights=interp))

    # Assert they match the truth within one bin of uncertainty
    assert np.isclose(interp_mean, data["means"][1], atol=bin_width)
    assert np.isclose(interp_std, data["stds"][1], atol=bin_width)
