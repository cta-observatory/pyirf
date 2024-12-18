import numpy as np
import pytest
from scipy.stats import norm

from pyirf.interpolation.base_interpolators import PDFNormalization


def expected_mean(a, b):
    return 5 + (a / 5) + (b / 15)


def expected_std(a, b):
    return 1 + 0.05 * (a + b)


def binned_normal_pdf(a, b, bins):
    dist = norm(
        loc=expected_mean(a, b),
        scale=expected_std(a, b),
    )
    pdf = np.diff(dist.cdf(bins))
    return pdf / np.diff(bins)


@pytest.fixture
def bins():
    return np.linspace(-10, 40, 1001)


@pytest.fixture
def simple_1D_data(bins):
    grid = np.array([[20], [40]])
    target = np.array([30])
    binned_pdf = np.array([binned_normal_pdf(x, 0, bins) for x in grid])

    truth = binned_normal_pdf(target, 0, bins)

    return {
        "grid": grid,
        "target": target,
        "binned_pdf": binned_pdf,
        "truth": truth,
    }


@pytest.fixture
def simple_2D_data(bins):
    grid = np.array([[20, 20], [60, 20], [40, 60]])
    target = np.array([25, 25])
    binned_pdf = np.array([binned_normal_pdf(*x, bins) for x in grid])

    truth = binned_normal_pdf(*target, bins)

    return {
        "grid": grid,
        "target": target,
        "binned_pdf": binned_pdf,
        "truth": truth,
    }


def test_estimate_mean_std(bins):
    from pyirf.interpolation.moment_morph_interpolator import _estimate_mean_std

    grid = np.array([[20], [40]])
    binned_pdf = np.array(
        [
            [
                [binned_normal_pdf(x, 0, bins), binned_normal_pdf(x + 1, 0, bins)],
                [
                    binned_normal_pdf(x + 1.5, 0, bins),
                    binned_normal_pdf(x + 10, 0, bins),
                ],
            ]
            for x in grid
        ]
    )

    true_mean = np.array(
        [
            [
                [expected_mean(x, 0), expected_mean(x + 1, 0)],
                [expected_mean(x + 1.5, 0), expected_mean(x + 10, 0)],
            ]
            for x in grid
        ]
    ).squeeze()

    true_std = np.array(
        [
            [
                [expected_std(x, 0), expected_std(x + 1, 0)],
                [expected_std(x + 1.5, 0), expected_std(x + 10, 0)],
            ]
            for x in grid
        ]
    ).squeeze()

    mean, std = _estimate_mean_std(bins, binned_pdf, PDFNormalization.AREA)

    # Assert estimation and truth within one bin
    np.testing.assert_allclose(mean, true_mean, atol=np.diff(bins)[0] / 2)
    np.testing.assert_allclose(std, true_std, atol=np.diff(bins)[0] / 2)


def test_lookup():
    from pyirf.interpolation.moment_morph_interpolator import _lookup

    bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    binned_pdf = np.array(
        [
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [[6, 7, 8, 9, 10], [1, 2, 3, 4, 5]],
        ]
    )

    x = np.array(
        [
            [[0.05, 0.15, 0.25, 0.35, 0.45], [0.15, 0.17, 0.21, 0.26, 0.35]],
            [[0.25, 0.25, 0.25, 0.25, 0.25], [-0.1, 0.25, 0.25, 0.25, 0.6]],
        ]
    )

    truth = np.array(
        [
            [[1, 2, 3, 4, 5], [7, 7, 8, 8, 9]],
            [[8, 8, 8, 8, 8], [0, 3, 3, 3, 0]],  # Under/Overflow is set to 0
        ]
    )

    np.testing.assert_allclose(_lookup(bins, binned_pdf, x), truth)


def test_linesegment_1D_interpolation_coefficients():
    from pyirf.interpolation.moment_morph_interpolator import (
        linesegment_1D_interpolation_coefficients,
    )

    grid_points = np.array([[20], [40]])
    target_point = np.array([[20]])

    np.testing.assert_allclose(
        linesegment_1D_interpolation_coefficients(grid_points, target_point),
        np.array([[1, 0]]),
    )

    target_point = np.array([[40]])

    np.testing.assert_allclose(
        linesegment_1D_interpolation_coefficients(grid_points, target_point),
        np.array([[0, 1]]),
    )

    target_point = np.array([[25]])

    # Values taken from eq. (9) and (10) in Moment Morph paper Baak et al.
    # https://doi.org/10.1016/j.nima.2014.10.033
    mfrac = (target_point[0, 0] - grid_points[0, 0]) / (
        grid_points[1, 0] - grid_points[0, 0]
    )
    cs = np.array([[1 - mfrac, mfrac]])

    res = linesegment_1D_interpolation_coefficients(grid_points, target_point)

    np.testing.assert_allclose(res, cs)
    assert np.sum(res) == 1
    assert np.all(np.logical_and(res < 1, res > 0))


def test_barycentric_2D_interpolation_coefficients():
    from pyirf.interpolation.moment_morph_interpolator import (
        barycentric_2D_interpolation_coefficients,
    )

    grid_points = np.array([[20, 20], [60, 20], [40, 60]])
    target_point = np.array([20, 20])

    np.testing.assert_allclose(
        barycentric_2D_interpolation_coefficients(grid_points, target_point),
        np.array([1, 0, 0]),
    )

    target_point = np.array([60, 20])

    np.testing.assert_allclose(
        barycentric_2D_interpolation_coefficients(grid_points, target_point),
        np.array([0, 1, 0]),
    )

    target_point = np.array([40, 60])

    np.testing.assert_allclose(
        barycentric_2D_interpolation_coefficients(grid_points, target_point),
        np.array([0, 0, 1]),
    )

    target_point = np.array([40, 20])

    np.testing.assert_allclose(
        barycentric_2D_interpolation_coefficients(grid_points, target_point),
        np.array([0.5, 0.5, 0]),
    )

    # Barycenter of triangle
    target_point = np.array([40, 100 / 3])
    res = barycentric_2D_interpolation_coefficients(grid_points, target_point)

    np.testing.assert_allclose(res, np.array([1, 1, 1]) / 3)


def test_moment_morph_estimation1D(bins, simple_1D_data):
    from pyirf.interpolation.moment_morph_interpolator import (
        linesegment_1D_interpolation_coefficients,
        moment_morph_estimation,
    )

    grid, target, binned_pdf, truth = simple_1D_data.values()

    coeffs = linesegment_1D_interpolation_coefficients(grid, target)
    res = moment_morph_estimation(bins, binned_pdf, coeffs, PDFNormalization.AREA)

    np.testing.assert_almost_equal(np.sum(res * np.diff(bins)), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_moment_morph_estimation2D(bins, simple_2D_data):
    from pyirf.interpolation.moment_morph_interpolator import (
        barycentric_2D_interpolation_coefficients,
        moment_morph_estimation,
    )

    grid, target, binned_pdf, truth = simple_2D_data.values()

    coeffs = barycentric_2D_interpolation_coefficients(grid, target)
    res = moment_morph_estimation(bins, binned_pdf, coeffs, PDFNormalization.AREA)

    np.testing.assert_almost_equal(np.sum(res * np.diff(bins)), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator1D(bins, simple_1D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, binned_pdf, truth = simple_1D_data.values()

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, binned_pdf=binned_pdf
    )

    res = interp(target)

    np.testing.assert_almost_equal(np.sum(res * np.diff(bins)), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator1D_dirac_delta_input():
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[1], [3]])
    bin_edges = np.array([0, 1, 2, 3, 4])
    binned_pdf = np.array(
        [
            [[0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25]],
            [[0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25]],
        ]
    )
    target = np.array([2])

    interp = MomentMorphInterpolator(grid, bin_edges, binned_pdf)
    res = interp(target)

    expected = np.array([[[0, 0, 1, 0], [0.25, 0.25, 0.25, 0.25]]])
    np.testing.assert_allclose(res, expected)


def test_MomentMorphInterpolator1D_all_empty(bins, simple_1D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, _, _ = simple_1D_data.values()
    binned_pdf = np.array([np.zeros(len(bins) - 1) for _ in grid])

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, binned_pdf=binned_pdf
    )

    res = interp(target)

    np.testing.assert_allclose(res, 0)


def test_MomentMorphInterpolator1D_partially_empty(bins, simple_1D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, binned_pdf, _ = simple_1D_data.values()

    binned_pdf[0, :] = np.zeros(len(bins) - 1)

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, binned_pdf=binned_pdf
    )

    res = interp(target)

    np.testing.assert_allclose(res, 0)


def test_MomentMorphInterpolator1D_mixed_data(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20], [40]])
    target = np.array([30])

    # Create template histograms
    binned_pdf = np.array(
        [
            [
                [binned_normal_pdf(x, 0, bins), binned_normal_pdf(x + 1, 0, bins)],
                [binned_normal_pdf(x + 10, 0, bins), binned_normal_pdf(x + 2, 0, bins)],
            ]
            for x in grid
        ]
    )

    # Make template histograms at indizes [:, 1, 1, :] all zeroed
    binned_pdf[:, 1, 1, :] = np.zeros(len(bins) - 1)

    # Zero template histogram at index [1, 0, 0, :]
    binned_pdf[1, 0, 0, :] = np.zeros(len(bins) - 1)

    truth = np.array(
        [
            [
                binned_normal_pdf(target, 0, bins),
                binned_normal_pdf(target + 1, 0, bins),
            ],
            [
                binned_normal_pdf(target + 10, 0, bins),
                binned_normal_pdf(target + 2, 0, bins),
            ],
        ]
    )

    # Expect zeros for at least partially zeroed input templates
    truth[0, 0, :] = np.zeros(len(bins) - 1)
    truth[1, 1, :] = np.zeros(len(bins) - 1)

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, binned_pdf=binned_pdf
    )

    res = interp(target)

    expected_norms = np.array([[[0, 1], [1, 0]]])
    np.testing.assert_allclose(np.sum(res * np.diff(bins), axis=-1), expected_norms)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, *binned_pdf.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator1D_extended_grid_extradims(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20], [40], [60], [80]])
    target = np.array([25])
    binned_pdf = np.array(
        [
            [
                [binned_normal_pdf(x, 0, bins), binned_normal_pdf(x + 1, 0, bins)],
                [
                    binned_normal_pdf(x + 10, 0, bins),
                    binned_normal_pdf(x + 2, 0, bins),
                ],
            ]
            for x in grid
        ]
    )

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, binned_pdf=binned_pdf
    )

    truth = np.array(
        [
            [
                binned_normal_pdf(target, 0, bins),
                binned_normal_pdf(target + 1, 0, bins),
            ],
            [
                binned_normal_pdf(target + 10, 0, bins),
                binned_normal_pdf(target + 2, 0, bins),
            ],
        ]
    )

    res = interp(target)

    np.testing.assert_allclose(np.sum(res * np.diff(bins), axis=-1), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, *binned_pdf.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator2D(bins, simple_2D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, binned_pdf, truth = simple_2D_data.values()

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, binned_pdf=binned_pdf
    )

    res = interp(target)

    np.testing.assert_almost_equal(np.sum(res * np.diff(bins)), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator2D_partially_empty(bins, simple_2D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, binned_pdf, _ = simple_2D_data.values()

    binned_pdf[0, :] = np.zeros(len(bins) - 1)

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, binned_pdf=binned_pdf
    )

    res = interp(target)

    np.testing.assert_allclose(res, 0)


def test_MomentMorphInterpolator2D_all_empty(bins, simple_2D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, _, _ = simple_2D_data.values()
    binned_pdf = np.array([np.zeros(len(bins) - 1) for _ in grid])

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, binned_pdf=binned_pdf
    )

    res = interp(target)

    np.testing.assert_allclose(res, 0)


def test_MomentMorphInterpolator2D_mixed(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20, 20], [60, 20], [40, 60]])
    target = np.array([25, 25])

    # Create template histograms
    binned_pdf = np.array(
        [
            [
                [binned_normal_pdf(a, b, bins), binned_normal_pdf(a + 1, b, bins)],
                [binned_normal_pdf(a + 10, b, bins), binned_normal_pdf(a + 2, b, bins)],
            ]
            for a, b in grid
        ]
    )

    # Make template histograms at indizes [:, 1, 1, :] all zeroed
    binned_pdf[:, 1, 1, :] = np.zeros(len(bins) - 1)

    # Zero template histogram at index [1, 0, 0, :]
    binned_pdf[1, 0, 0, :] = np.zeros(len(bins) - 1)

    truth = np.array(
        [
            [
                binned_normal_pdf(*target, bins),
                binned_normal_pdf(target[0] + 1, target[1], bins),
            ],
            [
                binned_normal_pdf(target[0] + 10, target[1], bins),
                binned_normal_pdf(target[0] + 2, target[1], bins),
            ],
        ]
    )

    # Expect zeros for at least partially zeroed input templates
    truth[0, 0, :] = np.zeros(len(bins) - 1)
    truth[1, 1, :] = np.zeros(len(bins) - 1)

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, binned_pdf=binned_pdf
    )

    res = interp(target)

    expected_norms = np.array([[[0, 1], [1, 0]]])
    np.testing.assert_allclose(np.sum(res * np.diff(bins), axis=-1), expected_norms)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, *binned_pdf.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator1D_extended_grid(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20], [40], [60], [80]])
    target = np.array([25])
    binned_pdf = np.array([binned_normal_pdf(x, 0, bins) for x in grid])

    interp = MomentMorphInterpolator(
        grid_points=grid,
        bin_edges=bins,
        binned_pdf=binned_pdf,
    )

    res = interp(target)
    truth = binned_normal_pdf(target, 0, bins)

    np.testing.assert_almost_equal(np.sum(res * np.diff(bins)), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator2D_extended_grid(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20, 20], [40, 20], [30, 40], [50, 20], [45, 40]])
    target = np.array([25, 25])
    binned_pdf = np.array([binned_normal_pdf(a, b, bins) for a, b in grid])

    interp = MomentMorphInterpolator(
        grid_points=grid,
        bin_edges=bins,
        binned_pdf=binned_pdf,
    )

    res = interp(target)
    truth = binned_normal_pdf(*target, bins)

    np.testing.assert_almost_equal(np.sum(res * np.diff(bins)), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator2D_extended_grid_extradims(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20, 20], [40, 20], [30, 40], [50, 20], [45, 40]])
    a = 25
    b = 25
    target = np.array([a, b])
    binned_pdf = np.array(
        [
            [
                [binned_normal_pdf(a, b, bins), binned_normal_pdf(a + 1, b, bins)],
                [
                    binned_normal_pdf(a + 10, b, bins),
                    binned_normal_pdf(a + 2, b + 5, bins),
                ],
            ]
            for a, b in grid
        ]
    )

    interp = MomentMorphInterpolator(
        grid_points=grid,
        bin_edges=bins,
        binned_pdf=binned_pdf,
    )

    truth = np.array(
        [
            [
                binned_normal_pdf(a, b, bins),
                binned_normal_pdf(a + 1, b, bins),
            ],
            [
                binned_normal_pdf(a + 10, b, bins),
                binned_normal_pdf(a + 2, b + 5, bins),
            ],
        ]
    )

    res = interp(target)

    np.testing.assert_allclose(np.sum(res * np.diff(bins), axis=-1), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, *binned_pdf.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator3D():
    from pyirf.interpolation import MomentMorphInterpolator

    bins = np.linspace(0, 1, 11)

    grid = np.array([[0, 0, 0], [0, 20, 0], [20, 0, 0], [20, 20, 0], [10, 10, 10]])

    binned_pdf = np.array([np.ones(len(bins) - 1) / (len(bins) - 1) for _ in grid])

    with pytest.raises(
        NotImplementedError,
        match="Interpolation in more then two dimension not impemented.",
    ):
        MomentMorphInterpolator(
            grid_points=grid,
            bin_edges=bins,
            binned_pdf=binned_pdf,
        )
