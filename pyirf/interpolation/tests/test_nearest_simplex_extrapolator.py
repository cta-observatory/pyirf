import numpy as np
import pytest
from scipy.stats import norm


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


def test_ParametrizedNearestSimplexExtrapolator_1DGrid():
    """Test ParametrizedNearestSimplexExtrapolator on a 1D Grid with linearly varying data"""
    from pyirf.interpolation import ParametrizedNearestSimplexExtrapolator

    grid_points = np.array([[0], [1], [2]])

    slope = np.array([[[0, 1], [1, 1]], [[0, 2], [2, 3]], [[0, 3], [3, 5]]])
    dummy_data1 = grid_points[0] * slope + 1
    dummy_data2 = grid_points[1] * slope + 1
    dummy_data3 = grid_points[2] * slope + 1

    dummy_data = np.array([dummy_data1, dummy_data2, dummy_data3])

    interpolator = ParametrizedNearestSimplexExtrapolator(
        grid_points=grid_points,
        params=dummy_data,
    )
    target_point1 = np.array([3])
    interpolant1 = interpolator(target_point1)

    dummy_data_target1 = (3 * slope + 1)[np.newaxis, :]

    np.testing.assert_allclose(interpolant1, dummy_data_target1)
    assert interpolant1.shape == (1, *dummy_data.shape[1:])

    target_point2 = np.array([[-2.5]])
    interpolant2 = interpolator(target_point2)

    dummy_data_target2 = (-2.5 * slope + 1)[np.newaxis, :]

    np.testing.assert_allclose(interpolant2, dummy_data_target2)
    assert interpolant2.shape == (1, *dummy_data.shape[1:])


def test_ParametrizedNearestSimplexExtrapolator_2DGrid():
    """Test ParametrizedNearestSimplexExtrapolator on a 2D Grid with independently, linearly
    varying data in both grid dimensions"""
    from pyirf.interpolation import ParametrizedNearestSimplexExtrapolator

    grid_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    slope = np.array([[[0, 1], [1, 1]], [[0, 2], [2, 3]], [[3, 0], [3, 5]]])
    intercept = np.array([[[0, 1], [1, 1]], [[0, -1], [-1, -1]], [[10, 11], [11, 11]]])

    # Create 3 times 2 linear samples, each of the form (mx * px + my * py + nx + ny)
    # with slope m, intercept n at each grid_point p
    dummy_data = np.array(
        [
            np.array(
                [
                    np.dot((m.T * p + n), np.array([1, 1]))
                    for m, n in zip(slope, intercept)
                ]
            ).squeeze()
            for p in grid_points
        ]
    )

    interpolator = ParametrizedNearestSimplexExtrapolator(
        grid_points=grid_points,
        params=dummy_data,
    )

    target_point1 = np.array([5.5, 3.5])
    interpolant1 = interpolator(target_point1)
    dummy_data_target1 = np.array(
        [
            np.dot((m.T * target_point1 + n), np.array([1, 1]))
            for m, n in zip(slope, intercept)
        ]
    ).squeeze()

    np.testing.assert_allclose(interpolant1.squeeze(), dummy_data_target1)
    assert interpolant1.shape == (1, *dummy_data.shape[1:])

    target_point2 = np.array([[-2.5, -5.5]])
    interpolant2 = interpolator(target_point2)

    dummy_data_target2 = np.array(
        [
            [
                np.dot((m.T * target_point2 + n), np.array([1, 1]))
                for m, n in zip(slope, intercept)
            ]
        ]
    )

    np.testing.assert_allclose(interpolant2, dummy_data_target2)
    assert interpolant2.shape == (1, *dummy_data.shape[1:])


def test_ParametrizedNearestSimplexExtrapolator_3DGrid():
    """Test ParametrizedNearestSimplexExtrapolator on a 3D Grid, which is currently
    not implemented"""
    from pyirf.interpolation import ParametrizedNearestSimplexExtrapolator

    grid_points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    dummy_data = np.array([[0, 1], [1, 1], [0, 2], [2, 3]])

    with pytest.raises(
        NotImplementedError,
        match="Extrapolation in more then two dimension not impemented.",
    ):
        ParametrizedNearestSimplexExtrapolator(
            grid_points=grid_points,
            params=dummy_data,
        )


def test_MomentMorphNearestSimplexExtrapolator_1DGrid(bins):
    """Test ParametrizedNearestSimplexExtrapolator on a 1D Grid"""
    from pyirf.interpolation import MomentMorphNearestSimplexExtrapolator

    grid_points = np.array([[20], [30], [40]])

    n_bins = len(bins) - 1
    bin_width = np.diff(bins)

    # Create template histograms
    binned_pdf = np.array(
        [
            [
                [binned_normal_pdf(x, 0, bins), binned_normal_pdf(x + 1, 0, bins)],
                [
                    binned_normal_pdf(x + 10, 0, bins),
                    np.zeros(n_bins),
                ],
                [np.zeros(n_bins), np.ones(n_bins) / n_bins / bin_width],
            ]
            for x in grid_points
        ]
    )

    # Include dirac-delta moving with grid_point x at [:, 2, 0, x]
    for i, bin_idx in enumerate(grid_points.ravel()):
        binned_pdf[i, 2, 0, bin_idx] = 1 / bin_width[bin_idx]

    # Zero template histogram at index [0, 0, 0, :], extrapolations "lower"
    # then this bin has consequently to be zero in these bins.
    binned_pdf[0, 0, 0, :] = 0.0

    extrap = MomentMorphNearestSimplexExtrapolator(
        grid_points=grid_points, binned_pdf=binned_pdf, bin_edges=bins
    )

    target1 = np.array([10])

    # target1 is below lowest data bin, thus [0, 0, :] needs to be zero
    truth1 = np.array(
        [
            [
                np.zeros(n_bins),
                binned_normal_pdf(target1 + 1, 0, bins),
            ],
            [
                binned_normal_pdf(target1 + 10, 0, bins),
                np.zeros_like(bin_width),
            ],
            [np.zeros(n_bins), np.ones(n_bins) / n_bins / bin_width],
        ]
    )

    # Inlcude dirac-delta
    delta = np.zeros(len(bins) - 1)
    delta[target1[0]] = 1
    delta /= bin_width
    truth1[2, 0] = delta

    res1 = extrap(target1)

    expected_norms1 = np.array([[[0, 1], [1, 0], [1, 1]]])
    np.testing.assert_allclose(np.sum(res1 * np.diff(bins), axis=-1), expected_norms1)
    assert np.all(np.isfinite(res1))
    assert res1.shape == (1, *binned_pdf.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res1.squeeze(), truth1, atol=4e-3, rtol=1e-5)

    target2 = np.array([45])

    truth2 = np.array(
        [
            [
                binned_normal_pdf(target2, 0, bins),
                binned_normal_pdf(target2 + 1, 0, bins),
            ],
            [
                binned_normal_pdf(target2 + 10, 0, bins),
                np.zeros(n_bins),
            ],
            [np.zeros(n_bins), np.ones(n_bins) / n_bins / bin_width],
        ]
    )

    # Inlcude dirac-delta
    truth2[2, 0, 45] = 1 / bin_width[45]

    res2 = extrap(target2)

    expected_norms2 = np.array([[[1, 1], [1, 0], [1, 1]]])
    np.testing.assert_allclose(np.sum(res2 * bin_width, axis=-1), expected_norms2)
    assert np.all(np.isfinite(res2))
    assert res2.shape == (1, *binned_pdf.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res2.squeeze(), truth2, atol=1e-3, rtol=1e-5)


def test_MomentMorphNearestSimplexExtrapolator_2DGrid(bins):
    """Test ParametrizedNearestSimplexExtrapolator on a 2D Grid"""
    from pyirf.interpolation import MomentMorphNearestSimplexExtrapolator

    n_bins = len(bins) - 1
    bin_width = np.diff(bins)
    grid_points = np.array([[20, 20], [30, 10], [40, 20], [50, 10]])

    # Create template histograms
    binned_pdf = np.array(
        [
            [
                [binned_normal_pdf(a, b, bins), binned_normal_pdf(a + 1, b, bins)],
                [binned_normal_pdf(a + 10, b, bins), np.zeros(n_bins)],
            ]
            for a, b in grid_points
        ]
    )

    # Zero template histogram at index [0, 0, 0, :] (the [20, 20] point),
    # so that targets interpolated from left simplex should also be zero at [0, 0, :]
    binned_pdf[0, 0, 0, :] = np.zeros(n_bins)

    extrap = MomentMorphNearestSimplexExtrapolator(
        grid_points=grid_points, binned_pdf=binned_pdf, bin_edges=bins
    )

    target1 = np.array([25, 10])

    # target1 is extrapolated from left simplex, thus [0, 0, :] needs to be zero
    truth1 = np.array(
        [
            [
                np.zeros(n_bins),
                binned_normal_pdf(target1[0] + 1, target1[1], bins),
            ],
            [
                binned_normal_pdf(target1[0] + 10, target1[1], bins),
                np.zeros(n_bins),
            ],
        ]
    )

    res1 = extrap(target1)

    expected_norms1 = np.array([[[0, 1], [1, 0]]])
    np.testing.assert_allclose(np.sum(res1 * bin_width, axis=-1), expected_norms1)
    assert np.all(np.isfinite(res1))
    assert res1.shape == (1, *binned_pdf.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res1.squeeze(), truth1, atol=1e-3, rtol=1e-5)

    target2 = np.array([45, 20])

    truth2 = np.array(
        [
            [
                binned_normal_pdf(*target2, bins),
                binned_normal_pdf(target2[0] + 1, target2[1], bins),
            ],
            [
                binned_normal_pdf(target2[0] + 10, target2[1], bins),
                np.zeros(n_bins),
            ],
        ]
    )

    res2 = extrap(target2)

    expected_norms2 = np.array([[[1, 1], [1, 0]]])
    np.testing.assert_allclose(np.sum(res2 * bin_width, axis=-1), expected_norms2)
    assert np.all(np.isfinite(res2))
    assert res2.shape == (1, *binned_pdf.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    np.testing.assert_allclose(res2.squeeze(), truth2, atol=1e-3, rtol=1e-5)


def test_MomentMorphNearestSimplexExtrapolator_3DGrid():
    """Test MomentMorphNearestSimplexExtrapolator on a 3D Grid, which is currently
    not implemented"""
    from pyirf.interpolation import MomentMorphNearestSimplexExtrapolator

    grid_points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    dummy_data = np.array([[0, 1], [1, 1], [0, 2], [2, 3]])
    bin_edges = np.array([0, 0.5, 1])

    with pytest.raises(
        NotImplementedError,
        match="Extrapolation in more then two dimension not impemented.",
    ):
        MomentMorphNearestSimplexExtrapolator(
            grid_points=grid_points, binned_pdf=dummy_data, bin_edges=bin_edges
        )


def test_MomentMorphNearestSimplexExtrapolator_below_zero_warning(bins):
    """
    Tests if ParametrizedNearestSimplexExtrapolator detects below-zero
    bin entries that arise with high extrapolation distances, cuts them off
    and renormalizes the extrapolation result after issuing a warning
    """
    from pyirf.interpolation import MomentMorphNearestSimplexExtrapolator

    grid_points = np.array([[30], [40]])
    bins = np.linspace(-10, 40, 51)

    binned_pdf = np.array(
        [
            [binned_normal_pdf(x, 0, bins), binned_normal_pdf(x + 10, 0, bins)]
            for x in grid_points
        ]
    )

    extrap = MomentMorphNearestSimplexExtrapolator(
        grid_points=grid_points, binned_pdf=binned_pdf, bin_edges=bins
    )

    with pytest.warns(
        match="Some bin-entries where below zero after extrapolation and "
        "thus cut off. Check result carefully."
    ):
        res = extrap(np.array([0]))

    np.testing.assert_allclose(np.sum(res, axis=-1), 1)
    assert np.all(res >= 0)
