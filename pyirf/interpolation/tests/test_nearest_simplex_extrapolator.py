import numpy as np
import pytest
from scipy.stats import norm


def expected_mean(a, b):
    return 5 + (a / 5) + (b / 15)


def expected_std(a, b):
    return 1 + 0.05 * (a + b)


def binned_normal_pdf(mean_std_args, bins):
    pdf = np.diff(
        norm(loc=expected_mean(*mean_std_args), scale=expected_std(*mean_std_args)).cdf(
            bins
        )
    )
    return pdf / pdf.sum()


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

    dummy_data_target1 = 3 * slope + 1

    assert np.allclose(interpolant1, dummy_data_target1)
    assert interpolant1.shape == (1, *dummy_data.shape[1:])

    target_point2 = np.array([[-2.5]])
    interpolant2 = interpolator(target_point2)

    dummy_data_target2 = -2.5 * slope + 1

    assert np.allclose(interpolant2, dummy_data_target2)
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

    assert np.allclose(interpolant1.squeeze(), dummy_data_target1)
    assert interpolant1.shape == (1, *dummy_data.shape[1:])

    target_point2 = np.array([[-2.5, -5.5]])
    interpolant2 = interpolator(target_point2)

    dummy_data_target2 = np.array(
        [
            np.dot((m.T * target_point2 + n), np.array([1, 1]))
            for m, n in zip(slope, intercept)
        ]
    ).squeeze()

    assert np.allclose(interpolant2, dummy_data_target2)
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

    # Create template histograms
    bin_contents = np.array(
        [
            [
                [binned_normal_pdf([x, 0], bins), binned_normal_pdf([x + 1, 0], bins)],
                [
                    binned_normal_pdf([x + 10, 0], bins),
                    np.zeros(len(bins) - 1),
                ],
                [np.zeros(len(bins) - 1), np.ones(len(bins) - 1) / (len(bins) - 1)],
            ]
            for x in grid_points
        ]
    )

    # Include dirac-delta moving with grid_point x at [:, 2, 0, x]
    bin_contents[0, 2, 0, 20] = 1
    bin_contents[1, 2, 0, 30] = 1
    bin_contents[2, 2, 0, 40] = 1

    # Zero template histogram at index [0, 0, 0, :], extrapolations "lower"
    # then this bin has consequently to be zero in these bins.
    bin_contents[0, 0, 0, :] = np.zeros(len(bins) - 1)

    extrap = MomentMorphNearestSimplexExtrapolator(
        grid_points=grid_points, bin_contents=bin_contents, bin_edges=bins
    )

    target1 = np.array([10])

    # target1 is below lowest data bin, thus [0, 0, :] needs to be zero
    truth1 = np.array(
        [
            [
                np.zeros(len(bins) - 1),
                binned_normal_pdf([target1 + 1, 0], bins),
            ],
            [
                binned_normal_pdf([target1 + 10, 0], bins),
                np.zeros(len(bins) - 1),
            ],
            [np.zeros(len(bins) - 1), np.ones(len(bins) - 1) / (len(bins) - 1)],
        ]
    )

    # Inlcude dirac-delta
    truth1[2, 0, 10] = 1

    res1 = extrap(target1)

    expected_norms1 = np.array([[0, 1], [1, 0], [1, 1]])
    assert np.allclose(np.sum(res1, axis=-1), expected_norms1)
    assert np.all(np.isfinite(res1))
    assert res1.shape == (1, *bin_contents.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res1.squeeze(), truth1, atol=1e-3, rtol=1e-5)

    target2 = np.array([45])

    truth2 = np.array(
        [
            [
                binned_normal_pdf([target2, 0], bins),
                binned_normal_pdf([target2 + 1, 0], bins),
            ],
            [
                binned_normal_pdf([target2 + 10, 0], bins),
                np.zeros(len(bins) - 1),
            ],
            [np.zeros(len(bins) - 1), np.ones(len(bins) - 1) / (len(bins) - 1)],
        ]
    )

    # Inlcude dirac-delta
    truth2[2, 0, 45] = 1

    res2 = extrap(target2)

    expected_norms2 = np.array([[1, 1], [1, 0], [1, 1]])
    assert np.allclose(np.sum(res2, axis=-1), expected_norms2)
    assert np.all(np.isfinite(res2))
    assert res2.shape == (1, *bin_contents.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res2.squeeze(), truth2, atol=1e-3, rtol=1e-5)


def test_MomentMorphNearestSimplexExtrapolator_2DGrid(bins):
    """Test ParametrizedNearestSimplexExtrapolator on a 2D Grid"""
    from pyirf.interpolation import MomentMorphNearestSimplexExtrapolator

    grid_points = np.array([[20, 20], [30, 10], [40, 20], [50, 10]])

    # Create template histograms
    bin_contents = np.array(
        [
            [
                [binned_normal_pdf(x, bins), binned_normal_pdf([x[0] + 1, x[1]], bins)],
                [
                    binned_normal_pdf([x[0] + 10, x[1]], bins),
                    np.zeros(len(bins) - 1),
                ],
            ]
            for x in grid_points
        ]
    )

    # Zero template histogram at index [0, 0, 0, :] (the [20, 20] point),
    # so that targets interpolated from left simplex should also be zero at [0, 0, :]
    bin_contents[0, 0, 0, :] = np.zeros(len(bins) - 1)

    extrap = MomentMorphNearestSimplexExtrapolator(
        grid_points=grid_points, bin_contents=bin_contents, bin_edges=bins
    )

    target1 = np.array([25, 10])

    # target1 is extrapolated from left simplex, thus [0, 0, :] needs to be zero
    truth1 = np.array(
        [
            [
                np.zeros(len(bins) - 1),
                binned_normal_pdf([target1[0] + 1, target1[1]], bins),
            ],
            [
                binned_normal_pdf([target1[0] + 10, target1[1]], bins),
                np.zeros(len(bins) - 1),
            ],
        ]
    )

    res1 = extrap(target1)

    expected_norms1 = np.array([[0, 1], [1, 0]])
    assert np.allclose(np.sum(res1, axis=-1), expected_norms1)
    assert np.all(np.isfinite(res1))
    assert res1.shape == (1, *bin_contents.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res1.squeeze(), truth1, atol=1e-3, rtol=1e-5)

    target2 = np.array([45, 20])

    truth2 = np.array(
        [
            [
                binned_normal_pdf(target2, bins),
                binned_normal_pdf([target2[0] + 1, target2[1]], bins),
            ],
            [
                binned_normal_pdf([target2[0] + 10, target2[1]], bins),
                np.zeros(len(bins) - 1),
            ],
        ]
    )

    res2 = extrap(target2)

    expected_norms2 = np.array([[1, 1], [1, 0]])
    assert np.allclose(np.sum(res2, axis=-1), expected_norms2)
    assert np.all(np.isfinite(res2))
    assert res2.shape == (1, *bin_contents.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res2.squeeze(), truth2, atol=1e-3, rtol=1e-5)


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
            grid_points=grid_points, bin_contents=dummy_data, bin_edges=bin_edges
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

    bin_contents = np.array(
        [
            [binned_normal_pdf([x, 0], bins), binned_normal_pdf([x + 10, 0], bins)]
            for x in grid_points
        ]
    )

    extrap = MomentMorphNearestSimplexExtrapolator(
        grid_points=grid_points, bin_contents=bin_contents, bin_edges=bins
    )

    with pytest.warns(
        match="Some bin-entries where below zero after extrapolation and "
        "thus cut off. Check result carefully."
    ):
        res = extrap(np.array([0]))

    assert np.allclose(np.sum(res, axis=-1), 1)
    assert np.all(res >= 0)
