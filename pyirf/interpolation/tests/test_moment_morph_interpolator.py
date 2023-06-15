import numpy as np
import pytest
from scipy.stats import norm


def expected_mean(a, b):
    return 5 + (a / 5) + (b / 15)


def expected_std(a, b):
    return 1 + 0.05 * (a + b)


@pytest.fixture
def bins():
    return np.linspace(-10, 40, 1001)


@pytest.fixture
def simple_1D_data(bins):
    grid = np.array([[20], [40]])
    target = np.array([30])
    bin_contents = np.array(
        [
            np.diff(norm(loc=expected_mean(x, 0), scale=expected_std(x, 0)).cdf(bins))
            for x in grid
        ]
    )

    # Assert for probability outside binning
    bin_contents /= bin_contents.sum(axis=1)[:, np.newaxis]

    truth = np.diff(
        norm(loc=expected_mean(target, 0), scale=expected_std(target, 0)).cdf(bins)
    )
    truth /= truth.sum()

    return {
        "grid": grid,
        "target": target,
        "bin_contents": bin_contents,
        "truth": truth,
    }


@pytest.fixture
def simple_2D_data(bins):
    grid = np.array([[20, 20], [60, 20], [40, 60]])
    target = np.array([25, 25])
    bin_contents = np.array(
        [
            np.diff(
                norm(loc=expected_mean(x[0], x[1]), scale=expected_std(x[0], x[1])).cdf(
                    bins
                )
            )
            for x in grid
        ]
    )

    # Assert for probability outside binning
    bin_contents /= bin_contents.sum(axis=1)[:, np.newaxis]

    truth = np.diff(
        norm(
            loc=expected_mean(target[0], target[1]),
            scale=expected_std(target[0], target[1]),
        ).cdf(bins)
    )
    truth /= truth.sum()

    return {
        "grid": grid,
        "target": target,
        "bin_contents": bin_contents,
        "truth": truth,
    }


def test_estimate_mean_std(bins):
    from pyirf.interpolation.moment_morph_interpolator import estimate_mean_std

    grid = np.array([[20], [40]])
    bin_contents = np.array(
        [
            [
                [
                    np.diff(
                        norm(loc=expected_mean(x, 0), scale=expected_std(x, 0)).cdf(
                            bins
                        )
                    ),
                    np.diff(
                        norm(
                            loc=expected_mean(x + 1, 0), scale=expected_std(x + 1, 0)
                        ).cdf(bins)
                    ),
                ],
                [
                    np.diff(
                        norm(
                            loc=expected_mean(x + 1.5, 0),
                            scale=expected_std(x + 1.5, 0),
                        ).cdf(bins)
                    ),
                    np.diff(
                        norm(
                            loc=expected_mean(x + 2, 0), scale=expected_std(x + 2, 0)
                        ).cdf(bins)
                    ),
                ],
            ]
            for x in grid
        ]
    )

    true_mean = np.array(
        [
            [
                [expected_mean(x, 0), expected_mean(x + 1, 0)],
                [expected_mean(x + 1.5, 0), expected_mean(x + 2, 0)],
            ]
            for x in grid
        ]
    ).squeeze()

    true_std = np.array(
        [
            [
                [expected_std(x, 0), expected_std(x + 1, 0)],
                [expected_std(x + 1.5, 0), expected_std(x + 2, 0)],
            ]
            for x in grid
        ]
    ).squeeze()

    mean, std = estimate_mean_std(bins, bin_contents)

    # Assert estimation and truth within one bin
    assert np.allclose(mean, true_mean, atol=np.diff(bins)[0] / 2)
    assert np.allclose(std, true_std, atol=np.diff(bins)[0] / 2)


def test_lookup():
    from pyirf.interpolation.moment_morph_interpolator import lookup

    bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    bin_contents = np.array(
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

    assert np.allclose(lookup(bins, bin_contents, x), truth)


def test_linesegment_1D_interpolation_coefficients():
    from pyirf.interpolation.moment_morph_interpolator import (
        linesegment_1D_interpolation_coefficients,
    )

    grid_points = np.array([[20], [40]])
    target_point = np.array([[20]])

    assert np.allclose(
        linesegment_1D_interpolation_coefficients(grid_points, target_point),
        np.array([1, 0]),
    )

    target_point = np.array([[40]])

    assert np.allclose(
        linesegment_1D_interpolation_coefficients(grid_points, target_point),
        np.array([0, 1]),
    )

    target_point = np.array([[25]])

    # Values taken from eq. (9) and (10) in Moment Morph paper Baak et al.
    # https://doi.org/10.1016/j.nima.2014.10.033
    mfrac = (target_point[0, 0] - grid_points[0, 0]) / (
        grid_points[1, 0] - grid_points[0, 0]
    )
    cs = np.array([1 - mfrac, mfrac])

    res = linesegment_1D_interpolation_coefficients(grid_points, target_point)

    assert np.allclose(res, cs)
    assert np.sum(res) == 1
    assert np.all(np.logical_and(res < 1, res > 0))


def test_baryzentric_2D_interpolation_coefficients():
    from pyirf.interpolation.moment_morph_interpolator import (
        baryzentric_2D_interpolation_coefficients,
    )

    grid_points = np.array([[20, 20], [60, 20], [40, 60]])
    target_point = np.array([20, 20])

    assert np.allclose(
        baryzentric_2D_interpolation_coefficients(grid_points, target_point),
        np.array([1, 0, 0]),
    )

    target_point = np.array([60, 20])

    assert np.allclose(
        baryzentric_2D_interpolation_coefficients(grid_points, target_point),
        np.array([0, 1, 0]),
    )

    target_point = np.array([40, 60])

    assert np.allclose(
        baryzentric_2D_interpolation_coefficients(grid_points, target_point),
        np.array([0, 0, 1]),
    )

    target_point = np.array([40, 20])

    assert np.allclose(
        baryzentric_2D_interpolation_coefficients(grid_points, target_point),
        np.array([0.5, 0.5, 0]),
    )

    # Barycenter of triangle
    target_point = np.array([40, 100 / 3])
    res = baryzentric_2D_interpolation_coefficients(grid_points, target_point)

    assert np.allclose(res, np.array([1, 1, 1]) / 3)


def test_moment_morph_estimation1D(bins, simple_1D_data):
    from pyirf.interpolation.moment_morph_interpolator import (
        linesegment_1D_interpolation_coefficients,
        moment_morph_estimation,
    )

    grid, target, bin_contents, truth = simple_1D_data.values()

    coeffs = linesegment_1D_interpolation_coefficients(grid, target)
    res = moment_morph_estimation(bins, bin_contents, coeffs)

    assert np.isclose(np.sum(res), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_moment_morph_estimation2D(bins, simple_2D_data):
    from pyirf.interpolation.moment_morph_interpolator import (
        baryzentric_2D_interpolation_coefficients,
        moment_morph_estimation,
    )

    grid, target, bin_contents, truth = simple_2D_data.values()

    coeffs = baryzentric_2D_interpolation_coefficients(grid, target)
    res = moment_morph_estimation(bins, bin_contents, coeffs)

    assert np.isclose(np.sum(res), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator1D(bins, simple_1D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, bin_contents, truth = simple_1D_data.values()

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.isclose(np.sum(res), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator1D_dirac_delta_input():
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[1], [3]])
    bin_edges = np.array([0, 1, 2, 3, 4])
    bin_contents = np.array(
        [
            [[0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25]],
            [[0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25]],
        ]
    )
    target = np.array([2])

    interp = MomentMorphInterpolator(grid, bin_edges, bin_contents)
    res = interp(target)

    assert np.allclose(res, np.array([[0, 0, 1, 0], [0.25, 0.25, 0.25, 0.25]]))


def test_MomentMorphInterpolator1D_all_empty(bins, simple_1D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, _, _ = simple_1D_data.values()
    bin_contents = np.array([np.zeros(len(bins) - 1) for _ in grid])

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.allclose(res, 0)


def test_MomentMorphInterpolator1D_partially_empty(bins, simple_1D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, bin_contents, _ = simple_1D_data.values()

    bin_contents[0, :] = np.zeros(len(bins) - 1)

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.allclose(res, 0)


def test_MomentMorphInterpolator1D_mixed_data(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20], [40]])
    target = np.array([30])

    # Create template histograms
    bin_contents = np.array(
        [
            [
                [
                    np.diff(
                        norm(loc=expected_mean(x, 0), scale=expected_std(x, 0)).cdf(
                            bins
                        )
                    ),
                    np.diff(
                        norm(
                            loc=expected_mean(x + 1, 0), scale=expected_std(x + 1, 0)
                        ).cdf(bins)
                    ),
                ],
                [
                    np.diff(
                        norm(
                            loc=expected_mean(x + 1.5, 0),
                            scale=expected_std(x + 1.5, 0),
                        ).cdf(bins)
                    ),
                    np.diff(
                        norm(
                            loc=expected_mean(x + 2, 0), scale=expected_std(x + 2, 0)
                        ).cdf(bins)
                    ),
                ],
            ]
            for x in grid
        ]
    )

    bin_contents /= bin_contents.sum(axis=-1)[..., np.newaxis]

    # Make template histograms at indizes [:, 1, 1, :] all zeroed
    bin_contents[:, 1, 1, :] = np.zeros(len(bins) - 1)

    # Zero template histogram at index [1, 0, 0, :]
    bin_contents[1, 0, 0, :] = np.zeros(len(bins) - 1)

    truth = np.array(
        [
            [
                np.diff(
                    norm(
                        loc=expected_mean(target, 0), scale=expected_std(target, 0)
                    ).cdf(bins)
                ),
                np.diff(
                    norm(
                        loc=expected_mean(target + 1, 0),
                        scale=expected_std(target + 1, 0),
                    ).cdf(bins)
                ),
            ],
            [
                np.diff(
                    norm(
                        loc=expected_mean(target + 1.5, 0),
                        scale=expected_std(target + 1.5, 0),
                    ).cdf(bins)
                ),
                np.diff(
                    norm(
                        loc=expected_mean(target + 2, 0),
                        scale=expected_std(target + 2, 0),
                    ).cdf(bins)
                ),
            ],
        ]
    )
    truth /= truth.sum(axis=-1)[..., np.newaxis]

    # Expect zeros for at least partially zeroed input templates
    truth[0, 0, :] = np.zeros(len(bins) - 1)
    truth[1, 1, :] = np.zeros(len(bins) - 1)

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    expected_norms = np.array([[0, 1], [1, 0]])
    assert np.allclose(np.sum(res, axis=-1), expected_norms)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, *bin_contents.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator1D_extended_grid_extradims(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20], [40], [60], [80]])
    target = np.array([25])
    bin_contents = np.array(
        [
            [
                [
                    np.diff(
                        norm(loc=expected_mean(x, 0), scale=expected_std(x, 0)).cdf(
                            bins
                        )
                    ),
                    np.diff(
                        norm(
                            loc=expected_mean(x + 1, 0), scale=expected_std(x + 1, 0)
                        ).cdf(bins)
                    ),
                ],
                [
                    np.diff(
                        norm(
                            loc=expected_mean(x + 1.5, 0),
                            scale=expected_std(x + 1.5, 0),
                        ).cdf(bins)
                    ),
                    np.diff(
                        norm(
                            loc=expected_mean(x + 2, 0), scale=expected_std(x + 2, 0)
                        ).cdf(bins)
                    ),
                ],
            ]
            for x in grid
        ]
    )

    bin_contents /= bin_contents.sum(axis=-1)[..., np.newaxis]

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    truth = np.array(
        [
            [
                np.diff(
                    norm(
                        loc=expected_mean(target, 0), scale=expected_std(target, 0)
                    ).cdf(bins)
                ),
                np.diff(
                    norm(
                        loc=expected_mean(target + 1, 0),
                        scale=expected_std(target + 1, 0),
                    ).cdf(bins)
                ),
            ],
            [
                np.diff(
                    norm(
                        loc=expected_mean(target + 1.5, 0),
                        scale=expected_std(target + 1.5, 0),
                    ).cdf(bins)
                ),
                np.diff(
                    norm(
                        loc=expected_mean(target + 2, 0),
                        scale=expected_std(target + 2, 0),
                    ).cdf(bins)
                ),
            ],
        ]
    )
    truth /= truth.sum(axis=-1)[..., np.newaxis]

    res = interp(target)

    assert np.allclose(np.sum(res, axis=-1), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, *bin_contents.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-4, rtol=1e-5)


def test_MomentMorphInterpolator2D(bins, simple_2D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, bin_contents, truth = simple_2D_data.values()

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.isclose(np.sum(res), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator2D_partially_empty(bins, simple_2D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, bin_contents, _ = simple_2D_data.values()

    bin_contents[0, :] = np.zeros(len(bins) - 1)

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.allclose(res, 0)


def test_MomentMorphInterpolator2D_all_empty(bins, simple_2D_data):
    from pyirf.interpolation import MomentMorphInterpolator

    grid, target, _, _ = simple_2D_data.values()
    bin_contents = np.array([np.zeros(len(bins) - 1) for _ in grid])

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.allclose(res, 0)


def test_MomentMorphInterpolator2D_mixed(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20, 20], [60, 20], [40, 60]])
    target = np.array([25, 25])

    # Create template histograms
    bin_contents = np.array(
        [
            [
                [
                    np.diff(
                        norm(
                            loc=expected_mean(x[0], x[1]),
                            scale=expected_std(x[0], x[1]),
                        ).cdf(bins)
                    ),
                    np.diff(
                        norm(
                            loc=expected_mean(x[0] + 1, x[1]),
                            scale=expected_std(x[0] + 1, x[1]),
                        ).cdf(bins)
                    ),
                ],
                [
                    np.diff(
                        norm(
                            loc=expected_mean(x[0] + 1.5, x[1]),
                            scale=expected_std(x[0] + 1.5, x[1]),
                        ).cdf(bins)
                    ),
                    np.diff(
                        norm(
                            loc=expected_mean(x[0] + 2, x[1]),
                            scale=expected_std(x[0] + 2, x[1]),
                        ).cdf(bins)
                    ),
                ],
            ]
            for x in grid
        ]
    )

    bin_contents /= bin_contents.sum(axis=-1)[..., np.newaxis]

    # Make template histograms at indizes [:, 1, 1, :] all zeroed
    bin_contents[:, 1, 1, :] = np.zeros(len(bins) - 1)

    # Zero template histogram at index [1, 0, 0, :]
    bin_contents[1, 0, 0, :] = np.zeros(len(bins) - 1)

    truth = np.array(
        [
            [
                np.diff(
                    norm(
                        loc=expected_mean(target[0], target[1]),
                        scale=expected_std(target[0], target[1]),
                    ).cdf(bins)
                ),
                np.diff(
                    norm(
                        loc=expected_mean(target[0] + 1, target[1]),
                        scale=expected_std(target[0] + 1, target[1]),
                    ).cdf(bins)
                ),
            ],
            [
                np.diff(
                    norm(
                        loc=expected_mean(target[0] + 1.5, target[1]),
                        scale=expected_std(target[0] + 1.5, target[1]),
                    ).cdf(bins)
                ),
                np.diff(
                    norm(
                        loc=expected_mean(target[0] + 2, target[1]),
                        scale=expected_std(target[0] + 2, target[1]),
                    ).cdf(bins)
                ),
            ],
        ]
    )
    truth /= truth.sum(axis=-1)[..., np.newaxis]

    # Expect zeros for at least partially zeroed input templates
    truth[0, 0, :] = np.zeros(len(bins) - 1)
    truth[1, 1, :] = np.zeros(len(bins) - 1)

    interp = MomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    expected_norms = np.array([[0, 1], [1, 0]])
    assert np.allclose(np.sum(res, axis=-1), expected_norms)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, *bin_contents.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator1D_extended_grid(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20], [40], [60], [80]])
    target = np.array([25])
    bin_contents = np.array(
        [
            np.diff(norm(loc=expected_mean(x, 0), scale=expected_std(x, 0)).cdf(bins))
            for x in grid
        ]
    )
    # Assert for probability outside binning
    bin_contents /= bin_contents.sum(axis=1)[:, np.newaxis]

    interp = MomentMorphInterpolator(
        grid_points=grid,
        bin_edges=bins,
        bin_contents=bin_contents,
    )

    res = interp(target)
    truth = np.diff(
        norm(loc=expected_mean(target, 0), scale=expected_std(target, 0)).cdf(bins)
    )
    truth /= truth.sum()

    assert np.isclose(np.sum(res), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator2D_extended_grid(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20, 20], [40, 20], [30, 40], [50, 20], [45, 40]])
    target = np.array([25, 25])
    bin_contents = np.array(
        [
            np.diff(
                norm(loc=expected_mean(x[0], x[1]), scale=expected_std(x[0], x[1])).cdf(
                    bins
                )
            )
            for x in grid
        ]
    )
    # Assert for probability outside binning
    bin_contents /= bin_contents.sum(axis=1)[:, np.newaxis]

    interp = MomentMorphInterpolator(
        grid_points=grid,
        bin_edges=bins,
        bin_contents=bin_contents,
    )

    res = interp(target)
    truth = np.diff(
        norm(
            loc=expected_mean(target[0], target[1]),
            scale=expected_std(target[0], target[1]),
        ).cdf(bins)
    )
    truth /= truth.sum()

    assert np.isclose(np.sum(res), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator2D_extended_grid_extradims(bins):
    from pyirf.interpolation import MomentMorphInterpolator

    grid = np.array([[20, 20], [40, 20], [30, 40], [50, 20], [45, 40]])
    target = np.array([25, 25])
    bin_contents = np.array(
        [
            [
                [
                    np.diff(
                        norm(
                            loc=expected_mean(x[0], x[1]),
                            scale=expected_std(x[0], x[1]),
                        ).cdf(bins)
                    ),
                    np.diff(
                        norm(
                            loc=expected_mean(x[0] + 1, x[1]),
                            scale=expected_std(x[0] + 1, x[1]),
                        ).cdf(bins)
                    ),
                ],
                [
                    np.diff(
                        norm(
                            loc=expected_mean(x[0] + 1.5, x[1]),
                            scale=expected_std(x[0] + 1.5, x[1]),
                        ).cdf(bins)
                    ),
                    np.diff(
                        norm(
                            loc=expected_mean(x[0] + 2, x[1]),
                            scale=expected_std(x[0] + 2, x[1]),
                        ).cdf(bins)
                    ),
                ],
            ]
            for x in grid
        ]
    )

    bin_contents /= bin_contents.sum(axis=-1)[..., np.newaxis]

    interp = MomentMorphInterpolator(
        grid_points=grid,
        bin_edges=bins,
        bin_contents=bin_contents,
    )

    truth = np.array(
        [
            [
                np.diff(
                    norm(
                        loc=expected_mean(target[0], target[1]),
                        scale=expected_std(target[0], target[1]),
                    ).cdf(bins)
                ),
                np.diff(
                    norm(
                        loc=expected_mean(target[0] + 1, target[1]),
                        scale=expected_std(target[0] + 1, target[1]),
                    ).cdf(bins)
                ),
            ],
            [
                np.diff(
                    norm(
                        loc=expected_mean(target[0] + 1.5, target[1]),
                        scale=expected_std(target[0] + 1.5, target[1]),
                    ).cdf(bins)
                ),
                np.diff(
                    norm(
                        loc=expected_mean(target[0] + 2, target[1]),
                        scale=expected_std(target[0] + 2, target[1]),
                    ).cdf(bins)
                ),
            ],
        ]
    )
    truth /= truth.sum(axis=-1)[..., np.newaxis]

    res = interp(target)

    assert np.allclose(np.sum(res, axis=-1), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, *bin_contents.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator3D():
    from pyirf.interpolation import MomentMorphInterpolator

    bins = np.linspace(0, 1, 11)

    grid = np.array([[0, 0, 0], [0, 20, 0], [20, 0, 0], [20, 20, 0], [10, 10, 10]])

    bin_contents = np.array([np.ones(len(bins) - 1) / (len(bins) - 1) for _ in grid])

    with pytest.raises(
        NotImplementedError,
        match="Interpolation in more then two dimension not impemented.",
    ):
        MomentMorphInterpolator(
            grid_points=grid,
            bin_edges=bins,
            bin_contents=bin_contents,
        )
