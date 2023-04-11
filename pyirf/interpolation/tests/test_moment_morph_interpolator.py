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


def test_Base1DMomentMorphInterpolator_oversized_grid(bins):
    from pyirf.interpolation import Base1DMomentMorphInterpolator

    full_grid = np.array([[20], [40], [60]])
    bin_contents_full_grid = np.array(
        [
            np.diff(norm(loc=expected_mean(x, 0), scale=expected_std(x, 0)).cdf(bins))
            for x in full_grid
        ]
    )

    with pytest.raises(
        ValueError, match="This base class can only interpolate between two points."
    ):
        Base1DMomentMorphInterpolator(
            grid_points=full_grid, bin_edges=bins, bin_contents=bin_contents_full_grid
        )


def test_Base1DMomentMorphInterpolator(bins, simple_1D_data):
    from pyirf.interpolation import Base1DMomentMorphInterpolator

    grid, target, bin_contents, truth = simple_1D_data.values()

    interp = Base1DMomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.isclose(np.sum(res), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_Base1DMomentMorphInterpolator_all_empty(bins, simple_1D_data):
    from pyirf.interpolation import Base1DMomentMorphInterpolator

    grid, target, _, _ = simple_1D_data.values()
    bin_contents = np.array([np.zeros(len(bins) - 1) for _ in grid])

    interp = Base1DMomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.allclose(res, 0)


def test_Base1DMomentMorphInterpolator_partially_empty(bins, simple_1D_data):
    from pyirf.interpolation import Base1DMomentMorphInterpolator

    grid, target, bin_contents, _ = simple_1D_data.values()

    bin_contents[0, :] = np.zeros(len(bins) - 1)

    interp = Base1DMomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.allclose(res, 0)


def test_Base1DMomentMorphInterpolator_mixed():
    from pyirf.interpolation import Base1DMomentMorphInterpolator

    bins = np.linspace(-10, 40, 1001)

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

    interp = Base1DMomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    expected_norms = np.array([[0, 1], [1, 0]])
    assert np.allclose(np.sum(res, axis=-1), expected_norms)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, *bin_contents.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_Base2DTriangularMomentMorphInterpolator_oversized_grid(bins):
    from pyirf.interpolation import Base2DTriangularMomentMorphInterpolator

    full_grid = np.array([[20, 20], [40, 20], [30, 40], [50, 20], [45, 40]])
    bin_contents_full_grid = np.array(
        [
            np.diff(
                norm(loc=expected_mean(x[0], x[1]), scale=expected_std(x[0], x[1])).cdf(
                    bins
                )
            )
            for x in full_grid
        ]
    )
    # Assert for probability outside binning
    bin_contents_full_grid /= bin_contents_full_grid.sum(axis=1)[:, np.newaxis]

    with pytest.raises(
        ValueError, match="This base class can only interpolate in a triangle."
    ):
        Base2DTriangularMomentMorphInterpolator(
            grid_points=full_grid, bin_edges=bins, bin_contents=bin_contents_full_grid
        )


def test_Base2DTriangularMomentMorphInterpolator(bins, simple_2D_data):
    from pyirf.interpolation import Base2DTriangularMomentMorphInterpolator

    grid, target, bin_contents, truth = simple_2D_data.values()

    interp = Base2DTriangularMomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.isclose(np.sum(res), 1)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, len(bins) - 1)
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_Base2DTriangularMomentMorphInterpolator_partially_empty(bins, simple_2D_data):
    from pyirf.interpolation import Base2DTriangularMomentMorphInterpolator

    grid, target, bin_contents, _ = simple_2D_data.values()

    bin_contents[0, :] = np.zeros(len(bins) - 1)

    interp = Base2DTriangularMomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.allclose(res, 0)


def test_Base2DTriangularMomentMorphInterpolator_all_empty(bins, simple_2D_data):
    from pyirf.interpolation import Base2DTriangularMomentMorphInterpolator

    grid, target, _, _ = simple_2D_data.values()
    bin_contents = np.array([np.zeros(len(bins) - 1) for _ in grid])

    interp = Base2DTriangularMomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    assert np.allclose(res, 0)


def test_Base2DTriangularMomentMorphInterpolator_mixed(bins):
    from pyirf.interpolation import Base2DTriangularMomentMorphInterpolator

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

    interp = Base2DTriangularMomentMorphInterpolator(
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents
    )

    res = interp(target)

    expected_norms = np.array([[0, 1], [1, 0]])
    assert np.allclose(np.sum(res, axis=-1), expected_norms)
    assert np.all(np.isfinite(res))
    assert res.shape == (1, *bin_contents.shape[1:])
    # Assert truth and result matching within +- 0.1%, atol dominates comparison
    assert np.allclose(res.squeeze(), truth, atol=1e-3, rtol=1e-5)


def test_MomentMorphInterpolator_simple_1DGrid(bins):
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
        axis=-1,
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


def test_MomentMorphInterpolator_extended_1DGrid(bins):
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
        grid_points=grid, bin_edges=bins, bin_contents=bin_contents, axis=-1
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


def test_MomentMorphInterpolator_simple_2DGrid(bins):
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
        axis=-1,
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


def test_MomentMorphInterpolator_extended_2DGrid(bins):
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
        axis=-1,
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


def test_MomentMorphInterpolator_3D_Grid():
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
            axis=-1,
        )


def test_Base1DMomentMorphInterpolator_dirac_delta_input():
    from pyirf.interpolation import Base1DMomentMorphInterpolator

    grid = np.array([[1], [3]])
    bin_edges = np.array([0, 1, 2, 3, 4])
    bin_contents = np.array(
        [
            [[0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25]],
            [[0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25]],
        ]
    )
    target = np.array([2])

    interp = Base1DMomentMorphInterpolator(grid, bin_edges, bin_contents)
    res = interp(target)

    assert np.allclose(res, np.array([[0, 0, 1, 0], [0.25, 0.25, 0.25, 0.25]]))
