import numpy as np
import pytest


def test_find_visible_facets():
    """Test for visible facets finding utility"""
    from pyirf.interpolation.visible_edges_extrapolator import find_visible_facets

    grid = np.array([[0, 0], [20, 0], [40, 0], [10, 20], [30, 20], [10, 10]])

    target1 = np.array([[0, -20]])
    res1 = find_visible_facets(grid_points=grid, target_point=target1)

    # From target1 facets spanned by grid point 0 to 1 and 1 to 2 are visible
    assert np.all((res1 == grid[0]) | (res1 == grid[1]) | (res1 == grid[2]))

    target2 = np.array([[20, 30]])
    res2 = find_visible_facets(grid_points=grid, target_point=target2)

    # From target2 only the facet spanned by grid points 4 and 5 is visible
    assert np.all((res2 == grid[4]) | (res2 == grid[5]))

    target3 = np.array([[-10, -20]])
    res3 = find_visible_facets(grid_points=grid, target_point=target3)

    # From target3 again facets spanned by grid point 0 to 1 and 1 to 2 are visible
    assert np.all((res3 == grid[0]) | (res3 == grid[1]) | (res3 == grid[2]))

    target4 = np.array([[-11, -20]])
    res4 = find_visible_facets(grid_points=grid, target_point=target4)

    # From target4 additional the facet spanned by points 0 and 3 becomes visible
    assert np.all(
        (res4 == grid[0]) | (res4 == grid[1]) | (res4 == grid[2]) | (res4 == grid[3])
    )


def test_compute_extrapolation_weights():
    """Test for extrapolation weight computation utility"""
    from pyirf.interpolation.visible_edges_extrapolator import (
        compute_extrapolation_weights,
        find_visible_facets,
    )

    grid = np.array([[0, 0], [20, 0], [40, 0], [10, 20], [30, 20], [10, 10]])

    target1 = np.array([[20, 30]])
    vis_facets1 = find_visible_facets(grid_points=grid, target_point=target1)
    res1 = compute_extrapolation_weights(vis_facets1, target1, m=0)
    # From target 1 only one facet is visible, thus weight should be 1
    np.testing.assert_array_almost_equal(res1, np.array([1]))

    target2 = np.array([[0, -20]])
    # From target 2 two facets are visible, the facet spanned by points 0 and 1 under pi/4,
    # the facet spanned by points 1 and 2 under arctan(2) - pi/4
    vis_facets2 = find_visible_facets(grid_points=grid, target_point=target2)
    expected_m0 = np.array([np.pi / 4, np.arctan(2) - np.pi / 4])
    expected_m0 /= np.sum(expected_m0)

    res2 = compute_extrapolation_weights(vis_facets2, target2, m=0)

    np.testing.assert_array_almost_equal(res2, expected_m0)

    # For m=1 angles are taken to the power of m+1=2
    expected_m1 = np.array([np.pi / 4, np.arctan(2) - np.pi / 4]) ** 2
    expected_m1 /= np.sum(expected_m1)

    res3 = compute_extrapolation_weights(vis_facets2, target2, m=1)

    np.testing.assert_array_almost_equal(res3, expected_m1)


def test_ParametrizedVisibleEdgesExtrapolator_invalid_m():
    """Test to assure errors are raised for invalid values of m (strings, non finite values,
    negative values and non-integer)."""
    from pyirf.interpolation.visible_edges_extrapolator import (
        ParametrizedVisibleEdgesExtrapolator,
    )

    grid = np.array([[0, 0], [20, 0], [40, 0], [10, 20], [30, 20], [10, 10]])

    with pytest.raises(TypeError, match="Only positive integers allowed for m, got a."):
        ParametrizedVisibleEdgesExtrapolator(grid_points=grid, params=grid[:, 0], m="a")

    with pytest.raises(
        ValueError, match="Only positive integers allowed for m, got inf."
    ):
        ParametrizedVisibleEdgesExtrapolator(
            grid_points=grid, params=grid[:, 0], m=np.inf
        )

    with pytest.raises(
        ValueError, match="Only positive integers allowed for m, got -1."
    ):
        ParametrizedVisibleEdgesExtrapolator(grid_points=grid, params=grid[:, 0], m=-1)

    with pytest.raises(
        ValueError, match="Only positive integers allowed for m, got 1.2."
    ):
        ParametrizedVisibleEdgesExtrapolator(grid_points=grid, params=grid[:, 0], m=1.2)


def test_ParametrizedVisibleEdgesExtrapolator_1D_fallback():
    """Test that Extrapolator falls back to Nearest Simplex Interpolation for 1D grids as only one simplex (line segment) can be visible by design."""
    from pyirf.interpolation.nearest_simplex_extrapolator import (
        ParametrizedNearestSimplexExtrapolator,
    )
    from pyirf.interpolation.visible_edges_extrapolator import (
        ParametrizedVisibleEdgesExtrapolator,
    )

    grid = np.array([[0], [20], [40]])

    vis_edge_extrap = ParametrizedVisibleEdgesExtrapolator(
        grid_points=grid, params=grid, m=1
    )
    nearest_simplex_extrap = ParametrizedNearestSimplexExtrapolator(
        grid_points=grid, params=grid
    )

    target = np.array([[-10]])

    assert vis_edge_extrap(target) == nearest_simplex_extrap(target)


def test_ParametrizedVisibleEdgesExtrapolator_2D_fallback():
    """Test that Extrapolator falls back to Nearest Simplex Interpolation for 1´2D grids where only one simplex is be visible."""
    from pyirf.interpolation.nearest_simplex_extrapolator import (
        ParametrizedNearestSimplexExtrapolator,
    )
    from pyirf.interpolation.visible_edges_extrapolator import (
        ParametrizedVisibleEdgesExtrapolator,
    )

    grid = np.array([[0, 0], [20, 0], [40, 0], [10, 20], [30, 20]])
    params = grid[:, 0] + grid[:, 1]

    # from target point, only the simplex spanned by points at indices 1, 3 and 4 is
    # visible. Thus, no blending over visible edges is needed.
    target = np.array([[20, 30]])

    vis_edge_extrap = ParametrizedVisibleEdgesExtrapolator(
        grid_points=grid, params=params, m=1
    )
    nearest_simplex_extrap = ParametrizedNearestSimplexExtrapolator(
        grid_points=grid, params=params
    )

    assert vis_edge_extrap(target) == nearest_simplex_extrap(target)


def test_ParametrizedVisibleEdgeExtrapolator_2D_grid_linear():
    """Test whether results resemble the truth for a linear testcase"""
    from pyirf.interpolation.visible_edges_extrapolator import (
        ParametrizedVisibleEdgesExtrapolator,
    )

    grid_points = np.array([[0, 0], [2, 0], [4, 0], [1, 2], [3, 2]])
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

    extrapolator = ParametrizedVisibleEdgesExtrapolator(
        grid_points=grid_points,
        params=dummy_data,
        m=1,
    )

    target_point = np.array([1, -1])
    extrapolant = extrapolator(target_point)
    dummy_data_target = np.array(
        [
            np.dot((m.T * target_point + n), np.array([1, 1]))
            for m, n in zip(slope, intercept)
        ]
    )[np.newaxis, :]

    np.testing.assert_allclose(extrapolant, dummy_data_target)
    assert extrapolant.shape == (1, *dummy_data.shape[1:])


def test_ParametrizedVisibleEdgeExtrapolator_2D_grid_smoothness():
    """Test whether results are smooth for a non-linear testcase"""
    from pyirf.interpolation.visible_edges_extrapolator import (
        ParametrizedVisibleEdgesExtrapolator,
    )

    grid_points = np.array([[0, 0], [2, 0], [4, 0], [1, 2], [3, 2]])

    slope = np.array([[[0, 1], [1, 1]], [[0, 2], [2, 3]], [[3, 0], [3, 5]]])
    intercept = np.array([[[0, 1], [1, 1]], [[0, -1], [-1, -1]], [[10, 11], [11, 11]]])

    # Create 3 times 2 linear samples, each of the form (mx * px + my * py + nx + ny)
    # with slope m, intercept n at each grid_point p
    dummy_data = np.array(
        [
            np.array(
                [
                    np.dot((m.T * p + n), np.array([1, 1])) + p[0] * p[1]
                    for m, n in zip(slope, intercept)
                ]
            ).squeeze()
            for p in grid_points
        ]
    )

    extrapolator = ParametrizedVisibleEdgesExtrapolator(
        grid_points=grid_points,
        params=dummy_data,
        m=1,
    )

    # The horizontal line at x=2 seperates two domains where the nerarest simplex
    # changes. If the Extrapolator works correctly, the transition should be smooth.
    target_point_left = np.array([1.9999999, -1])
    target_point_right = np.array([2.00000001, -1])

    extrapolant_left = extrapolator(target_point_left)
    extrapolant_right = extrapolator(target_point_right)

    np.testing.assert_allclose(extrapolant_left, extrapolant_right)
