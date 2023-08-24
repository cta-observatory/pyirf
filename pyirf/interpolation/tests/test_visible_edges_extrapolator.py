import numpy as np


def test_find_visible_facets():
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
