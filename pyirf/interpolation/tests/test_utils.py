import numpy as np
import pytest
from scipy.spatial import Delaunay


@pytest.fixture
def non_rect_grid():
    grid = np.array([[0, 0], [10, 20], [30, 20], [20, 0], [40, 0]])

    return Delaunay(grid)


def test_plumb_point_distance():
    """Test line-segment to point distance computation"""
    from pyirf.interpolation.utils import plumb_point_dist

    # Test vertical line
    line = np.array([[0, 0], [0, 1]])

    # Plumb point between end-points
    assert plumb_point_dist(line, np.array([-1, 0.5])) == 1
    assert plumb_point_dist(line, np.array([-0.7, 0.25])) == 0.7

    # Plumb point on one end-point
    assert plumb_point_dist(line, np.array([0, 2.1])) == 1.1
    assert plumb_point_dist(line, np.array([0, -1])) == 1

    # Plumb point not between end-points, nearest point is (0, 0)
    assert plumb_point_dist(line, np.array([-1, -1])) == np.sqrt(2)
    # Nearest point is (0, 1)
    assert plumb_point_dist(line, np.array([3, 3])) == np.sqrt(13)

    # Test horzontal line
    line = np.array([[0, 0], [1, 0]])

    # Plumb point between end-points
    assert plumb_point_dist(line, np.array([0.5, 0.5])) == 0.5

    # Plumb point in extention of line, nearest point is (1, 0)
    assert plumb_point_dist(line, np.array([2, 0])) == 1

    # Plumb point on end point
    assert plumb_point_dist(line, np.array([1, 1])) == 1

    # Nearest point is (0, 0)
    assert plumb_point_dist(line, np.array([-1, -1])) == np.sqrt(2)

    # Test arbitrary line
    line = np.array([[1, 1], [-1, -1]])
    # isclose needed here, as there is a small numerical deviation
    # of +/- eps in this case. Plumb point between end-points
    assert np.isclose(plumb_point_dist(line, np.array([-1, 1])), np.sqrt(2))

    # Nearest point is (-1, -1)
    assert plumb_point_dist(line, np.array([-2, -3])) == np.sqrt(5)


def test_point_facet_angle():
    """Test angle computation in triangle, function should return cos(angle)"""
    from pyirf.interpolation.utils import point_facet_angle

    line = np.array([[0, 0], [0, 1]])

    assert np.isclose(
        point_facet_angle(line, np.array([1, 0])), np.cos(45 * np.pi / 180)
    )
    assert np.isclose(
        point_facet_angle(line, np.array([-1, 0])), np.cos(45 * np.pi / 180)
    )
    # these points build a same-side triangle
    assert np.isclose(
        point_facet_angle(line, np.array([np.sqrt(3) / 2, 0.5])),
        np.cos(60 * np.pi / 180),
    )


def test_find_nearest_facet_rect_grid():
    """Test nearest facet finding on rectanguar grid"""
    from pyirf.interpolation.utils import find_nearest_facet

    rect_grid = Delaunay(
        np.array([[0, 0], [0, 20], [20, 20], [20, 0], [40, 0], [40, 20]])
    )
    qhull_points = rect_grid.points[rect_grid.convex_hull]

    nearest_facet_ind = find_nearest_facet(qhull_points, np.array([10, -5]))
    assert np.logical_or(
        np.array_equal(qhull_points[nearest_facet_ind], np.array([[0, 0], [20, 0]])),
        np.array_equal(qhull_points[nearest_facet_ind], np.array([[20, 0], [0, 0]])),
    )

    nearest_facet_ind = find_nearest_facet(qhull_points, np.array([45, 15]))
    assert np.logical_or(
        np.array_equal(qhull_points[nearest_facet_ind], np.array([[40, 0], [40, 20]])),
        np.array_equal(qhull_points[nearest_facet_ind], np.array([[40, 20], [40, 0]])),
    )

    nearest_facet_ind = find_nearest_facet(qhull_points, np.array([-10, -1]))
    assert np.logical_or(
        np.array_equal(qhull_points[nearest_facet_ind], np.array([[0, 0], [0, 20]])),
        np.array_equal(qhull_points[nearest_facet_ind], np.array([[0, 20], [0, 0]])),
    )


def test_find_nearest_facet_non_rect_grid(non_rect_grid):
    """Test nearest facet finding on a non rectanguar grid to catch some more cases"""
    from pyirf.interpolation.utils import find_nearest_facet

    qhull_points = non_rect_grid.points[non_rect_grid.convex_hull]

    nearest_facet_ind = find_nearest_facet(qhull_points, np.array([5, 20]))
    assert np.logical_or(
        np.array_equal(qhull_points[nearest_facet_ind], np.array([[0, 0], [10, 20]])),
        np.array_equal(qhull_points[nearest_facet_ind], np.array([[10, 20], [0, 0]])),
    )

    nearest_facet_ind = find_nearest_facet(qhull_points, np.array([35, 30]))
    assert np.logical_or(
        np.array_equal(qhull_points[nearest_facet_ind], np.array([[10, 20], [30, 20]])),
        np.array_equal(qhull_points[nearest_facet_ind], np.array([[30, 20], [10, 20]])),
    )


def test_find_simplex_to_facet(non_rect_grid):
    """
    Test facet-to-simplex finding on non rectangular grid, as the triangulation
    is clear in this case as it is build from left to right and not ambiguous.
    For the rectangular grid used above two triangulations exist.
    """
    from pyirf.interpolation.utils import find_simplex_to_facet

    simplices_points = non_rect_grid.points[non_rect_grid.simplices]

    assert find_simplex_to_facet(simplices_points, np.array([[0, 0], [0, 20]])) == 0
    assert find_simplex_to_facet(simplices_points, np.array([[10, 20], [30, 20]])) == 1
    assert find_simplex_to_facet(simplices_points, np.array([[30, 20], [40, 0]])) == 2


def test_find_nearest_simplex(non_rect_grid):
    """
    Test whole nearest simplex finding on non rectangular grid, as the triangulation
    is clear in this case as it is build from left to right and not ambiguous.
    For the rectangular grid used above two triangulations exist.
    """
    from pyirf.interpolation.utils import find_nearest_simplex

    assert find_nearest_simplex(non_rect_grid, np.array([-10, -10])) == 0
    assert find_nearest_simplex(non_rect_grid, np.array([10, 30])) == 1
    assert find_nearest_simplex(non_rect_grid, np.array([20.00000000001, -10])) == 2


def test_get_bin_width():
    from pyirf.interpolation.utils import get_bin_width
    from pyirf.interpolation import PDFNormalization

    bins = np.array([0, 1, 3])
    np.testing.assert_allclose(get_bin_width(bins, PDFNormalization.AREA), [1, 2])

    bins = np.array([0, np.pi / 3, np.pi / 2])
    width = get_bin_width(bins, PDFNormalization.CONE_SOLID_ANGLE)
    np.testing.assert_allclose(width, [np.pi, np.pi])
