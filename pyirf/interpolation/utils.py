import astropy.units as u
import numpy as np

from ..utils import cone_solid_angle
from .base_interpolators import PDFNormalization


def get_bin_width(bin_edges, normalization):
    if normalization is PDFNormalization.AREA:
        return np.diff(bin_edges)

    if normalization is PDFNormalization.CONE_SOLID_ANGLE:
        return np.diff(cone_solid_angle(bin_edges).to_value(u.sr))

    raise ValueError(f"Invalid PDF normalization: {normalization}")


def plumb_point_dist(line, target):
    """
    Compute minimal distance between target and line under the constraint, that it has
    to lay between the points building the line and not on the extension of it.

    Parameters
    ----------
    line: np.ndarray, shape=(2, M)
        Array of two points spanning a line segment. Might be in two or three dims M.
    target: np.ndarray, shape=(M)
        Target point, of which the minimal distance to line segement is needed

    Returns
    -------
    d_min: float
        Minimal distance to line segement between points in line
    """
    A = line[0]
    B = line[1]
    P = target

    # Costruct the footpoint/plumb point of the target projected onto
    # both lines OA + r1*AB (F1) and OB + r1*BA (F2), for details see
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
    F1 = A + np.dot(P - A, B - A) * (B - A) / np.dot(B - A, B - A)
    F2 = B + np.dot(P - B, A - B) * (A - B) / np.dot(A - B, A - B)

    # Find, at which parameter value r1/r2 the plumb point lies on line F1/F2
    if B[0] - A[0] == 0:
        r1 = (F1[1] - A[1]) / (B[1] - A[1])
    else:
        r1 = (F1[0] - A[0]) / (B[0] - A[0])

    if A[0] - B[0] == 0:
        r2 = (F2[1] - B[1]) / (A[1] - B[1])
    else:
        r2 = (F2[0] - B[0]) / (A[0] - B[0])

    # If |r1| + |r2| == 1, the plomb point lies between A and B, thus the
    # distance is the seareched one. In cases where the plumb point is A or B
    # use the second method consistently, as there might be +/- eps differences
    # due to different methods of computation.
    if np.isclose(np.abs(r1) + np.abs(r2), 1) and not (
        np.isclose(r1, 0) or np.isclose(r2, 0)
    ):
        # Compute distance of plumb point to line, for details see
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_vector_formulation
        return np.linalg.norm(np.cross(P - A, B - A)) / np.linalg.norm(B - A)
    # If not, the nearest point A <= x <= B is one A and B, thus the searched distance
    # is the one to this nearest point
    else:
        return np.min(np.array([np.linalg.norm(A - P), np.linalg.norm(B - P)]))


def point_facet_angle(line, target):
    """
    Compute cos(angle) between target and a line segment"

    Parameters
    ----------
    line: np.ndarray, shape=(2, M)
        Array of two points spanning a line segment. Might be in two or three dims M.
    target: np.ndarray, shape=(M)
        Target point, of which the angle is needed.

    Returns
    -------
    cos_angle: float
        Cosine of angle at target in the triangle ABT with line-points A and B and target T.
    """
    PB = line[1] - target
    PA = line[0] - target

    # For details see https://en.wikipedia.org/wiki/Angle#Dot_product_and_generalisations
    return np.dot(PB, PA) / (np.linalg.norm(PA) * np.linalg.norm(PB))


def find_nearest_facet(qhull_points, target):
    """
    Search nearest facet by looking for the closest point on a facet. If this point is an edge point
    (which by definition lays on two facets) use the one with the lowest cos of angle
    (maximising angle) as fallback

    Parameters
    ----------
    qhull_points: np.ndarray, shape=(n_facets, 2, 2)
        Array containting both points building a facet for all facets in a
        grids convex hull.
    target: np.ndarray, shape=(2)
        Target point, of which the nearest facet on the convex hull is wanted.

    Returns
    -------
    nearest_facet_ind: int
        Index of the nearest facet in qhull_points.
    """
    plumbs = np.array(
        [
            (plumb_point_dist(line, target), point_facet_angle(line, target))
            for line in qhull_points
        ],
        dtype=[("plumb_dist", "<f16"), ("facet_point_angle", "<f16")],
    )

    return np.argsort(plumbs, order=["plumb_dist", "facet_point_angle"])[0]


def find_simplex_to_facet(simplices_points, facet_points):
    """
    Find simplex of triangulation corresponding to one facet of the convex hull
    utilizing that each facet is only part of one triangulation simplex

    Parameters
    ----------
    simplices_points: np.ndarray, shape=(n_simplices, 3, 2)
        Array containting all three points building a triangluation simplex for all
        simplices in a grid's triangulation.
    facet_points: np.ndarray, shape=(2, 2)
        Points of which the containing simplex is wanted.

    Returns
    -------
    simplex_ind: int
        Index of the simplex corresponding to a facet spanned by facet_points.
    """
    # A qhull facet can only be part of one simplex, both end-points have to be part
    # of the points spanning the simplex
    lookup = np.logical_or(
        np.all(simplices_points == facet_points[0], axis=-1),
        np.all(simplices_points == facet_points[1], axis=-1),
    )

    # There is only one simplex, where both points are part of, thus sum(lookup) = 2
    return np.argmax(np.sum(lookup, axis=-1))


def find_nearest_simplex(triangulation, target):
    """
    Find nearest simplex to target by finding the nearest convex hull facet and
    the corresponding simplex

    Parameters
    ----------
    triangulation: scipy.spatial.Delaunay object
        Delaunay triangulation of an input grid, shapes of original grid has to match
        target's one.
    target: np.ndarray, shape=(2)
        Target point, of which the nearest facet on the convex hull is wanted.
        Has to lay outside of the grid for a usable result, which is not checked here.

    Returns
    -------
    simplex_ind: int
        Index of the nearest simplex to a target point.
    """
    qhull = triangulation.convex_hull
    qhull_points = triangulation.points[qhull]

    nearest_facet_ind = find_nearest_facet(qhull_points, target)
    facet_points = qhull_points[nearest_facet_ind]

    simplices_points = triangulation.points[triangulation.simplices]

    return find_simplex_to_facet(simplices_points, facet_points)
