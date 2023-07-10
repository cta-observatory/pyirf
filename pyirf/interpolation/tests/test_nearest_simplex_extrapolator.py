import numpy as np
import pytest


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
    """Test ParametrizedNearestSimplexExtrapolator on a 3D Grid. which is currently
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
