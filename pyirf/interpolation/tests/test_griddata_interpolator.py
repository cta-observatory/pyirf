"""Tests for GridDataInterpolator"""
import numpy as np


def test_GridDataInterpolator_1DGrid():
    """Test GridDataInterpolator on 1D Grid"""
    from pyirf.interpolation import GridDataInterpolator

    grid_points = np.array([[0], [1]])
    target_point = np.array([[0.5]])

    dummy_data1 = np.array([[[0, 1], [1, 1]], [[0, 2], [2, 3]], [[0, 3], [3, 5]]])
    dummy_data2 = np.array([[[0, 2], [2, 2]], [[0, 4], [4, 6]], [[0, 6], [6, 10]]])

    dummy_data = np.array([dummy_data1, dummy_data2])

    interpolator = GridDataInterpolator(
        grid_points=grid_points, params=dummy_data, method="linear"
    )
    interpolant = interpolator(target_point)

    dummy_data_target = 1.5 * dummy_data1

    assert np.allclose(interpolant, dummy_data_target)
    assert interpolant.shape == (1, *dummy_data.shape[1:])


def test_GridDataInterpolator_2DGrid():
    """Test GridDataInterpolator on 2D Grid"""
    from pyirf.interpolation import GridDataInterpolator

    grid_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_point = np.array([[0.5, 0.5]])

    dummy_data1 = np.array([[[0, 1], [1, 1]], [[0, 2], [2, 3]], [[0, 3], [3, 5]]])
    dummy_data2 = np.array([[[0, 2], [1, 2]], [[0, 4], [2, 6]], [[0, 6], [3, 10]]])
    dummy_data3 = np.array([[[0, 1], [2, 1]], [[0, 2], [4, 3]], [[0, 3], [6, 5]]])
    dummy_data4 = np.array([[[0, 2], [2, 2]], [[0, 4], [4, 6]], [[0, 6], [6, 10]]])

    dummy_data = np.array([dummy_data1, dummy_data2, dummy_data3, dummy_data4])

    interpolator = GridDataInterpolator(
        grid_points=grid_points, params=dummy_data, method="linear"
    )
    interpolant = interpolator(target_point)

    dummy_data_target = 1.5 * dummy_data1

    assert np.allclose(interpolant, dummy_data_target)
    assert interpolant.shape == (1, *dummy_data.shape[1:])
