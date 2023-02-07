"""Tests for base interpolator classes"""
import numpy as np


def test_GridDataInterpolator_1DGrid():
    """Test checks performed to enforce correct data structure"""
    from pyirf.interpolators import GridDataInterpolator

    dummy_data_dtype = [("mu", "<f4"), ("sigma", "<f4")]
    grid_points = np.array([[0], [1]])
    target_point = np.array([[0.5]])

    dummy_data1 = np.array(
        [[(0, 1), (1, 1)], [(0, 2), (2, 3)], [(0, 3), (3, 5)]], dtype=dummy_data_dtype
    )
    dummy_data2 = np.array(
        [[(0, 2), (2, 2)], [(0, 4), (4, 6)], [(0, 6), (6, 10)]], dtype=dummy_data_dtype
    )

    Interpolator = GridDataInterpolator(
        grid_points, params=np.array([dummy_data1, dummy_data2])
    )
    interpolant = Interpolator(target_point)

    dummy_data_target = np.array(
        [[(0, 1.5), (1.5, 1.5)], [(0, 3), (3, 4.5)], [(0, 4.5), (4.5, 7.5)]],
        dtype=dummy_data_dtype,
    )

    assert interpolant.dtype == dummy_data_target.dtype

    for param_name in dummy_data1.dtype.names:
        assert np.allclose(interpolant[param_name], dummy_data_target[param_name])


def test_GridDataInterpolator_2DGrid():
    """Test checks performed to enforce correct data structure"""
    from pyirf.interpolators import GridDataInterpolator

    dummy_data_dtype = [("mu", "<f4"), ("sigma", "<f4")]
    grid_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_point = np.array([[0.5, 0.5]])

    dummy_data1 = np.array(
        [[(0, 1), (1, 1)], [(0, 2), (2, 3)], [(0, 3), (3, 5)]], dtype=dummy_data_dtype
    )
    dummy_data2 = np.array(
        [[(0, 2), (1, 2)], [(0, 4), (2, 6)], [(0, 6), (3, 10)]], dtype=dummy_data_dtype
    )
    dummy_data3 = np.array(
        [[(0, 1), (2, 1)], [(0, 2), (4, 3)], [(0, 3), (6, 5)]], dtype=dummy_data_dtype
    )
    dummy_data4 = np.array(
        [[(0, 2), (2, 2)], [(0, 4), (4, 6)], [(0, 6), (6, 10)]], dtype=dummy_data_dtype
    )

    Interpolator = GridDataInterpolator(
        grid_points,
        params=np.array([dummy_data1, dummy_data2, dummy_data3, dummy_data4]),
    )
    interpolant = Interpolator(target_point)

    dummy_data_target = np.array(
        [[(0, 1.5), (1.5, 1.5)], [(0, 3), (3, 4.5)], [(0, 4.5), (4.5, 7.5)]],
        dtype=dummy_data_dtype,
    )

    assert interpolant.dtype == dummy_data_target.dtype

    for param_name in dummy_data1.dtype.names:
        assert np.allclose(interpolant[param_name], dummy_data_target[param_name])
