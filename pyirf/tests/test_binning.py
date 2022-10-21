import astropy.units as u
from astropy.table import QTable
import numpy as np
import pyirf.binning as binning


def test_add_overflow_bins():
    from pyirf.binning import add_overflow_bins

    bins = np.array([1, 2])
    bins_uo = add_overflow_bins(bins)

    assert len(bins_uo) == 4
    assert bins_uo[0] == 0
    assert np.isinf(bins_uo[-1])
    assert np.all(bins_uo[1:-1] == bins)

    bins = np.array([1, 2])
    bins_uo = add_overflow_bins(bins, positive=False)

    assert bins_uo[0] < 0
    assert np.isinf(bins_uo[0])

    # make sure we don't any bins if over / under is already present
    bins = np.array([0, 1, 2, np.inf])
    assert len(add_overflow_bins(bins)) == len(bins)


def test_add_overflow_bins_units():
    from pyirf.binning import add_overflow_bins

    bins = np.array([1, 2]) * u.TeV
    bins_uo = add_overflow_bins(bins)

    assert bins_uo.unit == bins.unit
    assert len(bins_uo) == 4
    assert bins_uo[0] == 0 * u.TeV
    assert np.isinf(bins_uo[-1])
    assert np.all(bins_uo[1:-1] == bins)


def test_bins_per_decade():
    from pyirf.binning import create_bins_per_decade

    bins = create_bins_per_decade(100 * u.GeV, 100 * u.TeV)

    assert bins.unit == u.GeV
    assert len(bins) == 16  # end inclusive if exactly fits per_decade

    assert bins[0] == 100 * u.GeV
    assert np.allclose(np.diff(np.log10(bins.to_value(u.GeV))), 0.2)

    bins = create_bins_per_decade(100 * u.GeV, 100 * u.TeV, 10)
    assert bins.unit == u.GeV
    assert len(bins) == 31  # end inclusive since it fits exactly

    assert bins[0] == 100 * u.GeV
    assert np.allclose(np.diff(np.log10(bins.to_value(u.GeV))), 0.1)

    bins = create_bins_per_decade(100 * u.GeV, 105 * u.TeV, 5)
    assert bins.unit == u.GeV
    assert len(bins) == 16  # end non-inclusive

    assert u.isclose(bins[-1], 100 * u.TeV) # last value at correct difference
    assert np.allclose(np.diff(np.log10(bins.to_value(u.GeV))), 0.2)


def test_create_histogram_table():
    '''Test create histogram table'''
    from pyirf.binning import create_histogram_table

    events = QTable({
        'reco_energy': [1, 1, 10, 100, 100, 100] * u.TeV,
    })
    bins = [0.5, 5, 50, 500] * u.TeV

    # test without weights
    hist = create_histogram_table(events, bins, key='reco_energy')
    assert np.all(hist['n'] == [2, 1, 3])
    assert np.all(hist['n_weighted'] == [2, 1, 3])

    # test with weights
    events['weight'] = [0.5, 2, 2.5, 0.5, 0.2, 0.3]
    hist = create_histogram_table(events, bins, key='reco_energy')
    assert np.all(hist['n'] == [2, 1, 3])
    assert np.all(hist['n_weighted'] == [2.5, 2.5, 1.0])

    # test with particle_types
    types = np.array(['gamma', 'electron', 'gamma', 'electron', 'gamma', 'proton'])
    events['particle_type'] = types
    hist = create_histogram_table(events, bins, key='reco_energy')

    assert np.all(hist['n'] == [2, 1, 3])
    assert np.all(hist['n_weighted'] == [2.5, 2.5, 1.0])

    assert np.all(hist['n_gamma'] == [1, 1, 1])
    assert np.all(hist['n_electron'] == [1, 0, 1])
    assert np.all(hist['n_proton'] == [0, 0, 1])

    assert np.allclose(hist['n_gamma_weighted'], [0.5, 2.5, 0.2])
    assert np.allclose(hist['n_electron_weighted'], [2, 0, 0.5])
    assert np.allclose(hist['n_proton_weighted'], [0, 0, 0.3])

    # test with empty table
    empty = events[:0]
    hist = create_histogram_table(empty, bins, key='reco_energy')
    zeros = np.zeros(3)
    assert np.all(hist['n'] == zeros)
    assert np.all(hist['n_weighted'] == zeros)


def test_calculate_bin_indices():
    from pyirf.binning import calculate_bin_indices, OVERFLOW_INDEX, UNDERFLOW_INDEX

    bins = np.array([0, 1, 2])
    values = [0.5, 0.5, 1, 1.1, 1.9, 2, -1, 2.5]

    true_idx = np.array([0, 0, 1, 1, 1, 2, -1, 2])
    true_valid = np.array([True, True, True, True, True, False, False, False])
    true_idx = np.array([0, 0, 1, 1, 1, OVERFLOW_INDEX, UNDERFLOW_INDEX, OVERFLOW_INDEX])

    idx, valid = calculate_bin_indices(values, bins)
    assert np.all(idx == true_idx)
    assert np.all(valid == true_valid)

    # test with units
    bins *= u.TeV
    values *= 1000 * u.GeV
    idx, valid = calculate_bin_indices(values, bins)
    assert np.all(idx == true_idx)
    assert np.all(valid == true_valid)


def test_resample_bins_1d():
    from pyirf.binning import resample_histogram1d

    n = 10
    data = np.ones(n)
    edges = np.arange(n + 1)

    # no resampling
    new_data = resample_histogram1d(data, edges, edges)
    np.testing.assert_array_almost_equal(new_data, data)

    # resampling, less bins
    new_edges = np.arange(n + 1, step=2)
    true_new_data = np.ones(n // 2) * 2
    new_data = resample_histogram1d(data, edges, new_edges)
    np.testing.assert_array_almost_equal(new_data, true_new_data)

    # resampling, smaller range (lose normalization)
    new_edges = np.arange(n // 2 + 1)
    true_new_data = np.ones(n // 2)
    new_data = resample_histogram1d(data, edges, new_edges)
    np.testing.assert_array_almost_equal(new_data, true_new_data)

    # resampling, larger range (fill with zeros)
    new_edges = np.arange(-n, 2 * n + 1)
    true_new_data = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n)])
    new_data = resample_histogram1d(data, edges, new_edges)
    np.testing.assert_array_almost_equal(new_data, true_new_data)


def test_resample_bins_nd():
    from pyirf.binning import resample_histogram1d
    n = 10

    # shape 2x3x10
    data = np.array([
        [
            np.ones(n),
            np.ones(n) * 2,
            np.ones(n) * 3,
        ],
        [
            np.ones(n) * 4,
            np.ones(n) * 5,
            np.ones(n) * 6,
        ],
    ])

    edges = np.arange(n + 1)
    new_data = resample_histogram1d(data, edges, edges, axis=2)
    np.testing.assert_array_almost_equal(new_data, data)

    edges = np.arange(3 + 1)
    new_data = resample_histogram1d(data, edges, edges, axis=1)
    np.testing.assert_array_almost_equal(new_data, data)


def test_resample_bins_units():
    from pyirf.binning import resample_histogram1d
    import pytest

    n = 10
    data = np.ones(n) * u.m

    old_edges = np.arange(n + 1) * u.TeV
    new_edges = old_edges.copy().to(u.GeV)

    new_data = resample_histogram1d(data, old_edges, new_edges)

    np.testing.assert_array_almost_equal(
        new_data.to_value(data.unit), data.to_value(data.unit),
    )

    with pytest.raises(ValueError):
        new_data = resample_histogram1d(data, old_edges, new_edges.value)

    with pytest.raises(u.core.UnitConversionError):
        new_data = resample_histogram1d(data, old_edges, u.Quantity(new_edges.value, unit=u.m))


def test_join_bin_lo_hi():
    """Test join_bin_hi_lo function."""
    bins = np.array([np.logspace(-1,3, 20)*u.TeV])
    bins_lo = bins[:,:-1]
    bins_hi = bins[:,1:]

    bins_joint = binning.join_bin_lo_hi(bins_lo,bins_hi)
    assert np.allclose(bins_joint, bins, rtol=1.e-5)


def test_split_bin_lo_hi():
    """Test split_bin_lo_hi function."""
    bins = np.array([np.logspace(-1,3, 20)*u.TeV])
    bins_lo_true = bins[:,:-1]
    bins_hi_true = bins[:,1:]

    bin_lo, bin_hi = binning.split_bin_lo_hi(bins)
    assert np.allclose(bin_lo, bins_lo_true, rtol=1.e-5)
    assert np.allclose(bin_hi, bins_hi_true, rtol=1.e-5)
